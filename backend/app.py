import os
import re
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaModel, pipeline
import warnings

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
codebert = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
codebert.eval()

# Initialize code generation pipeline
code_generator = pipeline(
    "text-generation",
    model="Salesforce/codegen-350M-mono",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=150
)

class SiameseNetwork(torch.nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x1, x2):
        combined = torch.cat([torch.abs(x1 - x2), x1 * x2], dim=1)
        return self.fc(combined)

# Initialize and load Siamese model
siamese_model = SiameseNetwork().to(device)
try:
    siamese_model.load_state_dict(
        torch.load("E:/ai-code-similarity/backend/siamese_plagiarism_model.pt", 
                map_location=device)
    )
    print('Siamese model loaded successfully')
except Exception as e:
    print(f'Error loading model weights: {e}')
    # Initialize with random weights if loading fails
    for layer in siamese_model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

siamese_model.eval()

def preprocess_code(code):
    """Basic code preprocessing"""
    if not code or not isinstance(code, str):
        return ""
    # Remove comments
    code = re.sub(r'//.*', '', code)  # Remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove multi-line comments
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    return code

def get_embedding(code):
    """Get CodeBERT embedding for a code snippet"""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = codebert(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def generate_code(question):
    """Generate reference code using LLM"""
    try:
        prompt = f"# CPP implementation of: {question}\n"
        result = code_generator(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Extract code from generated text
        generated_text = result[0]['generated_text']
        if '```' in generated_text:
            # Extract code block if present
            code_block = generated_text.split('```')[1]
            if code_block.startswith('Cpp'):
                code_block = code_block[6:]
            return code_block.strip()
        return generated_text.replace(prompt, '').strip()
    except Exception as e:
        print(f"Code generation failed: {str(e)}")
        return "def solution():\n    # Code generation failed"

@app.route('/generate_and_compare', methods=['POST'])
def generate_and_compare():
    try:
        data = request.json
        question = data.get('question', '')
        user_code = data.get('code', '')
        
        if not question or not user_code:
            return jsonify({"error": "Question and code are required"}), 400
        
        print(f"Generating reference code for: {question}")
        reference_code = generate_code(question)
        print(f"Generated reference code:\n{reference_code}")
        
        # Preprocess both code snippets
        clean_user_code = preprocess_code(user_code)
        clean_reference_code = preprocess_code(reference_code)
        
        # Get embeddings
        emb_user = get_embedding(clean_user_code)
        emb_ref = get_embedding(clean_reference_code)
        
        # Calculate similarity
        with torch.no_grad():
            similarity_score = siamese_model(emb_user, emb_ref).item()
        
        return jsonify({
            'similarity_score': similarity_score,
            'prediction': 'similar' if similarity_score >= 0.5 else 'not similar',
            'confidence': f'{min(100, int(similarity_score * 100))}%',
            'reference_code': reference_code
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "codebert-siamese",
        "device": str(device)
    })

if __name__ == '__main__':
    # Warm-up models
    dummy_code = "int main() { return 0; }"
    _ = get_embedding(dummy_code)
    _ = generate_code("Print hello world")
    print(f"Models warmed up on {device}")
    app.run(host='0.0.0.0', port=5000, debug=True)