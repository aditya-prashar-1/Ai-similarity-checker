import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

device = torch.device("cpu")

# Define the Siamese Network model (same structure used during training)
class SimilarityModel(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base"):
        super(SimilarityModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):
        output1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1)[0][:, 0, :]
        output2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2)[0][:, 0, :]
        return self.cos(output1, output2)

    # Helper method to embed and compare two code strings
    def predict_similarity(self, code1, code2):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        inputs1 = tokenizer(code1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs2 = tokenizer(code2, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            sim = self.forward(
                inputs1["input_ids"].to(device),
                inputs2["input_ids"].to(device),
                inputs1["attention_mask"].to(device),
                inputs2["attention_mask"].to(device)
            )
        return sim.item()
