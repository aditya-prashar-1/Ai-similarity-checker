# 🧠 AI Code Similarity Detection

A web application that analyzes the similarity between **user-written code** and **AI-generated reference implementations**.
This tool leverages **state-of-the-art NLP models for code** to detect similarity, help with plagiarism detection, and provide insights into coding approaches.

<img width="1350" height="810" alt="image" src="https://github.com/user-attachments/assets/a9381729-a879-4308-b2e3-9361c091155c" />

---

## 🚀 Architecture Overview

[DATA flow]
<img width="438" height="473" alt="image" src="https://github.com/user-attachments/assets/6d0c8725-1519-4361-8fea-3651dc50d99e" />

---

## ⚙️ Backend

The backend is powered by **Flask** and advanced AI models:

* **CodeBERT** → Embeds code snippets into vector representations
* **Siamese Neural Network** → Compares embeddings and calculates similarity scores
* **CodeGen** → Generates reference code implementations

📂 **Key Backend Files**

* `app.py` → Main Flask application
* `similarity_model.py` → Siamese network implementation
* `siamese_plagiarism_model.pt` → Pre-trained model weights

---

## 🎨 Frontend

The frontend is a **React application** providing an interactive interface with:

* 📝 Programming question input
* 💻 Code solution editor
* 🤖 AI-generated reference code display
* 📊 Visual similarity score representation

📂 **Key Frontend Files**

* `src/App.js` → Main React component
* `src/App.css` → Styling

---

## 🔄 Data Flow

1. User enters a programming question + code solution
2. Request sent to backend API
3. Backend generates reference code via **CodeGen**
4. Both code snippets embedded with **CodeBERT**
5. **Siamese Network** computes similarity score
6. Results returned and displayed in frontend

---

## ⚡ Setup Instructions

### 🔹 Backend Setup

```bash
# Navigate to backend
cd backend  

# Install dependencies
pip install -r requirements.txt  

# Run Flask server
python app.py  
```

👉 Server runs on `http://localhost:5000`

---

### 🔹 Frontend Setup

```bash
# Navigate to frontend
cd frontend  

# Install dependencies
npm install  

# Start dev server
npm start  
```

👉 App available at `http://localhost:3000`

---

## 🌐 API Endpoints

### `POST /generate_and_compare`

Generates reference code and compares with user solution.

**Request body:**

```json
{
  "question": "string",
  "code": "string"
}
```

**Response:**

```json
{
  "similarity_score": 0.85,
  "prediction": "Highly Similar",
  "confidence": "92%",
  "reference_code": "..."
}
```

### `GET /health`

Health check.

**Response:**

```json
{
  "status": "running",
  "model": "CodeBERT + Siamese",
  "device": "cuda"
}
```

---

## 🛠️ Technologies Used

* **Backend:** Python, Flask, PyTorch, Hugging Face Transformers
* **AI Models:** CodeBERT, CodeGen, Siamese Neural Network
* **Frontend:** React, JavaScript, CSS

---

## 🚧 Future Improvements

* 🔀 Multi-language support (C++, Java, etc.)
* 🧩 Deeper code similarity analysis
* 🔑 User authentication & history tracking
* ⚡ Performance optimization for large codebases
* 🎯 Richer reference code generation options

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue for major changes.

---

