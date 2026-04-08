# AdaptRAG AI — Domain-Adaptive RAG AI Assistant

An intelligent, multi-user AI chatbot built using **Retrieval-Augmented Generation (RAG)** that can adapt to any domain by ingesting custom data (PDFs, prescriptions, etc.).

---

## 🚀 Overview

AdapRAG AI is a **full-stack AI assistant** that combines:

* 🔍 Retrieval-Augmented Generation (RAG)
* 🧠 User memory & conversation tracking
* 💊 Prescription understanding
* 👥 Multi-user support
* 🌐 Web-based UI

Unlike traditional chatbots, AdapRAG AI can **learn from your own data** and provide **context-aware, accurate responses**.

---

## ✨ Key Features

### 🧠 1. Domain-Adaptive RAG

* Upload domain-specific documents (e.g., medical PDFs)
* Automatically converts them into embeddings
* Retrieves relevant context for answering queries

---

### 👤 2. Multi-User Support

* Each user has:

  * Separate conversations
  * Personalized memory
* Built using MongoDB for persistence

---

### 💬 3. Chat History Management

* Stores conversations and messages
* Context-aware responses using recent history

---

### 💊 4. Prescription Analysis

* Upload prescriptions (PDF/Image)
* Extracts and explains:

  * Medicines
  * Dosage
  * Purpose
  * Side effects

---

### 🧠 5. Intelligent Memory System

* Extracts important user info automatically
* Stores it for future interactions

---

### 🎯 6. Intent-Based Response System

Detects user intent:

* 🩺 Medical queries → Uses RAG
* 💊 Prescription queries → Uses prescription DB
* 💙 Emotional queries → Empathetic responses
* 💬 General queries → Normal conversation

---

### 🎨 7. Modern Web UI

* Dark theme
* ChatGPT-like interface
* Conversation sidebar
* Real-time chat experience

---

## 🏗️ Tech Stack

### 🔹 Backend

* Python (Flask)
* MongoDB (Database)
* LangChain (RAG pipeline)
* ChromaDB (Vector database)
* Groq API (LLM)

### 🔹 Frontend

* HTML
* CSS (Dark UI)
* JavaScript (Fetch API)

---

## 📂 Project Structure

```
AdaptRAG AI/
│
├── app.py                     # Main Flask app (API + routing)
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── .gitignore
│
├── src/                       # Core utilities & helpers
│   ├── __init__.py
│   ├── helper.py              # PDF loading, embeddings, preprocessing
│   └── prompt.py              # Prompt templates / formatting
│
├── services/                  # Business logic layer
│   ├── __init__.py
│   ├── answer.py              # Main RAG pipeline (generate_answer)
│   ├── chat_history.py        # Conversation handling
│   ├── intent.py              # Intent detection (medical, general, etc.)
│   ├── memory_manager.py      # User memory extraction & storage
│   └── prescription.py        # Prescription processing + storage
│
├── rag/                       # Retrieval-Augmented Generation modules
│   ├── __init__.py
│   ├── ingest.py              # Data ingestion → embeddings → vector DB
│   └── retriver.py            # Chroma retriever (optimized search)
│
├── frontend/                  # UI (client-side)
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data/                      # Input documents (PDFs for RAG)
│
└── chroma_db/                 # Vector DB (auto-generated, ignored)
```

---

## ⚙️ How It Works

### 🔁 RAG Pipeline

1. Load documents (PDFs)
2. Split into chunks
3. Convert into embeddings
4. Store in ChromaDB
5. Retrieve relevant chunks during query
6. Generate answer using LLM

---

### 🔍 Optimized Retrieval

* Top-K retrieval
* Re-ranking of documents
* Context filtering
* Query rewriting

---

## 🧪 How to Run Locally

### 1️⃣ Clone the repo

```
git clone https://github.com/karmawangchuk07/AdaptRAG-AI.git
cd AdapRAG AI
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Add environment variables

Create `.env` file:

```
MONGO_URI=your_mongodb_uri
GROQ_API_KEY=your_groq_api_key
```

---

### 5️⃣ Run the app

```
python app.py
```

---

### 6️⃣ Open in browser

```
http://localhost:5000
```

---

## 📦 Deployment Notes

* `chroma_db/` is **not stored in GitHub**
* It is **automatically rebuilt on startup**
* MongoDB stores persistent data (chat, memory, prescriptions)

---

## ⚠️ Limitations

* Vector DB rebuilds on restart (acceptable for demo)
* Not optimized for large-scale production yet

---

## 🚀 Future Improvements

* 🌐 Deploy with Docker
* ⚡ Use Pinecone / Qdrant for persistent vector DB
* 🔄 Streaming responses (real-time typing)
* 📊 Evaluation metrics for RAG

---

## 🧠 Key Learning Outcomes

* Built a complete RAG pipeline
* Implemented multi-user backend system
* Integrated vector DB + LLM
* Designed scalable architecture
* Developed full-stack AI application

---

## 📌 Use Cases

* Healthcare assistants
* Knowledge-based chatbots
* Document Q&A systems
* Personalized AI assistants

---

## 👨‍💻 Author

**Karma**

---
