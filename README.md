<p align="center">
  <h1 align="center">📚 RAG Document Chat</h1>
  <p align="center">
    <strong>Chat with your enterprise documents using AI — grounded answers, zero hallucination.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/LLM-Meta_Llama_3.2-0467DF?logo=meta&logoColor=white" alt="Meta Llama">
    <img src="https://img.shields.io/badge/Vector_DB-ChromaDB-4A154B?logo=databricks&logoColor=white" alt="ChromaDB">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
  <p align="center">
    <a href="https://rag-document-chat-mohamedsafiq.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Live_Demo-Click_Here-FF4B4B?style=for-the-badge" alt="Live Demo"></a>
  </p>
</p>

---

## 🎯 Overview

A **production-grade Retrieval-Augmented Generation (RAG)** web application that transforms how you interact with enterprise documents. Upload PDFs, DOCX files, or CSVs, and get AI-powered answers that are **grounded in your actual data** — with full source citations for transparency and trust.

> **Why RAG?** Unlike generic chatbots that hallucinate, this system retrieves real information from your documents before generating responses — ensuring accuracy, reliability, and compliance-readiness.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Ingestion** | Upload and parse PDF, DOCX, and CSV files seamlessly |
| 🧠 **Semantic Search** | Find relevant information using meaning, not just keywords |
| 🔒 **Strict Mode** | Answers *only* from documents — zero hallucination for enterprise/compliance use |
| 🔄 **Hybrid Mode** | Documents first, general knowledge as supplement — flexible UX |
| 📎 **Source Citations** | Every answer shows exactly which document, page, and section it came from |
| 💬 **Conversational Memory** | Multi-turn chat that remembers context across follow-up questions |
| 💾 **Persistent Storage** | Embeddings stored locally in ChromaDB — survives app restarts |
| ⚡ **ONNX Embeddings** | Lightweight, fast embedding via ONNX Runtime — no GPU or PyTorch required |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DOCUMENT PIPELINE                        │
│                                                                 │
│   📄 Upload ──▶ Parse & Extract ──▶ Chunk (500 chars) ──▶ Embed │
│   (PDF/DOCX/CSV)    (pypdf/docx)   (RecursiveTextSplitter) (ONNX)│
│                                                   │             │
│                                                   ▼             │
│                                           ┌──────────────┐     │
│                                           │  ChromaDB    │     │
│                                           │  Vector Store │     │
│                                           └──────┬───────┘     │
│                                                  │              │
│                        QUERY PIPELINE            │              │
│                                                  │              │
│   💬 Question ──▶ Semantic Search ───────────────┘              │
│        │              │                                         │
│        │              ▼                                         │
│        │     Top-K Relevant Chunks                              │
│        │              │                                         │
│        ▼              ▼                                         │
│   ┌─────────────────────────────┐                               │
│   │ Prompt Builder              │                               │
│   │ (Context + History + Mode)  │                               │
│   └──────────┬──────────────────┘                               │
│              ▼                                                  │
│   ┌─────────────────────────────┐                               │
│   │ Meta Llama 3.2 (HuggingFace)│                               │
│   └──────────┬──────────────────┘                               │
│              ▼                                                  │
│   🤖 Grounded Answer + 📎 Source Citations                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI with chat interface |
| **LLM** | Meta Llama 3.2 3B Instruct | Response generation via HuggingFace Inference API |
| **Embeddings** | ONNX MiniLM-L6-v2 | Fast, local semantic embeddings (no GPU needed) |
| **Vector DB** | ChromaDB | Persistent local vector storage & similarity search |
| **Doc Parsing** | PyPDF, python-docx, pandas | Multi-format document extraction |
| **Orchestration** | LangChain | Text splitting, document schemas |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [HuggingFace API Token](https://huggingface.co/settings/tokens) (free tier works)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Project\ 3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API token
#    Create a .env file with:
HUGGINGFACEHUB_API_TOKEN=your_token_here

# 4. Launch the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📖 Usage Guide

### Step 1: Upload Documents
Use the sidebar to upload one or more PDF, DOCX, or CSV files. Click **"🚀 Process Documents"** to ingest and embed them into the vector database.

### Step 2: Choose Response Mode

| Mode | Behavior | Best For |
|------|----------|----------|
| 🔒 **Strict** | Answers *only* from documents. Returns "insufficient information" if answer isn't found. | Compliance, legal, auditing |
| 🔄 **Hybrid** | Prioritizes documents, supplements with general knowledge when needed. | Research, exploration, learning |

### Step 3: Ask Questions
Type your question in the chat input. The system will:
1. Search the vector database for relevant document chunks
2. Build a context-aware prompt with conversation history
3. Generate a grounded response via Meta Llama 3.2
4. Display source citations with document name, page, and text snippet

---

## 📁 Project Structure

```
Project 3/
├── app.py              # Streamlit UI — chat interface, sidebar, mode toggle
├── ingest.py           # Document parsing & recursive text chunking
├── vector_store.py     # ChromaDB operations — embed, search, clear
├── rag_chain.py        # LLM prompting — strict/hybrid modes, citations
├── requirements.txt    # Python dependencies
├── .env                # HuggingFace API token (not committed)
├── .gitignore          # Ignores sensitive & generated files
└── chroma_db/          # Persistent vector storage (auto-created)
```

---

## 🔑 Core Concepts Demonstrated

- **Document Ingestion Pipelines** — Multi-format parsing with metadata extraction
- **Semantic Chunking** — Recursive text splitting with overlap for context preservation
- **Vector Databases** — Persistent embedding storage with ChromaDB
- **Prompt Engineering** — Mode-specific system prompts for hallucination control
- **Conversational Memory** — Multi-turn chat context passed to the LLM
- **Hallucination Mitigation** — Strict mode constrains outputs to document context only
- **Source Attribution** — Transparent citations for every response

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Built with ❤️ using LangChain, ChromaDB, Meta Llama & Streamlit</strong>
</p>
