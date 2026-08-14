# 3GPP Release 17 5G Core RAG Chatbot

A RAG chatbot grounded in 3GPP Release 17 standards:
- TS 23.501 — System Architecture for 5G System
- TS 23.502 — Procedures for 5G System
- TS 23.503 — Policy and Charging Control Framework

## Project layout

```
project/
├── data/               # place 3GPP PDFs here
├── src/
│   ├── ingestion/      # PDF extraction and chunking
│   ├── retrieval/      # dense + BM25 retrieval, reranking
│   ├── generation/     # LLM generation (Grok / Ollama)
│   ├── evaluation/     # evaluation utilities
│   └── utils/          # shared helpers
├── tests/              # pytest tests
├── qdrant_storage/     # local Qdrant data (git-ignored)
├── .env.example
├── requirements.txt
└── main.py
```

## Setup

### 1. Create virtual environment
```bash
python -m venv myenv
```

### 2. Activate it
Windows:
```powershell
.\myenv\Scripts\Activate.ps1
```
Linux/Mac:
```bash
source myenv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## Models

| Role | Model |
|------|-------|
| Embedding | nomic-ai/nomic-embed-text-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| LLM (primary) | grok-3-mini via xAI API |
| LLM (fallback) | mistral via local Ollama |
