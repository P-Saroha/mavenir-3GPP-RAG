# 3GPP Release 17 RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** for 3GPP Release 17 5G Core specifications. Ask technical questions about 3GPP standards and get grounded answers with proper citations.

---

## 🎯 Features

- **Semantic Search**: Dense vector embeddings + BM25 retrieval from 1,980 3GPP specification chunks
- **Cross-Encoder Reranking**: Relevance scoring using MS-Marco cross-encoder
- **Quality Gate**: Evidence validation with configurable thresholds
- **Context Expansion**: Pulls adjacent sections for complete answers
- **Citations**: All responses include `[S1]`, `[S2]` tags with source specs/sections/pages
- **Persistent Storage**: Chroma database on disk (survives restarts)
- **Free LLM**: Groq API (fast, high-quality, no cost)
- **PDF Upload**: Add new 3GPP specs via Streamlit UI
- **No Docker**: Everything runs locally in Python virtual environment

---

## 📋 Requirements

- Python 3.11+
- Virtual environment (myenv)
- 2GB+ disk space for embeddings + Chroma database
- Internet (for Groq API + HuggingFace model downloads)

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd f:\Projects\mavenir
myenv\Scripts\activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
MIN_RERANK_SCORE=0.5
```

Get free Groq API key: https://console.groq.com

### 3. Migrate Data to Chroma (First Time Only)

```bash
python migrate_qdrant_to_chroma.py
```

Output:
```
✓ Loaded 1980 chunks from chunks.jsonl
✓ Created Chroma collection: 3gpp_r17_5gcore
✓ Total inserted: 1980 chunks
✓ Chroma collection contains 1980 records
```

### 4. Start Backend

```bash
uvicorn src.api:app --port 8000 --reload
```

### 5. Start Frontend

In another terminal:
```bash
myenv\Scripts\activate.ps1
streamlit run main.py
```

Opens: http://localhost:8501

---

## 🔄 System Architecture

```mermaid
graph LR
    A["👤 User Query<br/>(Streamlit)"] --> B["🔍 Dense Retrieval<br/>(Chroma)<br/>top 30"]
    B --> C["⚖️ Cross-Encoder<br/>Reranking<br/>top 10"]
    C --> D["🎯 MMR Filter<br/>(Diversity)<br/>top 7"]
    D --> E{"✅ Quality<br/>Gate<br/>score≥0.5?"}
    E -->|PASS| F["📖 Context<br/>Expansion<br/>+Adjacent Sections"]
    E -->|REJECT| Z["❌ Cannot Answer"]
    F --> G["🏷️ Citation<br/>Tagging<br/>[S1][S2]..."]
    G --> H["🤖 LLM Generation<br/>(Groq API)<br/>GPT-OSS-120b"]
    H --> I["✓ Citation<br/>Validation<br/>&Parsing"]
    I --> J["📊 Final Response<br/>Answer + Sources<br/>(Streamlit)"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#ffe0b2
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#c8e6c9
    style I fill:#b3e5fc
    style J fill:#c8e6c9
    style Z fill:#ffcdd2
```

### Pipeline Stages

| Stage | Component | Time | Output |
|-------|-----------|------|--------|
| 1️⃣ **Dense Search** | Chroma (cosine similarity) | ~50ms | 30 candidates |
| 2️⃣ **Reranking** | Cross-encoder (MS-Marco) | ~500ms | 10 scored candidates |
| 3️⃣ **MMR Filter** | Diversity balance | ~20ms | 7 diverse candidates |
| 4️⃣ **Quality Gate** | Evidence validation | ~5ms | Pass/Reject |
| 5️⃣ **Context Expansion** | Adjacent section pull | ~100ms | Expanded chunks |
| 6️⃣ **Citation Tagging** | [S1], [S2]... assignment | ~10ms | Tagged evidence |
| 7️⃣ **LLM Generation** | Groq API call | ~2-3s | Answer with citations |
| 8️⃣ **Citation Validation** | Regex parse & verify | ~50ms | Final response |
| **TOTAL** | End-to-end | **~3-4s** | **Cited answer** |

### Data Flow

```
Query
  ↓
[Dense Search] → Chroma DB (./chroma_data/)
  ↓ (30 results)
[Reranking] → Cross-encoder scores
  ↓ (10 results)
[MMR Filter] → Diversity selection
  ↓ (7 results)
[Quality Gate] → Threshold check (score ≥ 0.5, count ≥ 2)
  ↓ (PASS)
[Context Expansion] → Pull adjacent sections
  ↓ (expanded chunks)
[Citation Tagging] → [S1], [S2]... assignment
  ↓ (tagged evidence)
[LLM Generation] → Groq API (with system prompt)
  ↓ (raw answer)
[Citation Validation] → Parse [Sx] tags
  ↓ (final answer)
Response → User
```

---

## 📁 Project Structure

```
f:\Projects\mavenir\
├── src/
│   ├── api.py                 # FastAPI backend (port 8000)
│   ├── rag.py                 # RAG pipeline orchestration
│   ├── retrieval/
│   │   ├── chroma_db.py       # Chroma persistent client
│   │   ├── dense_search_chroma.py  # Vector search
│   │   ├── reranker.py        # Cross-encoder scoring
│   │   ├── mmr.py             # Diversity filter
│   │   ├── quality_gate.py    # Evidence validation
│   │   └── context_builder.py # Context expansion
│   ├── generation/
│   │   ├── grok.py            # Groq API client
│   │   └── citations.py       # Citation parsing & tagging
│   ├── ingestion/
│   │   ├── parser.py          # PDF parsing
│   │   ├── chunker.py         # Text chunking
│   │   └── ingest.py          # Main ingestion pipeline
│   └── utils/
│       └── config.py          # Configuration
├── main.py                    # Streamlit UI
├── app.py                     # FastAPI entry point
├── config/
│   └── settings.py            # Pydantic settings
├── data/
│   ├── chunks.jsonl           # 1,980 chunk corpus
│   ├── embeddings.npy         # 1980x384 embedding vectors
│   └── pdfs/                  # Original 3GPP spec PDFs
├── chroma_data/               # Persistent Chroma database
├── tests/                     # Pytest test suite
├── .env                       # Environment variables (API keys)
├── requirements.txt           # Python dependencies
└── pyproject.toml            # Project metadata
```

---

## 🔧 Configuration

### `.env` File

```bash
# LLM Provider: Groq
GROQ_API_KEY=gsk_...                    # Get from https://console.groq.com
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b          # Latest Groq model

# Retrieval
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Fast (6M params)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2      # Relevance scoring

# Quality Gate Thresholds
MIN_RERANK_SCORE=0.5                    # Minimum cross-encoder score
MIN_EVIDENCE_COUNT=2                    # Minimum chunks required

# Retrieval Tuning
MMR_TOP_K=7                             # Diversity-filtered candidates
```

---

## 📊 Performance Metrics

| Stage | Time | Output |
|-------|------|--------|
| Dense search (Chroma) | ~50ms | 30 candidates |
| Reranking (cross-encoder) | ~500ms | 10 scored candidates |
| MMR filtering | ~20ms | 7 diverse candidates |
| Quality gate | ~5ms | Pass/Reject decision |
| LLM generation (Groq) | ~2-3s | Final answer |
| **Total** | **~3-4s** | **Cited answer** |

Embedding model: 6M params (fast on CPU)
Reranker: 22M params (runs locally on CPU)
LLM: Groq cloud (fast inference, free)

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Test specific module:
```bash
pytest tests/test_rag.py -v -k "test_answer_question"
```

Key test files:
- `test_rag.py` - Full pipeline
- `test_chroma_db.py` - Chroma retrieval
- `test_quality_gate.py` - Evidence validation
- `test_citations.py` - Citation parsing

---

## 📈 Adding New PDFs

### Via Streamlit UI
1. Click **"📄 Upload PDF"** in Streamlit
2. Select 3GPP PDF (up to 200MB)
3. Wait for: Parsing → Chunking → Embedding → Indexing
4. Query immediately (data auto-added to Chroma)

### Via CLI
```bash
python -m src.ingestion.ingest <path_to_pdf>
```

Processing time: ~30-60 min per 100 pages (CPU embedding)
For faster embedding: Use Google Colab with CUDA GPU

---

## 🔍 Usage Examples

### Via API
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of the AMF in 5G core network?"}'
```

Response:
```json
{
  "answer": "The Access and Mobility Management Function (AMF)...",
  "sources": [
    {
      "id": "S1",
      "spec": "23.501",
      "section": "6.1.1",
      "page": 42,
      "text": "..."
    }
  ],
  "supported": true
}
```

### Via Streamlit UI
1. Open http://localhost:8501
2. Type question in input box
3. Click **Ask**
4. See answer + expandable source sections

---

## 🛠️ Troubleshooting

### ❌ "I cannot reliably answer"
- **Cause**: Quality gate rejected evidence (score < 0.5 or < 2 chunks)
- **Fix**: Lower `MIN_RERANK_SCORE` in `.env` to 0.3
- **Debug**: Run `python diagnose_query.py` to see rerank scores

### ❌ No sources found
- **Cause**: Chroma database empty
- **Fix**: Re-run `python migrate_qdrant_to_chroma.py`
- **Check**: `python -c "from src.retrieval.chroma_db import get_collection; print(get_collection().count())"`

### ❌ Slow embedding on PDF upload
- **Cause**: CPU-based embedding (all-MiniLM-L6-v2 is ~6M params)
- **Fix**: Use Google Colab for GPU acceleration (5x faster)
- **Time**: ~30 min per 100 pages on CPU

### ❌ Groq API rate limit
- **Cause**: Free tier has limits (~30 requests/min)
- **Fix**: Wait 1 minute or upgrade plan
- **Status**: Check https://console.groq.com

---

## 📚 Key Components

### Dense Search (Chroma)
- Semantic vector search from Chroma persistent database
- Returns top 30 candidates by cosine similarity
- Embeddings: all-MiniLM-L6-v2 (384-dim, fast)

### Cross-Encoder Reranking
- MS-Marco trained cross-encoder (22M params)
- Scores (query, passage) relevance: -1 to +8 range
- Returns top 10 with scores

### MMR Diversity Filter
- Maximal Marginal Relevance balances relevance + diversity
- Prevents redundant evidence
- Returns top 7 candidates

### Quality Gate
- Validates evidence sufficiency
- Checks: score ≥ MIN_RERANK_SCORE (0.5) AND count ≥ 2
- Rejects low-confidence evidence before LLM

### Context Expansion
- Finds adjacent chunks in same section
- Expands context by ±1 neighbor per side
- Improves answer coherence

### Citation System
- LLM generates answers with [S1], [S2]... tags
- Fallback: Auto-inserts tags if LLM forgets
- Validation: Checks citations against source_map

---

## 🔐 Security Notes

- **API Key**: Groq API key in `.env` (never commit to git)
- **Local Processing**: Embeddings + reranking run locally (no data sent externally except to Groq for LLM)
- **Database**: Chroma data in `./chroma_data/` (persistent, local)
- **No auth**: Streamlit UI is open (use behind reverse proxy in production)

---

## 📝 License

Project for Mavenir. 3GPP specifications are proprietary.

---

## 🤝 Support

For issues:
1. Check troubleshooting section above
2. Review debug logs in terminal
3. Inspect `.env` configuration
4. Check Chroma data: `python migrate_qdrant_to_chroma.py --verify`

---

## 📊 Data Summary

| Metric | Value |
|--------|-------|
| Total chunks | 1,980 |
| Specs included | TS 23.501, TS 23.502, TS 23.503 (3GPP Release 17) |
| Embedding dimension | 384 (all-MiniLM-L6-v2) |
| Database engine | Chroma (DuckDB + Parquet) |
| Storage location | `./chroma_data/` |
| Query response time | ~3-4 seconds |

---

## 🚀 Future Enhancements

- [ ] Web UI authentication (FastAPI + JWT)
- [ ] GPU acceleration for embeddings
- [ ] Multi-spec search (R16, R15, etc.)
- [ ] Conversation history/multi-turn QA
- [ ] Batch processing API
- [ ] Response caching for common questions
