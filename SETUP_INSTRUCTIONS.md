# Setup Instructions for Running the RAG Chatbot

## Option A: Ready-to-Run (Recommended - No Rebuilding)

If you extracted the TAR file and want to use pre-indexed data:

```bash
# 1. Extract TAR
tar -xzf mavenir-3gpp-rag-codebase.tar.gz -C /path/to/extract

# 2. Setup Python environment
cd mavenir-3gpp-rag
python -m venv myenv
source myenv/bin/activate          # Mac/Linux
# OR
myenv\Scripts\activate.ps1         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize Chroma from pre-computed embeddings (1-2 minutes, one-time)
python initialize_chroma.py

# 5. Start the application
# Terminal 1: Start backend
uvicorn src.api:app --port 8000 --reload

# Terminal 2: Start frontend
streamlit run app.py
```

The system uses **pre-computed embeddings** included in the TAR file (`data/embeddings.npy`), so no re-embedding needed.

---

## What Happens During Setup

### Step 4: `python initialize_chroma.py`

This script:
1. Loads `data/chunks.jsonl` (2,330 chunks)
2. Loads `data/embeddings.npy` (pre-computed vectors)
3. Creates Chroma collection at `./chroma_data/`
4. Indexes all chunks with their embeddings
5. Time: 1-2 minutes (fast, because embeddings already computed)

**Result:** Ready to query immediately after!

---

## FAQ

**Q: Do I need to regenerate embeddings?**
A: No! Embeddings are pre-computed and included in the TAR file.

**Q: How long does initialization take?**
A: ~1-2 minutes. Much faster than re-embedding 2,330 chunks.

**Q: Can I skip initialization?**
A: No. Chroma needs to be indexed first. This is a one-time setup.

**Q: What if initialization fails?**
A: Check:
- Python 3.11+ installed: `python --version`
- Dependencies installed: `pip install -r requirements.txt`
- 50MB free disk space for Chroma index
- Internet connection (for model downloads)

---

## Option B: From Scratch (Advanced)

If you want to re-chunk and re-embed everything:

```bash
# 1-3. Same as Option A (setup environment)

# 4. Download models (one-time, ~500MB)
python setup_models.py

# 5. Re-chunk and re-embed all PDFs (30-60 min per 100 pages)
python -m src.ingestion.ingest data/pdfs/*.pdf

# 6. Initialize Chroma (automatic during ingestion)

# 7. Start application (same as Option A, steps 5-6)
```

**Note:** This is for modifying chunks or adding new PDFs. Not needed for evaluation.

---

## Production Deployment

For production use, consider:
- Docker containerization (Dockerfile provided)
- Redis caching layer
- Load balancer (Nginx)
- Authentication (FastAPI + JWT)
- Monitoring (Prometheus + Grafana)

See DESIGN.md for detailed architecture and scaling roadmap.
