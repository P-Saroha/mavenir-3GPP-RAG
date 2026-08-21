# Quick Setup

## Prerequisites
- Python 3.11+
- Internet connection

## Steps

1. **Extract TAR file**
   ```bash
   tar -xzf mavenir-3gpp-rag-codebase.tar.gz
   cd mavenir-3gpp-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate              # Mac/Linux
   # or
   myenv\Scripts\activate.ps1             # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Chroma** (one-time, ~1-2 minutes)
   ```bash
   python initialize_chroma.py
   ```
   This loads pre-computed embeddings and chunks into Chroma.

5. **Start backend** (Terminal 1)
   ```bash
   uvicorn src.api:app --port 8000 --reload
   ```

6. **Start frontend** (Terminal 2)
   ```bash
   streamlit run app.py
   ```

7. **Open browser**
   Navigate to: http://localhost:8501

## Done!

The system is ready to answer 3GPP questions. No re-embedding needed - all embeddings are pre-computed.

## Troubleshooting

- **Python not found**: Install Python 3.11+ from python.org
- **pip install fails**: Try `pip install --upgrade pip` first
- **initialize_chroma.py fails**: Ensure data/chunks.jsonl and data/embeddings.npy exist
- **Port 8000/8501 in use**: Change port in commands above
