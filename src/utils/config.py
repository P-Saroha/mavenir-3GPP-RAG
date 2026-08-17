import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider: xAI Grok (ONLY LLM provider) ─────────────────────────────────
GROK_API_KEY  = os.getenv("GROK_API_KEY", "")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL    = os.getenv("GROK_MODEL", "grok-3")

# ── Retrieval ──────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# ── Evidence quality gate thresholds ───────────────────────────────────────────
# Minimum reranker score for at least one candidate to pass the gate.
# cross-encoder/ms-marco-MiniLM-L6-v2 raw logits: ~0 for weak, >3 for strong.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "1.0"))

# Minimum number of candidates required.
MIN_EVIDENCE_COUNT = int(os.getenv("MIN_EVIDENCE_COUNT", "2"))

# ── MMR configuration ──────────────────────────────────────────────────────────
MMR_TOP_K = int(os.getenv("MMR_TOP_K", "7"))

# Required metadata fields on every evidence chunk.
REQUIRED_METADATA = ["spec", "section", "page"]
