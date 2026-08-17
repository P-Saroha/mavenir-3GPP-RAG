import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider: Groq (ONLY LLM provider) ─────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ── Retrieval ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# ── Evidence quality gate thresholds ───────────────────────────────────────────
# Minimum reranker score for at least one candidate to pass the gate.
# cross-encoder/ms-marco-MiniLM-L6-v2 raw logits: ~0 for weak, >3 for strong.
# Lowered to 0.5 to accept valid answers with decent evidence (was too strict at 1.0)
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.5"))

# Minimum number of candidates required.
MIN_EVIDENCE_COUNT = int(os.getenv("MIN_EVIDENCE_COUNT", "2"))

# ── MMR configuration ──────────────────────────────────────────────────────────
MMR_TOP_K = int(os.getenv("MMR_TOP_K", "7"))

# Required metadata fields on every evidence chunk.
REQUIRED_METADATA = ["spec", "section", "page"]
