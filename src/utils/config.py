import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM provider (Groq by default, xAI Grok if XAI_API_KEY is set) ───────────
XAI_API_KEY  = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL    = os.getenv("XAI_MODEL", "grok-3")

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")

# ── Evidence quality gate thresholds ─────────────────────────────────────────
# Minimum reranker score for at least one candidate to pass the gate.
# cross-encoder/ms-marco-MiniLM-L6-v2 raw logits: ~0 for weak, >3 for strong.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "1.0"))

# Minimum number of candidates required.
MIN_EVIDENCE_COUNT = int(os.getenv("MIN_EVIDENCE_COUNT", "2"))

# Required metadata fields on every evidence chunk.
REQUIRED_METADATA = ["spec", "section", "page"]
