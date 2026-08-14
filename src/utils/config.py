import os
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
