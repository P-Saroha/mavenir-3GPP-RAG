from src.utils.config import XAI_API_KEY, QDRANT_URL, EMBEDDING_MODEL, RERANKER_MODEL

def test_defaults():
    assert QDRANT_URL == "http://localhost:6333"
    assert EMBEDDING_MODEL == "nomic-ai/nomic-embed-text-v1.5"
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert isinstance(XAI_API_KEY, str)  # empty string is fine — no key required
