from src.utils.config import GROK_API_KEY, QDRANT_URL, EMBEDDING_MODEL, RERANKER_MODEL, MMR_TOP_K

def test_defaults():
    assert QDRANT_URL == "http://localhost:6333"
    # EMBEDDING_MODEL comes from .env (may be all-MiniLM or nomic)
    assert isinstance(EMBEDDING_MODEL, str)
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert MMR_TOP_K == 7
    assert isinstance(GROK_API_KEY, str)  # empty string is fine — no key required in tests
