from src.utils.config import GROQ_API_KEY, EMBEDDING_MODEL, RERANKER_MODEL, MMR_TOP_K

def test_defaults():
    # No QDRANT_URL anymore (using Chroma)
    # EMBEDDING_MODEL comes from .env (should be all-MiniLM)
    assert isinstance(EMBEDDING_MODEL, str)
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert MMR_TOP_K == 7
    assert isinstance(GROQ_API_KEY, str)  # empty string is fine — no key required in tests
