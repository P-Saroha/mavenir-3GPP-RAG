"""
tests/test_bm25.py
-------------------
Tests for src/retrieval/bm25.py
"""

from pathlib import Path
from src.retrieval.bm25 import search, tokenize, load_index, CACHE_PATH


def test_tokenize():
    tokens = tokenize("What is the AMF? It's a 5G function.")
    assert "amf" in tokens
    assert "5g" in tokens
    assert "what" in tokens
    # punctuation removed
    assert "?" not in tokens
    assert "." not in tokens


def test_search_returns_results():
    results = search("What is the role of the AMF?", top_k=5)
    assert len(results) == 5


def test_result_fields():
    results = search("AMF registration procedure", top_k=3)
    for r in results:
        for field in ("chunk_id", "score", "spec", "section", "page", "text"):
            assert field in r, f"Missing field: {field}"


def test_scores_descending():
    results = search("session management SMF PDU", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"


def test_relevant_result_for_amf_query():
    results = search("What is the role of the AMF?", top_k=5)
    # At least one result should mention AMF in the text
    texts = " ".join(r["text"].lower() for r in results)
    assert "amf" in texts


def test_cache_file_created():
    load_index()
    assert CACHE_PATH.exists(), "Cache file was not created"


def test_top_k_respected():
    for k in (1, 3, 10):
        results = search("UPF user plane function", top_k=k)
        assert len(results) == k
