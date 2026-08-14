"""
tests/test_dense_search.py
---------------------------
Tests for src/retrieval/dense_search.py.
Requires: Qdrant running + model cached locally.
"""

import pytest
from src.retrieval.dense_search import search

QUERIES = [
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
]


def test_search_returns_results():
    results = search(QUERIES[0], top_k=5)
    assert len(results) == 5


def test_result_fields():
    results = search(QUERIES[0], top_k=3)
    required = ("chunk_id", "score", "spec", "release", "version",
                 "section", "section_title", "parent_section",
                 "page", "page_end", "text")
    for r in results:
        for f in required:
            assert f in r, f"Missing field: {f}"


def test_scores_between_0_and_1():
    results = search(QUERIES[1], top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"


def test_scores_descending():
    results = search(QUERIES[2], top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_default_release_is_17():
    results = search(QUERIES[0], top_k=5)
    for r in results:
        assert r["release"] == "17", f"Expected release 17, got {r['release']}"


def test_spec_filter():
    results = search(QUERIES[3], top_k=5, spec="23.501")
    for r in results:
        assert r["spec"] == "23.501", f"Spec filter broken: {r['spec']}"


def test_amf_query_returns_amf_content():
    results = search("What is the role of the AMF?", top_k=5)
    texts = " ".join(r["text"].lower() for r in results)
    assert "amf" in texts


def test_top_k_respected():
    for k in (1, 3, 5):
        results = search(QUERIES[0], top_k=k)
        assert len(results) == k


@pytest.mark.parametrize("query", QUERIES)
def test_all_five_queries_return_results(query):
    results = search(query, top_k=5)
    assert len(results) > 0
    assert all(r["text"].strip() for r in results)
