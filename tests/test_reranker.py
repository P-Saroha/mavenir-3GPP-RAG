"""
tests/test_reranker.py
-----------------------
Tests for src/retrieval/reranker.py
"""

import pytest
from src.retrieval.reranker import rerank, retrieve_and_rerank, RERANK_OUTPUT_K


def _fake_candidates(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"id-{i}", "text": f"The AMF handles access and mobility for UE number {i}.",
            "spec": "23.501", "release": "17", "version": "17.13.0",
            "section": f"4.{i}", "section_title": "Test", "parent_section": "4",
            "page": i, "page_end": i, "rrf_score": 1.0 / (i + 1),
        }
        for i in range(n)
    ]


# ── unit tests ────────────────────────────────────────────────────────────────

def test_rerank_returns_top_k():
    candidates = _fake_candidates(20)
    results = rerank("What is the AMF?", candidates, top_k=8)
    assert len(results) == 8


def test_rerank_score_field_added():
    candidates = _fake_candidates(5)
    results = rerank("What is the AMF?", candidates, top_k=3)
    for r in results:
        assert "rerank_score" in r


def test_rerank_scores_descending():
    candidates = _fake_candidates(10)
    results = rerank("What is the AMF?", candidates, top_k=8)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_preserves_metadata():
    candidates = _fake_candidates(5)
    results = rerank("What is the AMF?", candidates, top_k=3)
    for r in results:
        for field in ("chunk_id", "spec", "section", "page", "text"):
            assert field in r


def test_rerank_empty_input():
    results = rerank("query", [], top_k=5)
    assert results == []


def test_rerank_fewer_candidates_than_top_k():
    candidates = _fake_candidates(3)
    results = rerank("What is the AMF?", candidates, top_k=8)
    assert len(results) == 3  # can't return more than we have


# ── integration tests ─────────────────────────────────────────────────────────

def test_retrieve_and_rerank_returns_results():
    results = retrieve_and_rerank("What is the role of the AMF?")
    assert len(results) == RERANK_OUTPUT_K


def test_retrieve_and_rerank_fields():
    results = retrieve_and_rerank("PDU session establishment")
    for r in results:
        for field in ("chunk_id", "rerank_score", "spec", "section", "page", "text"):
            assert field in r


def test_retrieve_and_rerank_relevant_for_amf():
    results = retrieve_and_rerank("What is the role of the AMF?")
    texts = " ".join(r["text"].lower() for r in results)
    assert "amf" in texts


def test_retrieve_and_rerank_spec_filter():
    results = retrieve_and_rerank("policy control PCF SMF", spec="23.503")
    specs = [r["spec"] for r in results]
    assert "23.503" in specs
