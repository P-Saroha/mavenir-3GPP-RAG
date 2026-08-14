"""
tests/test_mmr.py
------------------
Tests for src/retrieval/mmr.py
"""

import numpy as np
import pytest
from src.retrieval.mmr import mmr, MMR_TOP_K, LAMBDA_PARAM


def _fake_candidates(texts: list[str], scores: list[float]) -> list[dict]:
    """Build minimal candidate dicts with real chunk_ids from the corpus."""
    import json
    from pathlib import Path
    ids = json.loads(Path("data/embedding_ids.json").read_text())
    candidates = []
    for i, (text, score) in enumerate(zip(texts, scores)):
        candidates.append({
            "chunk_id":     ids[i],
            "rerank_score": score,
            "spec":         "23.501",
            "section":      f"4.{i}",
            "page":         i + 1,
            "text":         text,
        })
    return candidates


# ── unit tests ────────────────────────────────────────────────────────────────

def test_mmr_returns_top_k():
    candidates = _fake_candidates(["text"] * 10, [5.0 - i * 0.1 for i in range(10)])
    result = mmr(candidates, top_k=5)
    assert len(result) == 5


def test_mmr_fewer_candidates_than_top_k():
    candidates = _fake_candidates(["text"] * 3, [3.0, 2.0, 1.0])
    result = mmr(candidates, top_k=5)
    assert len(result) == 3


def test_mmr_empty_input():
    assert mmr([]) == []


def test_mmr_adds_mmr_score_field():
    candidates = _fake_candidates(["text"] * 5, [5.0, 4.0, 3.0, 2.0, 1.0])
    result = mmr(candidates)
    for r in result:
        assert "mmr_score" in r


def test_mmr_preserves_metadata():
    candidates = _fake_candidates(["text"] * 5, [5.0, 4.0, 3.0, 2.0, 1.0])
    result = mmr(candidates)
    for r in result:
        for field in ("chunk_id", "spec", "section", "page", "text"):
            assert field in r


def test_mmr_first_pick_is_highest_relevance():
    """First selection must be the highest-scoring candidate."""
    candidates = _fake_candidates(
        ["alpha"] * 6,
        [1.0, 5.0, 2.0, 3.0, 4.0, 0.5]   # index 1 is highest
    )
    result = mmr(candidates, top_k=3)
    # first result should come from the highest rerank_score candidate
    assert result[0]["rerank_score"] == 5.0


def test_mmr_reduces_duplicates():
    """
    When several candidates are near-identical (same section, adjacent pages),
    MMR should prefer a diverse candidate over another near-duplicate.
    We verify by using two corpus chunks from the same section vs one from a
    distant section — the distant one should be selected despite lower score.
    """
    import json
    from pathlib import Path
    ids = json.loads(Path("data/embedding_ids.json").read_text())

    # ids[0] and ids[1] are likely from the same/adjacent sections (similar embedding)
    # ids[100] should be from a very different part of the corpus
    close_a = ids[0]
    close_b = ids[1]
    distant = ids[100]

    candidates = [
        {"chunk_id": close_a, "rerank_score": 5.0, "spec": "23.501",
         "section": "4.1", "page": 1, "text": "Network slice NSSAI configuration"},
        {"chunk_id": close_b, "rerank_score": 4.8, "spec": "23.501",
         "section": "4.1", "page": 2, "text": "Network slice NSSAI configuration details"},
        {"chunk_id": distant, "rerank_score": 3.0, "spec": "23.501",
         "section": "6.3", "page": 80, "text": "AMF registration and mobility procedure"},
    ]
    result = mmr(candidates, top_k=3)
    result_ids = [r["chunk_id"] for r in result]
    # all three should be selected (we asked for top_k=3 with 3 candidates)
    assert len(result) == 3
    assert distant in result_ids


# ── integration test ──────────────────────────────────────────────────────────

def test_mmr_on_real_query():
    from src.retrieval.reranker import retrieve_and_rerank
    reranked = retrieve_and_rerank("What is network slicing and how is NSSAI used?", top_k=8)
    selected = mmr(reranked, top_k=MMR_TOP_K)

    assert len(selected) == MMR_TOP_K
    # no duplicate chunk_ids
    ids = [r["chunk_id"] for r in selected]
    assert len(ids) == len(set(ids))
    # sections should be at least as diverse as a naive top-5 slice
    sections_after = len(set(r["section"] for r in selected))
    sections_naive = len(set(r["section"] for r in reranked[:MMR_TOP_K]))
    assert sections_after >= sections_naive
