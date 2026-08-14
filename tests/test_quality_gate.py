"""
tests/test_quality_gate.py
---------------------------
Tests for src/retrieval/quality_gate.py
"""

import pytest
from src.retrieval.quality_gate import check
from src.utils.config import MIN_RERANK_SCORE, MIN_EVIDENCE_COUNT, REQUIRED_METADATA


def _candidate(score=5.0, spec="23.501", section="4.2", page=36):
    return {
        "chunk_id": "test-id",
        "rerank_score": score,
        "spec": spec,
        "section": section,
        "page": page,
        "text": "The AMF handles access and mobility management.",
    }


# ── structure of the return value ─────────────────────────────────────────────

def test_returns_dict_with_required_keys():
    result = check([_candidate()])
    assert "supported" in result
    assert "reason" in result
    assert "evidence" in result


def test_supported_is_bool():
    result = check([_candidate()])
    assert isinstance(result["supported"], bool)


# ── check 1: count ────────────────────────────────────────────────────────────

def test_fails_when_empty():
    result = check([])
    assert result["supported"] is False
    assert "Insufficient" in result["reason"]
    assert result["evidence"] == []


def test_fails_when_below_min_count():
    # MIN_EVIDENCE_COUNT=2, send only 1
    result = check([_candidate()])
    if MIN_EVIDENCE_COUNT > 1:
        assert result["supported"] is False


def test_passes_with_sufficient_count():
    candidates = [_candidate(score=5.0 - i) for i in range(MIN_EVIDENCE_COUNT)]
    result = check(candidates)
    # should not fail on count (may still fail on score if threshold is high)
    if result["supported"] is False:
        assert "Insufficient" not in result["reason"]


# ── check 2: score ────────────────────────────────────────────────────────────

def test_fails_when_all_scores_too_low():
    candidates = [_candidate(score=-5.0), _candidate(score=-3.0)]
    result = check(candidates)
    assert result["supported"] is False
    assert "weak" in result["reason"].lower()
    assert result["evidence"] == []


def test_passes_when_score_above_threshold():
    candidates = [_candidate(score=MIN_RERANK_SCORE + 1.0),
                  _candidate(score=MIN_RERANK_SCORE + 0.5)]
    result = check(candidates)
    assert result["supported"] is True
    assert len(result["evidence"]) == 2


# ── check 3: metadata ─────────────────────────────────────────────────────────

def test_fails_when_spec_missing():
    c = _candidate()
    c["spec"] = ""
    result = check([c, _candidate()])
    assert result["supported"] is False
    assert "metadata" in result["reason"].lower()


def test_fails_when_section_missing():
    c = _candidate()
    c["section"] = ""
    result = check([c, _candidate()])
    assert result["supported"] is False


def test_fails_when_page_missing():
    c = _candidate()
    c["page"] = None
    result = check([c, _candidate()])
    assert result["supported"] is False


# ── integration: real queries ─────────────────────────────────────────────────

def test_3gpp_query_passes():
    from src.retrieval.reranker import retrieve_and_rerank
    candidates = retrieve_and_rerank("What is the role of the AMF?", top_k=5)
    result = check(candidates)
    assert result["supported"] is True
    assert len(result["evidence"]) > 0


def test_unrelated_query_fails():
    from src.retrieval.reranker import retrieve_and_rerank
    candidates = retrieve_and_rerank("What is the capital of France?", top_k=5)
    result = check(candidates)
    assert result["supported"] is False


def test_outside_corpus_query_fails():
    from src.retrieval.reranker import retrieve_and_rerank
    candidates = retrieve_and_rerank("How do I bake a chocolate cake?", top_k=5)
    result = check(candidates)
    assert result["supported"] is False
