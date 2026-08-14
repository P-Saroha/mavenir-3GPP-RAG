"""
tests/test_context_builder.py
------------------------------
Tests for src/retrieval/context_builder.py
"""

from src.retrieval.context_builder import (
    _is_related_section, _normalise, expand_chunk, build_evidence, _get_corpus
)


# ── unit: section relationship ────────────────────────────────────────────────

def test_same_section():
    assert _is_related_section("4.2.1", "4.2.1") is True

def test_parent_section():
    assert _is_related_section("4.2.1", "4.2") is True
    assert _is_related_section("4.2.1", "4") is True

def test_direct_child_section():
    assert _is_related_section("4.2.1", "4.2.1.1") is True

def test_unrelated_sibling():
    assert _is_related_section("4.2.1", "4.3") is False
    assert _is_related_section("4.2.1", "4.2.2") is False

def test_too_deep_child():
    assert _is_related_section("4.2.1", "4.2.1.1.2") is False


# ── unit: normalise ───────────────────────────────────────────────────────────

def test_normalise_adds_page_start_from_page():
    chunk = {"chunk_id": "x", "page": 10, "spec": "23.501",
             "section": "4", "text": "test"}
    n = _normalise(chunk)
    assert n["page_start"] == 10
    assert n["page_end"] == 10

def test_normalise_keeps_existing_page_start():
    chunk = {"chunk_id": "x", "page_start": 5, "page_end": 7,
             "spec": "23.501", "section": "4", "text": "test"}
    n = _normalise(chunk)
    assert n["page_start"] == 5
    assert n["page_end"] == 7


# ── unit: expand_chunk ────────────────────────────────────────────────────────

def test_expand_returns_at_least_the_chunk():
    corpus = _get_corpus()
    chunk = corpus[50]
    result = expand_chunk(chunk, corpus, window=1)
    assert any(c["chunk_id"] == chunk["chunk_id"] for c in result)

def test_expand_neighbours_same_spec():
    corpus = _get_corpus()
    chunk = corpus[50]
    result = expand_chunk(chunk, corpus, window=1)
    for c in result:
        assert c["spec"] == chunk["spec"]

def test_expand_no_unrelated_sections():
    corpus = _get_corpus()
    chunk = corpus[100]
    result = expand_chunk(chunk, corpus, window=1)
    for c in result:
        assert _is_related_section(chunk["section"], c["section"])


# ── integration: build_evidence ───────────────────────────────────────────────

def _fake_reranked(n=3):
    corpus = _get_corpus()
    return [
        {**c, "rerank_score": 5.0, "rrf_score": 0.02, "score": 0.8,
         "page": c["page_start"]}
        for c in corpus[:n]
    ]

def test_build_evidence_returns_string():
    reranked = _fake_reranked()
    result = build_evidence(reranked)
    assert isinstance(result, str)
    assert len(result) > 0

def test_build_evidence_contains_section_headers():
    reranked = _fake_reranked()
    result = build_evidence(reranked)
    assert "[" in result and "§" in result

def test_build_evidence_respects_word_cap():
    from src.retrieval.context_builder import MAX_EVIDENCE_WORDS
    reranked = _fake_reranked(10)
    result = build_evidence(reranked)
    # allow one extra chunk worth of words (last chunk may slightly exceed due to truncation logic)
    assert len(result.split()) <= MAX_EVIDENCE_WORDS + 500

def test_build_evidence_no_duplicate_sections():
    reranked = _fake_reranked(5)
    # pass same chunks twice — should deduplicate
    result1 = build_evidence(reranked)
    result2 = build_evidence(reranked + reranked)
    assert result1 == result2
