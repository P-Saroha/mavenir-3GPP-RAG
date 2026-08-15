"""
tests/test_answer_eval.py
--------------------------
Tests for src/evaluation/answer_eval.py.
All tests are deterministic — no LLM calls.
"""

from src.evaluation.answer_eval import (
    keyword_overlap, citation_hits_gold, _tokens,
    _compute_output, MIN_KEYWORD_OVERLAP,
)


# ── _tokens ───────────────────────────────────────────────────────────────────

def test_tokens_lowercases():
    assert "amf" in _tokens("The AMF handles mobility")

def test_tokens_removes_stopwords():
    t = _tokens("the is a to and or")
    assert len(t) == 0

def test_tokens_removes_short_words():
    assert "is" not in _tokens("is it ok")

def test_tokens_keeps_technical_terms():
    t = _tokens("AMF SMF UPF NSSAI PDU")
    assert "amf" in t and "smf" in t and "upf" in t


# ── keyword_overlap ───────────────────────────────────────────────────────────

def test_perfect_overlap():
    score = keyword_overlap("AMF handles mobility management", "AMF handles mobility management")
    assert score == 1.0

def test_zero_overlap():
    score = keyword_overlap("chocolate cake recipe", "AMF handles mobility")
    assert score == 0.0

def test_partial_overlap():
    score = keyword_overlap("The AMF manages mobility", "AMF handles mobility management")
    assert 0.0 < score < 1.0

def test_empty_summary():
    assert keyword_overlap("some answer", "") == 0.0

def test_above_threshold():
    # answer contains most of the summary tokens
    answer  = "The AMF handles access and mobility management for 5G UE"
    summary = "AMF handles access mobility management"
    assert keyword_overlap(answer, summary) >= MIN_KEYWORD_OVERLAP


# ── citation_hits_gold ────────────────────────────────────────────────────────

def test_citation_hit():
    citations = [{"spec": "23.501", "section": "4.2.2", "page": 37}]
    assert citation_hits_gold(citations, "23.501", "4.2.2") is True

def test_citation_miss_wrong_spec():
    citations = [{"spec": "23.502", "section": "4.2.2"}]
    assert citation_hits_gold(citations, "23.501", "4.2.2") is False

def test_citation_miss_wrong_section():
    citations = [{"spec": "23.501", "section": "4.2.3"}]
    assert citation_hits_gold(citations, "23.501", "4.2.2") is False

def test_citation_hit_in_list():
    citations = [
        {"spec": "23.501", "section": "4.2.1"},
        {"spec": "23.501", "section": "4.2.2"},   # gold
        {"spec": "23.501", "section": "4.2.3"},
    ]
    assert citation_hits_gold(citations, "23.501", "4.2.2") is True

def test_citation_empty_list():
    assert citation_hits_gold([], "23.501", "4.2.2") is False


# ── _compute_output ───────────────────────────────────────────────────────────

def _make_row(qid, spec, section, supported, overlap, cit_hit,
              abstained=False, category="architecture"):
    return {
        "id": qid, "category": category,
        "question": "test?", "expected_spec": spec, "expected_section": section,
        "expected_summary": "summary", "answer": "answer",
        "supported": supported, "sources": [],
        "keyword_overlap": overlap, "citation_hit": cit_hit,
        "wrongly_abstained": abstained,
    }

def _make_unans_row(qid, refused):
    return {
        "id": qid, "category": "unanswerable",
        "question": "test?", "expected_spec": None, "expected_section": None,
        "expected_summary": "out of scope", "answer": "cannot answer",
        "supported": not refused, "sources": [],
        "keyword_overlap": None, "citation_hit": None,
        "correctly_refused": refused,
    }


def test_compute_perfect():
    rows = [_make_row(f"q{i}", "23.501", "4.2", True, 0.8, True) for i in range(4)]
    rows += [_make_unans_row("u1", True), _make_unans_row("u2", True)]
    out = _compute_output(rows)
    m = out["metrics"]
    assert m["correctness"] == 1.0
    assert m["citation_accuracy"] == 1.0
    assert m["abstention_accuracy"] == 1.0
    assert m["unsupported_rate"] == 0.0


def test_compute_all_wrong():
    rows = [_make_row(f"q{i}", "23.501", "4.2", True, 0.0, False) for i in range(4)]
    rows += [_make_unans_row("u1", False)]
    out = _compute_output(rows)
    m = out["metrics"]
    assert m["correctness"] == 0.0
    assert m["citation_accuracy"] == 0.0
    assert m["abstention_accuracy"] == 0.0


def test_compute_unsupported_rate():
    rows = [
        _make_row("q1", "23.501", "4.2", False, 0.0, False, abstained=True),
        _make_row("q2", "23.501", "4.2", True, 0.8, True,  abstained=False),
    ]
    out = _compute_output(rows)
    assert out["metrics"]["unsupported_rate"] == 0.5


def test_compute_output_keys():
    rows = [_make_row("q1", "23.501", "4.2", True, 0.5, True)]
    out = _compute_output(rows)
    for key in ("correctness", "mean_keyword_overlap",
                "citation_accuracy", "unsupported_rate"):
        assert key in out["metrics"]


# ── results file sanity (if already generated) ───────────────────────────────

def test_results_file_if_exists():
    from pathlib import Path
    import json
    p = Path("data/answer_results.json")
    if not p.exists():
        return   # not yet generated — skip gracefully
    data = json.loads(p.read_text())
    assert "metrics" in data
    assert "per_question" in data
    m = data["metrics"]
    for k in ("correctness", "citation_accuracy", "unsupported_rate"):
        assert 0.0 <= m[k] <= 1.0
