"""
tests/test_retrieval_eval.py
-----------------------------
Tests for src/evaluation/retrieval_eval.py
"""

import json
from pathlib import Path
from src.evaluation.retrieval_eval import compute_metrics, _rank_of_hit

RESULTS_PATH = Path("data/retrieval_results.json")


# ── unit: _rank_of_hit ────────────────────────────────────────────────────────

def test_rank_hit_at_1():
    results = [{"spec": "23.501", "section": "4.2.2"}]
    assert _rank_of_hit(results, "23.501", "4.2.2") == 1

def test_rank_hit_at_3():
    results = [
        {"spec": "23.501", "section": "4.2.1"},
        {"spec": "23.501", "section": "4.2.3"},
        {"spec": "23.501", "section": "4.2.2"},
    ]
    assert _rank_of_hit(results, "23.501", "4.2.2") == 3

def test_rank_no_hit():
    results = [{"spec": "23.501", "section": "4.2.1"}]
    assert _rank_of_hit(results, "23.501", "4.2.2") is None

def test_rank_wrong_spec():
    results = [{"spec": "23.502", "section": "4.2.2"}]
    assert _rank_of_hit(results, "23.501", "4.2.2") is None


# ── unit: compute_metrics ─────────────────────────────────────────────────────

def test_perfect_hit1():
    ranks = [1, 1, 1, 1]
    m = compute_metrics(ranks, [1, 3, 5])
    assert m["hit@1"] == 1.0
    assert m["mrr"]   == 1.0

def test_no_hits():
    ranks = [None, None, None]
    m = compute_metrics(ranks, [1, 3, 5])
    assert m["hit@1"] == 0.0
    assert m["mrr"]   == 0.0

def test_hit_at_3_not_1():
    ranks = [3, 3, 3, 3]
    m = compute_metrics(ranks, [1, 3, 5])
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0

def test_mrr_calculation():
    ranks = [1, 2, None]          # MRR = (1 + 0.5 + 0) / 3
    m = compute_metrics(ranks, [1])
    assert abs(m["mrr"] - 0.5) < 0.001


# ── integration: results file ─────────────────────────────────────────────────

def test_results_file_exists():
    assert RESULTS_PATH.exists(), "Run: python -m src.evaluation.retrieval_eval"

def test_results_has_all_systems():
    data = json.loads(RESULTS_PATH.read_text())
    for system in ("dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"):
        assert system in data["metrics"]

def test_results_has_all_metrics():
    data = json.loads(RESULTS_PATH.read_text())
    for system in data["metrics"].values():
        for key in ("hit@1", "hit@3", "hit@5", "mrr"):
            assert key in system
            assert 0.0 <= system[key] <= 1.0

def test_hybrid_rerank_mrr_beats_bm25():
    data = json.loads(RESULTS_PATH.read_text())
    assert data["metrics"]["hybrid_rerank"]["mrr"] > data["metrics"]["bm25_only"]["mrr"]

def test_per_question_count():
    data = json.loads(RESULTS_PATH.read_text())
    assert len(data["per_question"]) == 28
