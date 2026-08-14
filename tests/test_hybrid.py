"""
tests/test_hybrid.py
---------------------
Tests for src/retrieval/hybrid.py
"""

from src.retrieval.hybrid import hybrid_search, rrf_fuse


# ── unit tests for RRF ───────────────────────────────────────────────────────

def test_rrf_deduplicates():
    list_a = [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}]
    list_b = [{"chunk_id": "b", "text": "y"}, {"chunk_id": "c", "text": "z"}]
    result = rrf_fuse([list_a, list_b])
    ids = [r["chunk_id"] for r in result]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in RRF output"


def test_rrf_scores_descending():
    list_a = [{"chunk_id": str(i), "text": "x"} for i in range(5)]
    list_b = [{"chunk_id": str(i), "text": "x"} for i in range(5)]
    result = rrf_fuse([list_a, list_b])
    scores = [r["rrf_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_item_appearing_in_both_lists_scores_higher():
    # "overlap" appears rank-1 in both lists → should outscore "only_a" and "only_b"
    list_a = [{"chunk_id": "overlap", "text": ""}, {"chunk_id": "only_a", "text": ""}]
    list_b = [{"chunk_id": "overlap", "text": ""}, {"chunk_id": "only_b", "text": ""}]
    result = rrf_fuse([list_a, list_b])
    top_id = result[0]["chunk_id"]
    assert top_id == "overlap", f"Expected 'overlap' at top, got '{top_id}'"


def test_rrf_score_field_present():
    list_a = [{"chunk_id": "a", "text": "x"}]
    result = rrf_fuse([list_a])
    assert "rrf_score" in result[0]


# ── integration tests ────────────────────────────────────────────────────────

def test_hybrid_search_returns_results():
    results = hybrid_search("What is the role of the AMF?", top_k=10)
    assert len(results) == 10


def test_hybrid_result_fields():
    results = hybrid_search("PDU session establishment", top_k=5)
    for r in results:
        for field in ("chunk_id", "rrf_score", "spec", "section", "page", "text"):
            assert field in r, f"Missing field: {field}"


def test_hybrid_no_duplicate_chunk_ids():
    results = hybrid_search("network slicing NSSAI", top_k=20)
    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))


def test_hybrid_scores_descending():
    results = hybrid_search("UPF user plane function", top_k=10)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_spec_filter():
    results = hybrid_search("policy control PCF", top_k=10, spec="23.503")
    # dense results will all be spec=23.503; BM25 has no filter but RRF merges
    dense_specs = [r["spec"] for r in results if r["spec"] == "23.503"]
    assert len(dense_specs) > 0, "Expected some 23.503 results with spec filter"


def test_hybrid_top_k_respected():
    for k in (5, 10, 20):
        results = hybrid_search("SMF session management", top_k=k)
        assert len(results) == k
