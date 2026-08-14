"""
src/retrieval/hybrid.py
------------------------
Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion (RRF).

RRF formula:  score(d) = sum( 1 / (k + rank_i) )  for each ranked list i
where k=60 is the standard constant (Cormack et al., 2009).

No score weighting — RRF is rank-only, which makes it robust across
heterogeneous scoring scales (BM25 tf-idf vs cosine similarity).

Usage:
    python -m src.retrieval.hybrid
"""

from src.retrieval.bm25 import search as bm25_search
from src.retrieval.dense_search import search as dense_search

RRF_K = 60          # standard RRF constant
CANDIDATE_K = 20    # candidates fetched from each retriever


def rrf_fuse(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each list item must have a "chunk_id" key.
    Returns a deduplicated list sorted by RRF score descending,
    with the original result dict merged in.
    """
    rrf_scores: dict[str, float] = {}
    items: dict[str, dict] = {}          # chunk_id → result dict (first seen wins)

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in items:
                items[cid] = result

    fused = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    return [
        {**items[cid], "rrf_score": round(rrf_scores[cid], 6)}
        for cid in fused
    ]


def hybrid_search(
    query: str,
    top_k: int = 20,
    spec: str | None = None,
    release: str = "17",
) -> list[dict]:
    """
    Hybrid retrieval: fetch CANDIDATE_K from BM25 and dense, fuse with RRF.

    Returns top_k results sorted by RRF score.
    """
    bm25_results  = bm25_search(query, top_k=CANDIDATE_K)
    dense_results = dense_search(query, top_k=CANDIDATE_K, spec=spec, release=release)

    fused = rrf_fuse([bm25_results, dense_results])
    return fused[:top_k]


# ── CLI: side-by-side comparison ──────────────────────────────────────────────

TEST_QUERIES = [
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
]

SHOW_TOP = 5   # how many results to print per method


def _fmt(results: list[dict], score_key: str) -> str:
    lines = []
    for r in results[:SHOW_TOP]:
        score = r.get(score_key, r.get("score", 0))
        lines.append(
            f"    [{score_key}={score}]  "
            f"spec={r['spec']}  sec={r['section']}  p={r['page']}\n"
            f"      {' '.join(r['text'].split()[:18])} ..."
        )
    return "\n".join(lines)


def main():
    for query in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        bm25    = bm25_search(query, top_k=SHOW_TOP)
        dense   = dense_search(query, top_k=SHOW_TOP)
        hybrid  = hybrid_search(query, top_k=SHOW_TOP)

        print(f"\n── BM25 only ──────────────────────────────────────────────────")
        print(_fmt(bm25, "score"))

        print(f"\n── Dense only ─────────────────────────────────────────────────")
        print(_fmt(dense, "score"))

        print(f"\n── Hybrid (RRF) ───────────────────────────────────────────────")
        print(_fmt(hybrid, "rrf_score"))


if __name__ == "__main__":
    main()
