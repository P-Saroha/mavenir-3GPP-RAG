"""
src/retrieval/reranker.py
--------------------------
Cross-encoder reranker using cross-encoder/ms-marco-MiniLM-L6-v2.

Takes the top 20 hybrid candidates, scores each (query, passage) pair,
and returns the top 8 sorted by reranker score.

The model is a process-level singleton — loaded once, reused every call.

Usage:
    python -m src.retrieval.reranker
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.utils.config import RERANKER_MODEL
from src.retrieval.hybrid import hybrid_search

RERANK_INPUT_K  = 20   # candidates fed to the reranker
RERANK_OUTPUT_K = 8    # results returned after reranking

# ── singleton ─────────────────────────────────────────────────────────────────
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── rerank ────────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RERANK_OUTPUT_K,
) -> list[dict]:
    """
    Score each (query, passage) pair with the cross-encoder.
    Returns top_k results with a 'rerank_score' field added.

    Args:
        query:      the user's question
        candidates: list of result dicts (must have a 'text' key)
        top_k:      how many to return after reranking
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)   # returns numpy array

    # attach score and sort descending
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, candidate in scored[:top_k]:
        results.append({**candidate, "rerank_score": round(float(score), 4)})
    return results


def retrieve_and_rerank(
    query: str,
    top_k: int = RERANK_OUTPUT_K,
    spec: str | None = None,
    release: str = "17",
) -> list[dict]:
    """
    Full pipeline shortcut:
        hybrid_search(top 20) → rerank → top_k
    """
    candidates = hybrid_search(query, top_k=RERANK_INPUT_K, spec=spec, release=release)
    return rerank(query, candidates, top_k=top_k)


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
]


def main():
    for query in TEST_QUERIES:
        print(f"\n{'=' * 65}")
        print(f"QUERY: {query}")
        print("=" * 65)
        results = retrieve_and_rerank(query)
        for rank, r in enumerate(results, start=1):
            preview = " ".join(r["text"].split()[:20])
            print(f"  {rank}. [score={r['rerank_score']:7.4f}]  "
                  f"spec={r['spec']}  sec={r['section']}  p={r['page']}")
            print(f"     {preview} ...")
        print()


if __name__ == "__main__":
    main()
