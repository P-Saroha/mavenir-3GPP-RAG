"""
src/retrieval/mmr.py
---------------------
Maximal Marginal Relevance (MMR) step after reranking.

MMR balances relevance and diversity:
    score(d) = λ * relevance(d) - (1 - λ) * max_similarity(d, selected)

- relevance  = rerank_score (normalised to [0,1])
- similarity = cosine similarity between chunk embeddings (already normalised)
- λ = LAMBDA_PARAM  (higher → more relevance, lower → more diversity)

Embeddings are loaded once from disk — no re-encoding needed.

Usage:
    python -m src.retrieval.mmr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieval.reranker import retrieve_and_rerank

# ── config ────────────────────────────────────────────────────────────────────
LAMBDA_PARAM = 0.55     # balance between relevance (higher) and diversity (lower)
MMR_TOP_K = 6           # final chunks returned

EMBEDDINGS_PATH = Path("data/embeddings.npy")
IDS_PATH        = Path("data/embedding_ids.json")

# ── embedding index (loaded once) ─────────────────────────────────────────────
_emb_array: np.ndarray | None = None
_emb_index: dict[str, int] | None = None   # chunk_id → row index


def _get_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    global _emb_array, _emb_index
    if _emb_array is None:
        _emb_array = np.load(EMBEDDINGS_PATH)                  # (N, 768) float32
        ids: list[str] = json.loads(IDS_PATH.read_text())
        _emb_index = {cid: i for i, cid in enumerate(ids)}
    return _emb_array, _emb_index


# ── core MMR ──────────────────────────────────────────────────────────────────

def mmr(
    candidates: list[dict],
    top_k: int = MMR_TOP_K,
    lam: float = LAMBDA_PARAM,
) -> list[dict]:
    """
    Select top_k diverse chunks from candidates using MMR.

    candidates must have 'chunk_id' and 'rerank_score'.
    Returns selected chunks with an added 'mmr_score' field.
    """
    if not candidates:
        return []

    emb_array, emb_index = _get_embeddings()

    # collect embeddings for candidates that exist in the index
    vecs: list[np.ndarray] = []
    valid: list[dict] = []
    for c in candidates:
        idx = emb_index.get(c["chunk_id"])
        if idx is not None:
            vecs.append(emb_array[idx])
            valid.append(c)

    if not valid:
        return candidates[:top_k]

    V = np.array(vecs, dtype=np.float32)  # (M, 768), already unit-normalised

    # normalise rerank scores to [0, 1] for a common scale with cosine sim
    raw_scores = np.array([c["rerank_score"] for c in valid], dtype=np.float32)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        rel = (raw_scores - s_min) / (s_max - s_min)
    else:
        rel = np.ones(len(valid), dtype=np.float32)

    selected_indices: list[int] = []
    remaining = list(range(len(valid)))

    for _ in range(min(top_k, len(valid))):
        if not selected_indices:
            # first pick: highest relevance
            best = max(remaining, key=lambda i: rel[i])
        else:
            # subsequent picks: MMR criterion
            sel_vecs = V[selected_indices]             # (S, 768)
            # cosine similarity: V already normalised → dot product = cosine
            sim = V[remaining] @ sel_vecs.T            # (R, S)
            max_sim = sim.max(axis=1)                  # (R,)
            mmr_scores = lam * rel[remaining] - (1 - lam) * max_sim
            best_pos = int(np.argmax(mmr_scores))
            best = remaining[best_pos]

        selected_indices.append(best)
        remaining.remove(best)

    # build output with mmr_score = final MMR value (or relevance for first)
    results = []
    for rank, idx in enumerate(selected_indices):
        results.append({**valid[idx], "mmr_score": round(float(rel[idx]), 4)})
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

# Use a query where the same section tends to produce multiple similar chunks
TEST_QUERY = "What is network slicing and how is NSSAI used?"


def main():
    print(f"Query: {TEST_QUERY}\n")

    reranked = retrieve_and_rerank(TEST_QUERY, top_k=8)

    print("── Before MMR (top 8 reranked) ──────────────────────────────────")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. [rerank={r['rerank_score']:6.3f}]  "
              f"spec={r['spec']}  sec={r['section']}  p={r['page']}")
        print(f"     {' '.join(r['text'].split()[:15])} ...")

    print(f"\n── After MMR (top {MMR_TOP_K}, λ={LAMBDA_PARAM}) ──────────────────────────────────")
    selected = mmr(reranked)
    for i, r in enumerate(selected, 1):
        print(f"  {i}. [mmr={r['mmr_score']:.4f}]  "
              f"spec={r['spec']}  sec={r['section']}  p={r['page']}")
        print(f"     {' '.join(r['text'].split()[:15])} ...")

    before_sections = [r["section"] for r in reranked[:MMR_TOP_K]]
    after_sections  = [r["section"] for r in selected]
    print(f"\n  Sections before: {before_sections}")
    print(f"  Sections after : {after_sections}")
    unique_before = len(set(before_sections))
    unique_after  = len(set(after_sections))
    print(f"  Unique sections before: {unique_before}  after: {unique_after}")


if __name__ == "__main__":
    main()
