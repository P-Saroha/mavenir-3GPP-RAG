"""
src/retrieval/bm25.py
----------------------
BM25 sparse retrieval over 3GPP chunks.

The index is built once from chunks.jsonl and cached to disk as a pickle file.
Subsequent calls load from cache — no rebuild needed unless chunks change.

Usage:
    python -m src.retrieval.bm25
"""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/chunks.jsonl")
CACHE_PATH  = Path("data/bm25_cache.pkl")   # stores (chunks, bm25 index)


# ── text helpers ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ── index build / load ────────────────────────────────────────────────────────

def _build_index(chunks: list[dict]) -> BM25Okapi:
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def load_index() -> tuple[list[dict], BM25Okapi]:
    """
    Return (chunks, bm25_index).
    Loads from cache if available, otherwise builds and saves it.
    """
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            chunks, index = pickle.load(f)
        print(f"BM25 index loaded from cache ({len(chunks)} chunks)")
        return chunks, index

    print("Building BM25 index ...")
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    index  = _build_index(chunks)
    with CACHE_PATH.open("wb") as f:
        pickle.dump((chunks, index), f)
    print(f"BM25 index built and cached ({len(chunks)} chunks)")
    return chunks, index


# ── search ────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the BM25 index.
    Returns a list of dicts with: chunk_id, score, spec, section, page, text.
    """
    chunks, index = load_index()
    tokens = tokenize(query)
    scores = index.get_scores(tokens)

    # get top_k indices sorted by score descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        c = chunks[i]
        results.append({
            "chunk_id": c["chunk_id"],
            "score":    round(float(scores[i]), 4),
            "spec":     c["spec"],
            "section":  c["section"],
            "page":     c["page_start"],
            "text":     c["text"],
        })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    query = "What is the role of the AMF?"
    print(f"Query: {query}\n")
    results = search(query, top_k=5)
    print(f"Top {len(results)} BM25 results:")
    print("-" * 60)
    for r in results:
        print(f"  score={r['score']}  spec={r['spec']}  "
              f"section={r['section']}  page={r['page']}")
        print(f"  text : {' '.join(r['text'].split()[:25])} ...")
        print()


if __name__ == "__main__":
    main()
