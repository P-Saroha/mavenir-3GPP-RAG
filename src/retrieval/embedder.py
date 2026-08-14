"""
src/retrieval/embedder.py
--------------------------
Generates dense embeddings for all chunks using nomic-embed-text-v1.5.

Outputs:
    data/embeddings.npy        — float32 array, shape (N, 768)
    data/embedding_ids.json    — list of chunk_ids in row order

Cache behaviour:
    A SHA-1 fingerprint of each chunk's text is stored alongside the
    embeddings.  On the next run only chunks whose text has changed are
    re-encoded; the rest are loaded from the existing .npy file.

Usage:
    python -m src.retrieval.embedder            # full corpus
    python -m src.retrieval.embedder --test     # first 10 chunks only
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import EMBEDDING_MODEL

# ── paths ─────────────────────────────────────────────────────────────────────
CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
IDS_PATH = Path("data/embedding_ids.json")
CACHE_PATH = Path("data/embedding_cache.json")  # {chunk_id: text_sha1}

# nomic-embed-text-v1.5 prefix for corpus documents
DOCUMENT_PREFIX = "search_document: "
BATCH_SIZE = 64


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def embed(chunks: list[dict], model: SentenceTransformer) -> tuple[np.ndarray, list[str]]:
    """
    Encode chunks with caching.
    Returns (embeddings array, ordered list of chunk_ids).
    """
    # Load existing cache and embeddings if they exist
    cache: dict[str, str] = {}
    old_embs: dict[str, np.ndarray] = {}

    if CACHE_PATH.exists() and EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        old_ids: list[str] = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        old_array = np.load(EMBEDDINGS_PATH)
        old_embs = {cid: old_array[i] for i, cid in enumerate(old_ids)}
        print(f"Cache hit: {len(old_embs)} existing embeddings loaded")

    # Partition into cached vs needs encoding
    to_encode: list[dict] = []
    for c in chunks:
        sha = _sha1(c["text"])
        if c["chunk_id"] in old_embs and cache.get(c["chunk_id"]) == sha:
            pass  # will reuse from old_embs
        else:
            to_encode.append(c)

    print(f"Chunks to encode: {len(to_encode)}  (cached: {len(chunks) - len(to_encode)})")

    # Encode in batches
    new_embs: dict[str, np.ndarray] = {}
    if to_encode:
        texts = [DOCUMENT_PREFIX + c["text"] for c in to_encode]
        t0 = time.time()
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0
        print(f"Encoded {len(texts)} chunks in {elapsed:.1f}s")
        for c, vec in zip(to_encode, vectors):
            new_embs[c["chunk_id"]] = vec

    # Assemble final array in input order
    dim = next(iter(old_embs.values())).shape[0] if old_embs else (
          next(iter(new_embs.values())).shape[0] if new_embs else 768)

    ordered_ids = [c["chunk_id"] for c in chunks]
    array = np.zeros((len(chunks), dim), dtype=np.float32)
    new_cache: dict[str, str] = {}

    for i, c in enumerate(chunks):
        cid = c["chunk_id"]
        if cid in new_embs:
            array[i] = new_embs[cid]
        else:
            array[i] = old_embs[cid]
        new_cache[cid] = _sha1(c["text"])

    return array, ordered_ids, new_cache


def save(array: np.ndarray, ids: list[str], cache: dict[str, str]):
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, array)
    IDS_PATH.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def main(test_mode: bool = False):
    chunks = load_chunks(CHUNKS_PATH)
    if test_mode:
        chunks = chunks[:10]
        print(f"TEST MODE — encoding first {len(chunks)} chunks only")

    model = load_model()
    t_start = time.time()
    array, ids, cache = embed(chunks, model)
    save(array, ids, cache)

    print(f"\nChunks embedded : {len(ids)}")
    print(f"Embedding dim   : {array.shape[1]}")
    print(f"Total time      : {time.time() - t_start:.1f}s")
    print(f"Saved to        : {EMBEDDINGS_PATH}  ({EMBEDDINGS_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Encode first 10 chunks only")
    args = parser.parse_args()
    main(test_mode=args.test)
