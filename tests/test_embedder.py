"""
tests/test_embedder.py
-----------------------
Tests for src/retrieval/embedder.py.
Does NOT re-run the full embedding — validates saved artefacts only.
"""

import json
import numpy as np
from pathlib import Path

EMBEDDINGS_PATH = Path("data/embeddings.npy")
IDS_PATH = Path("data/embedding_ids.json")
CACHE_PATH = Path("data/embedding_cache.json")
CHUNKS_PATH = Path("data/chunks.jsonl")


def test_output_files_exist():
    assert EMBEDDINGS_PATH.exists(), "Run: python -m src.retrieval.embedder"
    assert IDS_PATH.exists()
    assert CACHE_PATH.exists()


def test_embedding_shape():
    arr = np.load(EMBEDDINGS_PATH)
    ids = json.loads(IDS_PATH.read_text())
    assert arr.ndim == 2
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    assert arr.shape[1] == 384, f"Expected dim 384 (all-MiniLM-L6-v2), got {arr.shape[1]}"
    assert arr.shape[0] == len(ids), "Row count must match id list length"


def test_embeddings_normalized():
    arr = np.load(EMBEDDINGS_PATH)
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings are not unit-normalized"


def test_embedding_dtype():
    arr = np.load(EMBEDDINGS_PATH)
    assert arr.dtype == np.float32


def test_ids_match_chunks():
    ids = set(json.loads(IDS_PATH.read_text()))
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    chunk_ids = {c["chunk_id"] for c in chunks}
    assert ids == chunk_ids, "Embedding IDs don't match chunk IDs"


def test_cache_covers_all_ids():
    ids = json.loads(IDS_PATH.read_text())
    cache = json.loads(CACHE_PATH.read_text())
    for cid in ids:
        assert cid in cache, f"chunk_id missing from cache: {cid}"


def test_no_zero_vectors():
    arr = np.load(EMBEDDINGS_PATH)
    zero_rows = np.where(np.all(arr == 0, axis=1))[0]
    assert len(zero_rows) == 0, f"Found {len(zero_rows)} zero embedding vectors"
