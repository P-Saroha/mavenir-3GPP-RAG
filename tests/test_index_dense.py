"""
tests/test_index_dense.py
--------------------------
Tests for src/retrieval/index_dense.py.
Requires Qdrant running: docker compose up -d
"""

import json
import numpy as np
from pathlib import Path
from src.retrieval.index_dense import get_client, ensure_collection, COLLECTION_NAME, VECTOR_DIM

IDS_PATH = Path("data/embedding_ids.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")


def test_collection_exists():
    client = get_client()
    names = [c.name for c in client.get_collections().collections]
    assert COLLECTION_NAME in names, f"Collection '{COLLECTION_NAME}' not found"


def test_point_count():
    client = get_client()
    count = client.count(collection_name=COLLECTION_NAME).count
    assert count == 1972, f"Expected 1972 points, got {count}"


def test_payload_fields_present():
    client = get_client()
    # fetch one point with payload
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=np.load(EMBEDDINGS_PATH)[0].tolist(),
        limit=1,
        with_payload=True,
    ).points
    assert results, "No results returned"
    payload = results[0].payload
    for field in ("chunk_id", "spec", "release", "version", "section",
                  "section_title", "parent_section", "page_start", "page_end", "text"):
        assert field in payload, f"Missing payload field: {field}"


def test_query_returns_top5():
    client = get_client()
    vector = np.load(EMBEDDINGS_PATH)[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True,
    ).points
    assert len(results) == 5


def test_top_result_score_is_high():
    client = get_client()
    vector = np.load(EMBEDDINGS_PATH)[10].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
        with_payload=True,
    ).points
    assert results[0].score >= 0.99, "Top result should match its own vector"


def test_ensure_collection_idempotent():
    client = get_client()
    # calling twice must not raise
    ensure_collection(client)
    ensure_collection(client)
    names = [c.name for c in client.get_collections().collections]
    assert COLLECTION_NAME in names
