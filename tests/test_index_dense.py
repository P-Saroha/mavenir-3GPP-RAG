"""
tests/test_index_dense.py
--------------------------
Tests for src/retrieval/index_dense.py (Chroma indexing).
Requires: Chroma collection pre-populated.
"""

import json
import numpy as np
from pathlib import Path
from src.retrieval.chroma_db import get_collection, COLLECTION_NAME
from src.retrieval.index_dense import VECTOR_DIM

IDS_PATH = Path("data/embedding_ids.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")


def test_collection_exists():
    collection = get_collection()
    count = collection.count()
    assert count > 0, f"Collection '{COLLECTION_NAME}' is empty"


def test_point_count():
    collection = get_collection()
    count = collection.count()
    assert count >= 1972, f"Expected ~1972 points, got {count}"


def test_metadata_fields_present():
    # Fetch one chunk
    collection = get_collection()
    results = collection.get(limit=1, include=["metadatas"])
    
    assert results and len(results["metadatas"]) > 0, "No chunks found"
    metadata = results["metadatas"][0]
    
    for field in ("spec", "release", "section", "section_title", 
                  "page", "page_end", "text", "source_type"):
        assert field in metadata, f"Missing metadata field: {field}"


def test_query_returns_results():
    collection = get_collection()
    embeddings = np.load(EMBEDDINGS_PATH)
    vector = embeddings[0].tolist()
    
    results = collection.query(
        query_embeddings=[vector],
        n_results=5,
        include=["metadatas", "documents", "distances"]
    )
    
    assert results and len(results["ids"]) > 0
    assert len(results["ids"][0]) <= 5


def test_top_result_score_is_high():
    collection = get_collection()
    embeddings = np.load(EMBEDDINGS_PATH)
    vector = embeddings[10].tolist()
    
    results = collection.query(
        query_embeddings=[vector],
        n_results=1,
        include=["distances"]
    )
    
    # Chroma returns distances, convert to similarity
    distance = results["distances"][0][0]
    similarity = 1 - distance
    assert similarity >= 0.95, f"Top result similarity should be high: {similarity}"


def test_vector_dimension():
    embeddings = np.load(EMBEDDINGS_PATH)
    assert embeddings.shape[1] == VECTOR_DIM, f"Expected {VECTOR_DIM}-dim vectors, got {embeddings.shape[1]}"
