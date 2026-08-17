"""
tests/test_chroma_db.py
-----------------------
Health-check test for Chroma (replaces test_qdrant_db.py).
No external services needed!
"""

import pytest
from src.retrieval.chroma_db import get_collection, health_check, COLLECTION_NAME


def test_health_check():
    assert health_check(), "Chroma is not accessible"


def test_collection_exists():
    collection = get_collection()
    assert collection is not None, "Failed to get collection"


def test_collection_has_data():
    collection = get_collection()
    count = collection.count()
    assert count > 0, f"Collection '{COLLECTION_NAME}' is empty"


def test_collection_name_correct():
    collection = get_collection()
    assert collection.name == COLLECTION_NAME
