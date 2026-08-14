"""
tests/test_qdrant_db.py
-----------------------
Health-check test for Qdrant.
Requires Docker to be running:  docker compose up -d
"""

import pytest
from src.retrieval.qdrant_db import get_client, create_collection, health_check

VECTOR_SIZE = 768  # nomic-embed-text-v1.5 output dimension


def test_health_check():
    client = get_client()
    assert health_check(client), "Qdrant is not reachable — run: docker compose up -d"


def test_create_collection_idempotent():
    client = get_client()
    # Call twice — second call must not raise
    create_collection(client, VECTOR_SIZE)
    create_collection(client, VECTOR_SIZE)
    names = [c.name for c in client.get_collections().collections]
    assert "3gpp_r17" in names
