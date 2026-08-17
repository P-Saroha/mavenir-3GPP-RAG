"""
src/retrieval/qdrant_db.py
--------------------------
Minimal Qdrant helpers:
  - get_client()        returns a QdrantClient pointed at QDRANT_URL
  - create_collection() creates the collection if it does not exist
  - health_check()      returns True when Qdrant is reachable
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.utils.config import QDRANT_URL

_log = logging.getLogger(__name__)

COLLECTION_NAME = "3gpp_r17_5gcore"


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def create_collection(client: QdrantClient, vector_size: int) -> None:
    """Create the collection only if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}' (dim={vector_size})")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")


def health_check(client: QdrantClient) -> bool:
    """Return True if Qdrant responds to a collections list request."""
    try:
        client.get_collections()
        return True
    except Exception as e:
        print(f"Qdrant health check failed: {e}")
        return False
