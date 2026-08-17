"""
src/retrieval/chroma_db.py
--------------------------
Minimal Chroma helpers (replaces Qdrant):
  - get_client()        returns a Chroma client
  - get_collection()    returns the 3gpp_r17_5gcore collection
  - health_check()      returns True when Chroma is reachable
"""

import logging
import os
from pathlib import Path

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb not installed. Run: pip install chromadb")

_log = logging.getLogger(__name__)

COLLECTION_NAME = "3gpp_r17_5gcore"

# ── Persistent storage path ────────────────────────────────────────────────────
# Store Chroma data in workspace root, not in memory
CHROMA_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_data")
Path(CHROMA_DATA_DIR).mkdir(parents=True, exist_ok=True)

_log.info(f"Chroma persistent storage: {CHROMA_DATA_DIR}")


def get_client():
    """Get Chroma client with persistent storage using new API."""
    # New Chroma API (v0.4+): use PersistentClient directly
    return chromadb.PersistentClient(path=CHROMA_DATA_DIR)


def get_collection():
    """Get the 3GPP collection from Chroma."""
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def health_check() -> bool:
    """Return True if Chroma is reachable."""
    try:
        client = get_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Chroma health check passed: {count} records in {COLLECTION_NAME}")
        return count > 0
    except Exception as e:
        print(f"Chroma health check failed: {e}")
        return False
