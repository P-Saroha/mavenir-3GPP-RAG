"""
src/retrieval/index_dense.py
-----------------------------
Indexes all chunks + dense embeddings into Qdrant.

Collection : 3gpp_r17_5gcore
Vector     : 768-dim cosine (nomic-embed-text-v1.5)
Payload    : chunk_id, spec, release, version, section, section_title,
             parent_section, page_start, page_end, text

Usage:
    python -m src.retrieval.index_dense          # full corpus
    python -m src.retrieval.index_dense --test   # first 100 chunks only
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.utils.config import QDRANT_URL

# ── constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "3gpp_r17_5gcore"
VECTOR_DIM = 768
BATCH_SIZE = 128

CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
IDS_PATH = Path("data/embedding_ids.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _point_id(chunk_id: str) -> str:
    """Convert a hex chunk_id to a UUID string for Qdrant.
    Deterministic: same chunk always maps to the same Qdrant point ID."""
    # chunk_id is a 32-char hex string — pad to 32 bytes for UUID
    return str(uuid.UUID(chunk_id.ljust(32, "0")[:32]))
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    """Create collection if it does not already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")


def load_data(limit: int | None = None) -> tuple[list[dict], np.ndarray, list[str]]:
    """Load chunks, embeddings, and the ordered ID list."""
    chunks_by_id = {
        c["chunk_id"]: c
        for c in (json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines())
    }
    emb_ids: list[str] = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    embeddings: np.ndarray = np.load(EMBEDDINGS_PATH)

    if limit:
        emb_ids = emb_ids[:limit]
        embeddings = embeddings[:limit]

    return chunks_by_id, embeddings, emb_ids


def index(client: QdrantClient, chunks_by_id: dict, embeddings: np.ndarray,
          emb_ids: list[str]) -> int:
    """
    Upsert chunks into Qdrant using deterministic UUID point IDs.
    Chunks already in the collection are skipped (upsert is idempotent).
    Returns number of points uploaded this run.
    """
    points = []
    for i, chunk_id in enumerate(emb_ids):
        c = chunks_by_id[chunk_id]
        points.append(PointStruct(
            id=_point_id(chunk_id),        # deterministic UUID, not row index
            vector=embeddings[i].tolist(),
            payload={
                "chunk_id":      c["chunk_id"],
                "spec":          c["spec"],
                "release":       c["release"],
                "version":       c["version"],
                "section":       c["section"],
                "section_title": c["section_title"],
                "parent_section":c["parent_section"],
                "page_start":    c["page_start"],
                "page_end":      c["page_end"],
                "text":          c["text"],
            },
        ))

    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start: start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  upserted {min(start + BATCH_SIZE, len(points))}/{len(points)}", end="\r")

    print()
    return len(points)


def query_top5(client: QdrantClient, vector: list[float]) -> None:
    """Query the collection and print the top 5 results."""
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True,
    )
    results = response.points
    print("\nTop 5 results:")
    print("-" * 60)
    for r in results:
        p = r.payload
        print(f"  score={r.score:.4f}  spec={p['spec']}  "
              f"section={p['section']}  pages={p['page_start']}-{p['page_end']}")
        print(f"  title : {p['section_title']}")
        print(f"  text  : {' '.join(p['text'].split()[:20])} ...")
        print()


def main(test_mode: bool = False):
    limit = 100 if test_mode else None
    if test_mode:
        print("TEST MODE — indexing first 100 chunks only")

    client = get_client()
    ensure_collection(client)

    print("Loading data ...")
    chunks_by_id, embeddings, emb_ids = load_data(limit=limit)
    print(f"Chunks to index: {len(emb_ids)}")

    print("Indexing ...")
    total = index(client, chunks_by_id, embeddings, emb_ids)
    print(f"Stored {total} points in '{COLLECTION_NAME}'")

    # verify count in Qdrant
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Qdrant reports {count} points in collection")

    # run a sample query using the first vector
    print("\nSample query (using first chunk vector) ...")
    query_top5(client, embeddings[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Index first 100 chunks only")
    args = parser.parse_args()
    main(test_mode=args.test)
