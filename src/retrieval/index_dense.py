"""
src/retrieval/index_dense.py
-----------------------------
Indexes all chunks + dense embeddings into Chroma (in-memory vector store).

Collection : 3gpp_r17_5gcore
Vector dimension: 384 (all-MiniLM-L6-v2)
Total indexed: ~1,980 chunks from official 3GPP specs

Usage:
    python -m src.retrieval.index_dense
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.retrieval.chroma_db import get_collection
from src.utils.config import EMBEDDING_MODEL

_log = logging.getLogger(__name__)

COLLECTION_NAME = "3gpp_r17_5gcore"
VECTOR_DIM = 384  # all-MiniLM-L6-v2


def index(chunks_by_id: dict, embeddings: np.ndarray, emb_ids: list[str]) -> int:
    """
    Upsert chunks into Chroma.

    Args:
        chunks_by_id:  {chunk_id: chunk_dict}
        embeddings:    (N, 384) array of pre-computed vectors
        emb_ids:       list of chunk IDs (matching embeddings order)

    Returns:
        Number of new chunks indexed this run
    """
    collection = get_collection()

    total = 0
    batch_size = 100

    for batch_start in range(0, len(emb_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(emb_ids))
        batch_ids = emb_ids[batch_start:batch_end]
        batch_embeddings = embeddings[batch_start:batch_end]

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for chunk_id, emb in zip(batch_ids, batch_embeddings):
            if chunk_id in chunks_by_id:
                chunk = chunks_by_id[chunk_id]
                ids.append(chunk_id)
                documents.append(chunk.get("text", ""))

                # Build metadata
                metadata = {
                    "spec": chunk.get("spec", ""),
                    "release": str(chunk.get("release", "17")),
                    "section": chunk.get("section", ""),
                    "section_title": chunk.get("section_title", ""),
                    "page": str(chunk.get("page", "")),
                    "page_start": str(chunk.get("page_start", "")),
                    "page_end": str(chunk.get("page_end", "")),
                    "parent_section": chunk.get("parent_section", ""),
                    "source_type": chunk.get("source_type", "3gpp_official"),
                    "document": chunk.get("document", ""),
                }
                metadatas.append(metadata)
                embeddings_list.append(emb.tolist())

        # Insert into Chroma
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list,
            )
            total += len(ids)
            _log.info(f"Indexed batch {batch_start // batch_size + 1}: {len(ids)} chunks")

    return total


def main():
    """
    Load chunks and embeddings from disk, then index into Chroma.
    """
    print("=" * 70)
    print("INDEXING: Chunks + Embeddings → Chroma")
    print("=" * 70)

    # Load chunks
    chunks_file = Path("data/chunks.jsonl")
    print(f"\n[1/3] Loading chunks from {chunks_file}...")
    chunks_by_id = {}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunks_by_id[chunk["chunk_id"]] = chunk
    print(f"✓ Loaded {len(chunks_by_id)} chunks")

    # Load embeddings
    embeddings_file = Path("data/embeddings.npy")
    print(f"\n[2/3] Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    print(f"✓ Loaded embeddings: shape {embeddings.shape}")

    # Load embedding IDs
    emb_ids_file = Path("data/embedding_ids.json")
    print(f"\n[3/3] Loading embedding IDs from {emb_ids_file}...")
    with open(emb_ids_file, encoding="utf-8") as f:
        emb_ids = json.load(f)
    print(f"✓ Loaded {len(emb_ids)} embedding IDs")

    # Verify counts match
    if len(chunks_by_id) != len(embeddings) or len(embeddings) != len(emb_ids):
        print(f"✗ Count mismatch: chunks={len(chunks_by_id)}, emb={len(embeddings)}, ids={len(emb_ids)}")
        return

    print(f"\n[4/4] Indexing into Chroma...")
    indexed = index(chunks_by_id, embeddings, emb_ids)

    # Verify
    collection = get_collection()
    count = collection.count()

    print("\n" + "=" * 70)
    print(f"✓ Indexed {indexed} chunks into Chroma")
    print(f"✓ Chroma collection now contains {count} total records")
    print("=" * 70)


if __name__ == "__main__":
    main()
