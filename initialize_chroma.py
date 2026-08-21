#!/usr/bin/env python
"""
initialize_chroma.py
--------------------
Initialize Chroma vector database from pre-computed embeddings.

Run this ONCE after extracting the TAR file.
Time: ~1-2 minutes

What it does:
1. Loads chunks.jsonl (2,330 chunks)
2. Loads embeddings.npy (pre-computed vectors)
3. Indexes everything into Chroma
4. Creates ./chroma_data/ directory

After this, the system is ready to query!
"""

import json
import shutil
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("Chroma Initialization from Pre-computed Embeddings")
print("=" * 80)

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Paths
CHUNKS_FILE = Path("data/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/embeddings.npy")
CHROMA_DIR = Path("chroma_data")

# Validate files exist
if not CHUNKS_FILE.exists():
    print(f"ERROR: {CHUNKS_FILE} not found")
    sys.exit(1)

if not EMBEDDINGS_FILE.exists():
    print(f"ERROR: {EMBEDDINGS_FILE} not found")
    sys.exit(1)

print(f"\n1. Loading chunks from {CHUNKS_FILE}...")
chunks = []
with open(CHUNKS_FILE, encoding="utf-8", errors="ignore") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"   Loaded {len(chunks)} chunks")

# Count by spec
by_spec = {}
for c in chunks:
    spec = c.get("spec", "unknown")
    by_spec[spec] = by_spec.get(spec, 0) + 1

print(f"\n   Breakdown by spec:")
for spec in sorted(by_spec.keys()):
    print(f"     {spec}: {by_spec[spec]}")

print(f"\n2. Loading embeddings from {EMBEDDINGS_FILE}...")
embeddings_array = np.load(EMBEDDINGS_FILE)
print(f"   Loaded embeddings: shape {embeddings_array.shape}")

if embeddings_array.shape[0] != len(chunks):
    print(f"ERROR: Mismatch - {embeddings_array.shape[0]} embeddings vs {len(chunks)} chunks")
    sys.exit(1)

print(f"\n3. Clearing old Chroma data...")
if CHROMA_DIR.exists():
    shutil.rmtree(CHROMA_DIR)
    print(f"   Removed {CHROMA_DIR}")

print(f"\n4. Initializing Chroma at {CHROMA_DIR}...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name="3gpp_r17_5gcore",
    metadata={"hnsw:space": "cosine"}
)
print(f"   Collection created: 3gpp_r17_5gcore")

print(f"\n5. Indexing {len(chunks)} chunks...")
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch_end = min(i + batch_size, len(chunks))
    batch_chunks = chunks[i:batch_end]
    batch_embeddings = embeddings_array[i:batch_end]
    
    ids = [c["chunk_id"] for c in batch_chunks]
    embeddings = [emb.tolist() for emb in batch_embeddings]
    metadatas = [
        {
            "spec": c.get("spec", "unknown"),
            "section": c.get("section", ""),
            "page_start": str(c.get("page_start", 0)),
            "page_end": str(c.get("page_end", 0)),
        }
        for c in batch_chunks
    ]
    documents = [c.get("text", "") for c in batch_chunks]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    
    print(f"   ✓ Indexed {batch_end}/{len(chunks)}")

print(f"\n" + "=" * 80)
print(f"SUCCESS: Chroma initialized with {len(chunks)} chunks")
print(f"=" * 80)
print(f"\nNext steps:")
print(f"  1. Start backend:  uvicorn src.api:app --port 8000 --reload")
print(f"  2. Start frontend: streamlit run app.py")
print(f"  3. Open http://localhost:8501 in browser")
print(f"\nReady to query!")
