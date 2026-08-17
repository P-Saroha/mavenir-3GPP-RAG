"""
src/ingestion/ingest.py
------------------------
Single entry point for the full ingestion pipeline.

Steps:
  1. Parse  — extract sections from any new PDFs in data/pdfs/
  2. Chunk  — split/merge sections into retrieval-ready chunks
  3. Embed  — encode chunks with nomic-embed-text-v1.5  (skips unchanged chunks)
  4. Index  — upsert vectors + metadata into Qdrant

Adding a new document:
  1. Drop the PDF into data/pdfs/
  2. Run:  python -m src.ingestion.ingest
  Existing documents are NOT re-processed. Only the new PDF goes through
  all four stages. The Qdrant upsert is idempotent — safe to re-run.

Usage:
    python -m src.ingestion.ingest           # full pipeline
    python -m src.ingestion.ingest --check   # verify counts without ingesting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PDF_DIR        = Path("data/pdfs")
PARSED_PATH    = Path("data/parsed.jsonl")
CHUNKS_PATH    = Path("data/chunks.jsonl")
EMBEDDINGS_PATH= Path("data/embeddings.npy")
IDS_PATH       = Path("data/embedding_ids.json")


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _)


def check_status():
    """Print current pipeline state without running anything."""
    pdfs    = sorted(PDF_DIR.glob("*.pdf"))
    parsed  = _count(PARSED_PATH)
    chunks  = _count(CHUNKS_PATH)
    emb_ids = len(json.loads(IDS_PATH.read_text())) if IDS_PATH.exists() else 0

    print("── Pipeline status ──────────────────────────────────────")
    print(f"  PDFs in data/pdfs/     : {len(pdfs)}")
    for p in pdfs:
        print(f"    {p.name}")
    print(f"  Parsed sections        : {parsed}")
    print(f"  Chunks                 : {chunks}")
    print(f"  Embeddings             : {emb_ids}")

    try:
        from src.retrieval.index_dense import get_client, COLLECTION_NAME
        client = get_client()
        count = client.count(collection_name=COLLECTION_NAME).count
        print(f"  Qdrant points          : {count}")
        print(f"  Qdrant collection      : {COLLECTION_NAME}")
    except Exception as e:
        print(f"  Qdrant                 : unreachable ({e})")
    print("─────────────────────────────────────────────────────────")


def step_parse():
    print("\n[1/4] Parsing PDFs ...")
    from src.ingestion.parser import main as parse_main
    parse_main()


def step_chunk():
    print("\n[2/4] Chunking sections ...")
    from src.ingestion.chunker import main as chunk_main
    chunk_main()


def step_embed():
    print("\n[3/4] Embedding chunks ...")
    print("      NOTE: embedding is slow on CPU (~2h for 2k chunks).")
    print("      Use Google Colab with T4 GPU for the first run (~3 min).")
    print("      The cache means only NEW/CHANGED chunks are re-encoded.")
    from src.retrieval.embedder import main as embed_main
    embed_main()


def step_index():
    print("\n[4/4] Indexing into Qdrant ...")
    from src.retrieval.index_dense import main as index_main
    index_main()


def run_pipeline(skip_embed: bool = False):
    step_parse()
    step_chunk()
    if skip_embed:
        print("\n[3/4] Skipping embedding (--skip-embed flag set).")
        print("      Copy embeddings.npy + embedding_ids.json to data/ then re-run.")
    else:
        step_embed()
    step_index()
    print("\n✓ Ingestion complete.")
    check_status()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full ingestion pipeline: parse → chunk → embed → index"
    )
    parser.add_argument("--check", action="store_true",
                        help="Print pipeline status without ingesting")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding step (use when embeddings already exist)")
    args = parser.parse_args()

    if args.check:
        check_status()
    else:
        run_pipeline(skip_embed=args.skip_embed)


if __name__ == "__main__":
    main()
