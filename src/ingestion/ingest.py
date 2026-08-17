"""
src/ingestion/ingest.py
-----------------------
Orchestrates the complete ingestion pipeline for uploaded PDFs.

Flow:
  1. Validate PDF exists and is readable
  2. Check for duplicates (by filename)
  3. Mark status: ingestion_started
  4. Parse PDF → parsed.jsonl
  5. Chunk → chunks.jsonl
  6. Embed (with cache) → embeddings.npy + embedding_ids.json
  7. Index into Chroma
  8. Mark status: ingestion_completed

Reuses all existing ingestion code (parser, chunker, embedder).

Usage (from Streamlit or CLI):
    python -m src.ingestion.ingest path/to/file.pdf
"""

import sys
import json
import numpy as np
from pathlib import Path

from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_sections, build_stats
from src.retrieval.embedder import embed, load_model, save, load_chunks
from src.ingestion.upload_handler import UploadHandler
from src.retrieval.chroma_db import get_collection, COLLECTION_NAME

# ── paths ──────────────────────────────────────────────────────────────────────
PARSED_PATH = Path("data/parsed.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
IDS_PATH = Path("data/embedding_ids.json")
CACHE_PATH = Path("data/embedding_cache.json")


def ingest_pdf(pdf_path: str) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    Returns status dict:
        {
            "success": bool,
            "message": str,
            "filename": str,
            "sections": int,
            "chunks": int,
            "error": str or None,
        }
    """
    pdf_path = Path(pdf_path)
    handler = UploadHandler()
    filename = pdf_path.name

    try:
        # ── 1. Validate file ───────────────────────────────────────────────────
        if not pdf_path.exists():
            return {
                "success": False,
                "message": f"File not found: {pdf_path}",
                "filename": filename,
                "error": "File not found",
            }

        if not pdf_path.suffix.lower() == ".pdf":
            return {
                "success": False,
                "message": f"Not a PDF file: {filename}",
                "filename": filename,
                "error": "Invalid file type",
            }

        # ── 2. Check for duplicates ────────────────────────────────────────────
        file_hash = handler.compute_hash(pdf_path)
        is_duplicate, prev_status = handler.check_duplicate(filename)
        
        if is_duplicate:
            return {
                "success": False,
                "message": f"Duplicate: {filename} was already processed (status: {prev_status})",
                "filename": filename,
                "error": "Already indexed",
            }

        # ── 3. Mark as started ─────────────────────────────────────────────────
        handler.mark_uploaded(filename, file_hash)
        handler.mark_ingestion_started(filename)

        # ── 4. Parse PDF ───────────────────────────────────────────────────────
        print(f"Parsing {filename} ...")
        sections = parse_pdf(pdf_path, source_type="uploaded")
        if not sections:
            raise Exception("No sections extracted from PDF")
        print(f"  > {len(sections)} sections")

        # ── 5. Append to parsed.jsonl ──────────────────────────────────────────
        PARSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PARSED_PATH.open("a", encoding="utf-8") as f:
            for sec in sections:
                f.write(json.dumps({
                    "document": sec.document,
                    "spec": sec.spec,
                    "release": sec.release,
                    "version": sec.version,
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "section": sec.section,
                    "section_title": sec.section_title,
                    "parent_section": sec.parent_section,
                    "text": sec.text,
                    "source_type": sec.source_type,
                }, ensure_ascii=False) + "\n")

        # ── 6. Chunk all sections (reload to include new ones) ────────────────
        print("Chunking all sections ...")
        all_sections = [
            json.loads(l)
            for l in PARSED_PATH.read_text(encoding="utf-8").splitlines()
        ]
        chunks = chunk_sections(all_sections)
        print(f"  > {len(chunks)} total chunks")

        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CHUNKS_PATH.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id,
                    "spec": c.spec,
                    "release": c.release,
                    "version": c.version,
                    "section": c.section,
                    "section_title": c.section_title,
                    "parent_section": c.parent_section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": c.text,
                    "source_type": c.source_type,
                    "document": c.document,
                }, ensure_ascii=False) + "\n")

        stats = build_stats(chunks)
        print(f"  > Stats: {stats}")

        # ── 7. Embed all chunks (with caching) ─────────────────────────────────
        print("Embedding chunks (may take minutes on CPU)...")
        chunks_list = load_chunks(CHUNKS_PATH)
        model = load_model()
        array, ids, cache = embed(chunks_list, model)
        save(array, ids, cache)
        print(f"  > {len(ids)} chunks embedded")

        # ── 8. Index into Chroma ──────────────────────────────────────────────
        print("Indexing into Chroma...")
        collection = get_collection()
        
        # Load all chunks and embeddings
        chunks_list = load_chunks(CHUNKS_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        embedding_ids = json.loads(IDS_PATH.read_text())
        
        # Get existing IDs from Chroma to avoid duplicates
        existing_ids_result = collection.get(include=[])
        existing_ids = set(existing_ids_result["ids"]) if existing_ids_result["ids"] else set()
        print(f"  > Chroma has {len(existing_ids)} existing chunks")
        
        # Only add NEW chunks (not already in Chroma)
        ids_to_add = []
        documents_to_add = []
        metadatas_to_add = []
        embeddings_to_add = []
        
        for chunk in chunks_list:
            chunk_id = chunk["chunk_id"]
            
            # Skip if already in Chroma
            if chunk_id in existing_ids:
                continue
            
            ids_to_add.append(chunk_id)
            documents_to_add.append(chunk["text"])
            metadatas_to_add.append({
                "spec": chunk.get("spec", ""),
                "section": chunk.get("section", ""),
                "page": str(chunk.get("page_start", "")),
                "page_start": str(chunk.get("page_start", "")),
                "page_end": str(chunk.get("page_end", "")),
                "section_title": chunk.get("section_title", ""),
                "release": str(chunk.get("release", "")),
                "source_type": chunk.get("source_type", "3gpp_official"),
                "document": chunk.get("document", ""),
            })
            
            # Find embedding for this chunk
            idx = embedding_ids.index(chunk_id)
            embeddings_to_add.append(embeddings[idx].tolist())
        
        # Add only new chunks to Chroma
        if ids_to_add:
            collection.add(
                ids=ids_to_add,
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                embeddings=embeddings_to_add
            )
            print(f"  > {len(ids_to_add)} NEW documents added to Chroma")
        else:
            print(f"  > All {len(chunks_list)} chunks already in Chroma (no duplicates)")

        # ── 9. Mark as completed ───────────────────────────────────────────────
        handler.mark_ingestion_completed(filename)

        return {
            "success": True,
            "message": f"✓ Successfully ingested {filename}",
            "filename": filename,
            "sections": len(sections),
            "chunks": len(chunks),
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        handler.mark_ingestion_failed(filename, error_msg)
        return {
            "success": False,
            "message": f"Ingestion failed: {error_msg}",
            "filename": filename,
            "error": error_msg,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.ingest <path/to/pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = ingest_pdf(pdf_path)

    print(f"\n{'='*60}")
    print(f"Result: {result['message']}")
    if not result["success"] and result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"Sections: {result.get('sections', 'N/A')}")
        print(f"Chunks: {result.get('chunks', 'N/A')}")


if __name__ == "__main__":
    main()
