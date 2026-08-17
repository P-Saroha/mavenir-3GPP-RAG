"""
src/retrieval/dense_search_chroma.py
------------------------------------
Dense retrieval using Chroma instead of Qdrant.

The embedding model is loaded once at module level (singleton).
Supports optional filtering by spec (e.g. "23.501") and release (default "17").

Result structure matches the old Qdrant version:
    chunk_id, score, spec, section, page, text, etc.

Usage:
    python -m src.retrieval.dense_search_chroma
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from src.utils.config import EMBEDDING_MODEL
from src.retrieval.chroma_db import get_collection

_log = logging.getLogger(__name__)

QUERY_PREFIX = "search_query: "

# ── singleton (loaded once per process) ────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _model


# ── search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 5,
    spec: str | None = None,
    release: str = "17",
) -> list[dict]:
    """
    Embed the query and retrieve top_k results from Chroma.

    Args:
        query:   natural-language question
        top_k:   number of results to return
        spec:    optional filter e.g. "23.501", "23.502", "23.503"
        release: 3GPP release number filter (default "17")

    Returns:
        list of dicts with keys:
        chunk_id, score, spec, release, section, section_title,
        page, page_end, text, source_type, document
    """
    model = _get_model()
    collection = get_collection()

    # Embed the query with the correct prefix
    vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    _log.info(f"Query vector dimension: {len(vector)}")

    # Build where filter for Chroma
    where_filter = None
    if spec:
        where_filter = {"spec": {"$eq": spec}}

    # Query Chroma
    try:
        response = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where_filter,
            include=["embeddings", "documents", "metadatas", "distances"]
        )
    except Exception as e:
        _log.error(f"Chroma query failed: {e}")
        return []

    results = []
    
    if response and response.get("ids") and len(response["ids"]) > 0:
        ids = response["ids"][0]  # First (only) query
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            # Chroma returns distances, convert to similarity score
            score = 1 - distances[i] if i < len(distances) else 0
            
            results.append({
                "chunk_id":       chunk_id,
                "score":          round(score, 4),
                "spec":           metadata.get("spec", ""),
                "section":        metadata.get("section", ""),
                "section_title":  metadata.get("section_title", ""),
                "parent_section": metadata.get("parent_section", ""),
                "page":           metadata.get("page_start", metadata.get("page", "")),
                "page_end":       metadata.get("page_end", ""),
                "text":           text,
                "source_type":    metadata.get("source_type", "3gpp_official"),
                "document":       metadata.get("document", ""),
            })
    
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
]


def main():
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = search(query, top_k=5)
        for r in results:
            print(f"  score={r['score']}  spec={r['spec']}  "
                  f"section={r['section']}  page={r['page']}")
            print(f"  title: {r['section_title']}")
            print(f"  text : {' '.join(r['text'].split()[:20])} ...")
            print()


if __name__ == "__main__":
    main()
