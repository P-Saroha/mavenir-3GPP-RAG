"""
src/retrieval/dense_search.py
------------------------------
Dense retrieval: embed a query then search Qdrant.

The embedding model is loaded once at module level (singleton).
Supports optional filtering by spec (e.g. "23.501") and release (default "17").

Result structure matches BM25:
    chunk_id, score, spec, section, page, text
    + extra metadata: release, version, section_title, parent_section, page_end

Usage:
    python -m src.retrieval.dense_search
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.utils.config import QDRANT_URL, EMBEDDING_MODEL
from src.retrieval.index_dense import COLLECTION_NAME

QUERY_PREFIX = "search_query: "

# ── singletons (loaded once per process) ─────────────────────────────────────
_model: SentenceTransformer | None = None
_client: QdrantClient | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


# ── search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 5,
    spec: str | None = None,
    release: str = "17",
) -> list[dict]:
    """
    Embed the query and retrieve top_k results from Qdrant.

    Args:
        query:   natural-language question
        top_k:   number of results to return
        spec:    optional filter e.g. "23.501", "23.502", "23.503"
        release: 3GPP release number filter (default "17")

    Returns:
        list of dicts with keys:
        chunk_id, score, spec, release, version,
        section, section_title, parent_section,
        page, page_end, text
    """
    model = _get_model()
    client = _get_client()

    # embed the query with the correct prefix
    vector = model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    # build payload filter
    conditions = [FieldCondition(key="release", match=MatchValue(value=release))]
    if spec:
        conditions.append(FieldCondition(key="spec", match=MatchValue(value=spec)))

    payload_filter = Filter(must=conditions)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=payload_filter,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for hit in response.points:
        p = hit.payload
        results.append({
            "chunk_id":       p["chunk_id"],
            "score":          round(hit.score, 4),
            "spec":           p["spec"],
            "release":        p["release"],
            "version":        p["version"],
            "section":        p["section"],
            "section_title":  p["section_title"],
            "parent_section": p["parent_section"],
            "page":           p["page_start"],
            "page_end":       p["page_end"],
            "text":           p["text"],
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
