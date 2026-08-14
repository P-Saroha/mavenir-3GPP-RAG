"""
src/rag.py
-----------
Complete end-to-end RAG pipeline.

Pipeline:
  user query
    → hybrid retrieval (BM25 + dense, top 20 each)
    → RRF fusion
    → cross-encoder reranking (top 8)
    → MMR diversity filter (top 5)
    → evidence quality gate
    → parent-context expansion
    → citation-tagged evidence build
    → LLM generation (Groq / xAI / Ollama)
    → citation validation
    → final response

Public API — one function:
    answer_question(query) -> {"answer", "sources", "supported"}

All internal details are hidden from the caller.
"""

from __future__ import annotations

from src.retrieval.hybrid   import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.mmr      import mmr
from src.retrieval.quality_gate import check
from src.retrieval.context_builder import _get_corpus, _normalise, expand_chunk
from src.generation.citations import (
    build_sourced_evidence,
    _system_prompt,
    validate_citations,
    CITATION_RE,
)
from src.generation.grok import _get_client, ACTIVE_MODEL, CANNOT_ANSWER

# ── pipeline constants ────────────────────────────────────────────────────────
HYBRID_TOP_K  = 20   # candidates from each retriever
RERANK_TOP_K  = 8    # after cross-encoder
MMR_TOP_K     = 5    # after MMR diversity filter
MAX_EVIDENCE_WORDS = 3000


def _build_expanded_evidence(mmr_chunks: list[dict]) -> tuple[str, dict]:
    """
    Run context expansion then build the citation-tagged evidence string.
    Returns (evidence_str, source_map).
    """
    corpus = _get_corpus()
    corpus_by_id = {c["chunk_id"]: c for c in corpus}

    seen: set[str] = set()
    expanded: list[dict] = []

    for chunk in mmr_chunks:
        for c in expand_chunk(_normalise(chunk, corpus_by_id), corpus, window=1):
            c = _normalise(c, corpus_by_id)
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                expanded.append(c)

    expanded.sort(key=lambda c: (c["spec"], c.get("page_start") or c.get("page", 0)))

    # word cap
    capped: list[dict] = []
    total = 0
    for c in expanded:
        w = len(c.get("text", "").split())
        if total + w > MAX_EVIDENCE_WORDS:
            break
        capped.append(c)
        total += w

    return build_sourced_evidence(capped)


def answer_question(query: str) -> dict:
    """
    Run the full RAG pipeline and return a clean response.

    Returns:
        {
            "answer":    str,          # grounded answer with [S1]..[SN] citations
            "sources":   list[dict],   # [{id, spec, release, section, page, title}]
            "supported": bool          # False if evidence was insufficient
        }
    """
    # ── 1. Hybrid retrieval + RRF ──────────────────────────────────────────────
    candidates = hybrid_search(query, top_k=HYBRID_TOP_K)

    # ── 2. Cross-encoder reranking ─────────────────────────────────────────────
    reranked = rerank(query, candidates, top_k=RERANK_TOP_K)

    # ── 3. MMR diversity filter ────────────────────────────────────────────────
    diverse = mmr(reranked, top_k=MMR_TOP_K)

    # ── 4. Evidence quality gate ───────────────────────────────────────────────
    gate = check(diverse)
    if not gate["supported"]:
        return {"answer": CANNOT_ANSWER, "sources": [], "supported": False}

    # ── 5. Context expansion + citation-tagged evidence ────────────────────────
    evidence_str, source_map = _build_expanded_evidence(gate["evidence"])
    if not source_map:
        return {"answer": CANNOT_ANSWER, "sources": [], "supported": False}

    # ── 6. LLM generation ─────────────────────────────────────────────────────
    client = _get_client()
    response = client.chat.completions.create(
        model=ACTIVE_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(list(source_map.keys()))},
            {"role": "user",   "content": f"Question: {query}\n\nEvidence:\n{evidence_str}"},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw_answer = response.choices[0].message.content.strip()

    # ── 7. Citation validation ─────────────────────────────────────────────────
    clean_answer, valid_citations, _ = validate_citations(raw_answer, source_map)

    return {
        "answer":    clean_answer,
        "sources":   valid_citations,
        "supported": True,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # 5 answerable
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
    # 3 difficult technical
    "What are the differences between SSC mode 1, 2, and 3 in 5G PDU sessions?",
    "How does the 5G system handle handover between 3GPP and non-3GPP access?",
    "What is the NSSF role in network slice selection and how does it interact with the AMF?",
    # 2 unanswerable / out of domain
    "What is the capital of France?",
    "How do I train a neural network from scratch?",
]


def main():
    from src.generation.grok import ACTIVE_PROVIDER
    print(f"Provider: {ACTIVE_PROVIDER}  |  Model: {ACTIVE_MODEL}\n")

    labels = (
        ["answerable"] * 5 +
        ["difficult technical"] * 3 +
        ["out-of-domain"] * 2
    )

    for query, label in zip(TEST_QUERIES, labels):
        print("=" * 65)
        print(f"[{label}]")
        print(f"Q: {query}")
        print("=" * 65)

        result = answer_question(query)

        print(f"Supported : {result['supported']}")
        print(f"Answer    :\n{result['answer']}")
        if result["sources"]:
            print(f"Sources ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"  {s['id']}  {s['spec']} §{s['section']} p.{s['page']}"
                      f"  — {s.get('title','')}")
        print()


if __name__ == "__main__":
    main()
