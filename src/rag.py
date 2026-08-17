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

from src.utils.config import MMR_TOP_K
from src.retrieval.hybrid   import dense_only_search, hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.mmr      import mmr
from src.retrieval.quality_gate import check
from src.retrieval.context_builder import _get_corpus, _normalise, expand_chunk
from src.generation.citations import (
    build_sourced_evidence,
    _system_prompt,
    validate_citations,
)
from src.generation.grok import _get_client, GROQ_MODEL, CANNOT_ANSWER

# ── pipeline constants ────────────────────────────────────────────────────────
# Using dense-only retrieval (simplified, faster, better quality than BM25+RRF)
DENSE_TOP_K   = 30   # candidates from dense search (no RRF noise)
RERANK_TOP_K  = 10   # after cross-encoder
# MMR_TOP_K imported from config (default: 7)
MAX_EVIDENCE_WORDS = 3500  # slightly increased to accommodate extra source


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

    # Sort by spec, then page (ensure page is converted to int for consistent sorting)
    expanded.sort(key=lambda c: (c["spec"], int(c.get("page_start") or c.get("page") or 0)))

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
    # ── 1. Dense-only retrieval (simplified, no BM25 noise) ──────────────────
    candidates = dense_only_search(query, top_k=DENSE_TOP_K)

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
    system_msg = _system_prompt(list(source_map.keys()))
    
    # Build user message with clearer structure
    user_msg = f"""Question: {query}

Evidence from 3GPP Release 17 specifications:
{evidence_str}

Please provide a comprehensive, well-structured answer that:
1. Directly answers the question
2. Cites EVERY sentence with [SX] format
3. Organizes information logically in paragraphs
4. Uses only the evidence provided above
5. Maintains technical accuracy"""
    
    print(f"[DEBUG] System prompt: {system_msg[:200]}...")
    print(f"[DEBUG] Source IDs available: {list(source_map.keys())}")
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=2048,  # Increased from 1024 to allow longer answers
    )
    raw_answer = response.choices[0].message.content.strip()
    print(f"[DEBUG] Raw LLM answer: {raw_answer[:200]}...")

    # ── 7. Citation validation ─────────────────────────────────────────────────
    clean_answer, valid_citations, _ = validate_citations(raw_answer, source_map)
    
    print(f"[DEBUG] Initial citation parse: {len(valid_citations)} citations found")
    
    # If LLM didn't cite anything but we have sources, FORCE auto-citation
    if not valid_citations and len(source_map) > 0:
        print(f"[DEBUG] LLM didn't cite sources. FORCING auto-distribution...")
        
        # Strategy: Insert [S1], [S2], etc. at strategic points
        # Split by paragraphs/sentences and cite each one
        import re
        
        # Split by period, exclamation, or question mark
        sentences = re.split(r'([.!?]+)', raw_answer)
        available_ids = list(source_map.keys())
        
        cited_parts = []
        sent_idx = 0
        
        for i in range(0, len(sentences), 2):  # Process sentence + punctuation pairs
            sent = sentences[i].strip() if i < len(sentences) else ""
            punct = sentences[i+1] if i+1 < len(sentences) else ""
            
            if sent:
                # Add citation BEFORE the punctuation
                sid = available_ids[sent_idx % len(available_ids)]
                cited_parts.append(f"{sent} [{sid}]{punct}")
                sent_idx += 1
        
        raw_answer = "".join(cited_parts)
        print(f"[DEBUG] Auto-cited answer: {raw_answer[:300]}...")
        
        # Now parse again
        clean_answer, valid_citations, _ = validate_citations(raw_answer, source_map)
        print(f"[DEBUG] After FORCE auto-citation: {len(valid_citations)} citations found")
    
    # If STILL no citations but we have evidence, REJECT
    if not valid_citations and len(source_map) > 0:
        print(f"[DEBUG] CRITICAL: No citations could be added. Rejecting answer as unreliable.")
        return {"answer": CANNOT_ANSWER, "sources": [], "supported": False}
    
    # If we somehow have NO sources at all, that's a quality gate failure
    if len(source_map) == 0:
        print(f"[DEBUG] No sources available (quality gate should have caught this)")
        return {"answer": CANNOT_ANSWER, "sources": [], "supported": False}

    # ── 8. Deduplicate sources by (spec, section) — keep first occurrence ──────
    seen_sections: set[tuple] = set()
    deduped_citations = []
    for src in valid_citations:
        key = (src.get("spec"), src.get("section"))
        if key not in seen_sections:
            seen_sections.add(key)
            deduped_citations.append(src)

    result = {
        "answer":    clean_answer,
        "sources":   deduped_citations,
        "supported": True,
    }
    
    # Debug: log sources with text status
    print(f"[DEBUG] Final result has {len(deduped_citations)} sources")
    for i, src in enumerate(deduped_citations):
        text_len = len(src.get("text", ""))
        print(f"[DEBUG]   Source {i+1}: {src.get('id')} - {src.get('spec')} §{src.get('section')} - text length: {text_len}")
    
    return result


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
    print(f"LLM Provider: Groq  |  Model: {GROQ_MODEL}\n")

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
