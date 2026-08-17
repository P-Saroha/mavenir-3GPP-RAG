"""
src/generation/citations.py
-----------------------------
Deterministic citation system for grounded LLM answers.

Flow:
  1. Assign [S1]..[SN] IDs to retrieved evidence chunks before calling LLM.
  2. Instruct the LLM to cite ONLY those IDs.
  3. After generation, parse every [Sx] tag in the answer.
  4. Verify each cited ID exists in the source map.
  5. Remove / flag any hallucinated IDs.
  6. Return structured citations mapped to real metadata.

Usage:
    python -m src.generation.citations
"""

from __future__ import annotations

import re

from src.generation.grok import _get_client, GROQ_MODEL, CANNOT_ANSWER
from src.retrieval.reranker import retrieve_and_rerank
from src.retrieval.quality_gate import check

# Matches [S1], [S2], ... [S99]
CITATION_RE = re.compile(r"\[S(\d+)\]")

# ── source-tagged evidence ────────────────────────────────────────────────────

def build_sourced_evidence(chunks: list[dict]) -> tuple[str, dict[str, dict]]:
    """
    Assign [S1]..[SN] IDs to chunks and return:
      - formatted evidence string with IDs
      - source_map: {"S1": {spec, release, section, page, text, source_type, document}, ...}
    """
    source_map: dict[str, dict] = {}
    parts: list[str] = []

    for i, c in enumerate(chunks, start=1):
        sid = f"S{i}"
        source_map[sid] = {
            "id":       f"[{sid}]",
            "spec":     c.get("spec", ""),
            "release":  c.get("release", "17"),
            "section":  c.get("section", ""),
            "page":     c.get("page", c.get("page_start", "")),
            "page_end": c.get("page_end", ""),
            "title":    c.get("section_title", ""),
            "source_type": c.get("source_type", "3gpp_official"),
            "document": c.get("document", ""),
            "text":     c.get("text", ""),  # Include full text for verification
            "chunk_id": c.get("chunk_id", ""),
        }
        
        # Build detailed header with all reference info
        if c.get("source_type") == "uploaded":
            header = (f"[{sid}] {c.get('document','')} "
                      f"§{c.get('section','')} pp.{c.get('page_start','')}-{c.get('page_end','')}")
        else:
            page_range = c.get("page", c.get("page_start", ""))
            if c.get("page_end") and c.get("page_end") != c.get("page"):
                page_range = f"{c.get('page_start','')}-{c.get('page_end','')}"
            header = (f"[{sid}] TS {c.get('spec','')} Release {c.get('release','17')} "
                      f"§{c.get('section','')} pp.{page_range} — {c.get('section_title','')}")
        
        parts.append(f"{header}\n{c.get('text','').strip()}")

    return "\n\n".join(parts), source_map


# ── citation-aware system prompt ──────────────────────────────────────────────

def _system_prompt(source_ids: list[str]) -> str:
    ids_str = ", ".join(f"[{sid}]" for sid in source_ids)
    return f"""You are a 3GPP Release 17 5G Core standards assistant.

AVAILABLE SOURCES (use ONLY these):
{ids_str}

MANDATORY CITATION RULES:
1. EVERY sentence with factual information MUST end with a citation: [S1], [S2], [S3], etc.
2. ONLY use the source IDs listed above.
3. DO NOT cite non-existent IDs or use section numbers like [4.26.3-1].
4. DO NOT use outside knowledge. Only paraphrase the provided evidence.
5. If evidence is insufficient, respond ONLY with: "{CANNOT_ANSWER}"

FORMAT EXAMPLE:
The AMF manages registration [S1]. It coordinates with the SMF [S2]. See TS 23.502 section 4.26.3 [S3].

Now answer the user's question. Remember: CITE EVERY SENTENCE."""


# ── citation parsing and validation ──────────────────────────────────────────

def parse_citations(text: str) -> list[str]:
    """Return all unique [Sx] IDs found in text, in order of first appearance."""
    # Handle various unicode issues (zero-width spaces, etc.)
    # Normalize the text
    text = text.replace('\u200b', '')  # Remove zero-width space
    text = text.replace('\u200c', '')  # Remove zero-width non-joiner
    text = text.replace('\u200d', '')  # Remove zero-width joiner
    text = text.replace('\u2060', '')  # Remove word joiner
    
    seen = set()
    result = []
    for m in CITATION_RE.finditer(text):
        sid = f"S{m.group(1)}"
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def validate_citations(
    answer_text: str,
    source_map: dict[str, dict],
) -> tuple[str, list[dict], list[str]]:
    """
    Validate cited IDs against the source map.

    Returns:
        clean_answer  — answer with unknown IDs replaced by [INVALID]
        valid_citations — list of {id, spec, release, section, page, title}
        invalid_ids   — list of IDs cited by LLM that don't exist
    """
    # Normalize unicode issues
    answer_text = answer_text.replace('\u200b', '')  # Remove zero-width space
    answer_text = answer_text.replace('\u200c', '')  # Remove zero-width non-joiner
    answer_text = answer_text.replace('\u200d', '')  # Remove zero-width joiner
    answer_text = answer_text.replace('\u2060', '')  # Remove word joiner
    
    cited = parse_citations(answer_text)
    valid: list[dict] = []
    invalid: list[str] = []

    print(f"[DEBUG] Parsed citations from answer: {cited}")
    print(f"[DEBUG] Available source IDs: {list(source_map.keys())}")

    for sid in cited:
        if sid in source_map:
            valid.append({"id": f"[{sid}]", **source_map[sid]})
            print(f"[DEBUG] Valid citation found: {sid}")
        else:
            invalid.append(sid)
            print(f"[DEBUG] Invalid/hallucinated citation: {sid}")

    # replace unknown IDs in the answer text
    clean = answer_text
    for sid in invalid:
        clean = clean.replace(f"[{sid}]", "[INVALID]")

    print(f"[DEBUG] Valid citations count: {len(valid)}, Invalid: {len(invalid)}")
    return clean, valid, invalid


# ── main pipeline ─────────────────────────────────────────────────────────────

def answer_with_citations(query: str, top_k: int = 5) -> dict:
    """
    Full pipeline with deterministic citations.

    Returns:
    {
        "query":       str,
        "supported":   bool,
        "answer":      str,          # clean answer with [Sx] tags
        "citations":   list[dict],   # verified {id, spec, release, section, page, title}
        "invalid_ids": list[str],    # any IDs the LLM hallucinated
        "evidence":    list[dict],   # raw retrieved chunks
    }
    """
    candidates = retrieve_and_rerank(query, top_k=top_k)
    gate = check(candidates)

    if not gate["supported"]:
        return {
            "query":       query,
            "supported":   False,
            "answer":      CANNOT_ANSWER,
            "citations":   [],
            "invalid_ids": [],
            "evidence":    [],
        }

    evidence_str, source_map = build_sourced_evidence(gate["evidence"])
    source_ids = list(source_map.keys())

    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(source_ids)},
            {"role": "user",   "content": f"Question: {query}\n\nEvidence:\n{evidence_str}"},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw_answer = response.choices[0].message.content.strip()

    clean_answer, valid_citations, invalid_ids = validate_citations(raw_answer, source_map)

    return {
        "query":       query,
        "supported":   True,
        "answer":      clean_answer,
        "citations":   valid_citations,
        "invalid_ids": invalid_ids,
        "evidence":    gate["evidence"],
    }


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
        print("=" * 65)
        print(f"Q: {query}")
        print("=" * 65)
        result = answer_with_citations(query)
        print(f"Supported   : {result['supported']}")
        print(f"Invalid IDs : {result['invalid_ids'] or 'none'}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  {c['id']}  {c['spec']} §{c['section']} p.{c['page']}  — {c['title']}")
        print()


if __name__ == "__main__":
    main()
