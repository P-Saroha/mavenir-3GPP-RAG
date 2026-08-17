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

from src.generation.grok import _get_client, GROK_MODEL, CANNOT_ANSWER
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
            "title":    c.get("section_title", ""),
            "source_type": c.get("source_type", "3gpp_official"),
            "document": c.get("document", ""),
        }
        header = (f"[{sid}] {c.get('spec','')} "
                  f"§{c.get('section','')} p.{c.get('page', c.get('page_start',''))}"
                  f" — {c.get('section_title','')}")
        parts.append(f"{header}\n{c.get('text','').strip()}")

    return "\n\n".join(parts), source_map


# ── citation-aware system prompt ──────────────────────────────────────────────

def _system_prompt(source_ids: list[str]) -> str:
    ids_str = ", ".join(f"[{sid}]" for sid in source_ids)
    return f"""You are a 3GPP Release 17 5G Core standards assistant.

Available sources: {ids_str}

Rules:
- Use ONLY the evidence provided. Do not use outside knowledge.
- Every factual claim MUST be followed by a citation: [S1], [S2], etc.
- Use ONLY the source IDs listed above. Do NOT invent new ones.
- Do not invent technical facts, section numbers, or procedures.
- If evidence is insufficient, respond exactly:
  "{CANNOT_ANSWER}"
- Be concise and technical."""


# ── citation parsing and validation ──────────────────────────────────────────

def parse_citations(text: str) -> list[str]:
    """Return all unique [Sx] IDs found in text, in order of first appearance."""
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
    cited = parse_citations(answer_text)
    valid: list[dict] = []
    invalid: list[str] = []

    for sid in cited:
        if sid in source_map:
            valid.append({"id": f"[{sid}]", **source_map[sid]})
        else:
            invalid.append(sid)

    # replace unknown IDs in the answer text
    clean = answer_text
    for sid in invalid:
        clean = clean.replace(f"[{sid}]", "[INVALID]")

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
        model=GROK_MODEL,
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
