"""
src/retrieval/context_builder.py
---------------------------------
Parent-context expansion for retrieved chunks.

For each reranked chunk, optionally pull in adjacent chunks from the same
section (or direct parent/child sections) within the same spec, to give the
LLM enough surrounding context without fetching unrelated material.

Rules:
  - Only expand within the same spec.
  - A neighbour is eligible if its section is the same, a parent, or a
    direct child of the selected chunk's section.
  - Pages must be adjacent or overlapping (gap <= PAGE_GAP).
  - Deduplicate by chunk_id; preserve page order.
  - Total evidence is capped at MAX_EVIDENCE_WORDS to stay within LLM context.

Usage:
    python -m src.retrieval.context_builder
"""

from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.reranker import retrieve_and_rerank

CHUNKS_PATH = Path("data/chunks.jsonl")
PAGE_GAP = 1            # max page gap to consider a chunk adjacent
MAX_EVIDENCE_WORDS = 3000   # hard cap to avoid overflowing LLM context window
EXPAND_WINDOW = 1       # how many adjacent chunks each side to consider


# ── section helpers ───────────────────────────────────────────────────────────

def _is_related_section(base: str, candidate: str) -> bool:
    """
    True if candidate is the same section, a parent, or a direct child of base.

    Examples:
        base=4.2.1  candidate=4.2.1   → True  (same)
        base=4.2.1  candidate=4.2     → True  (parent)
        base=4.2.1  candidate=4.2.1.1 → True  (direct child, one level deeper)
        base=4.2.1  candidate=4.3     → False (sibling)
        base=4.2.1  candidate=4.2.1.1.2 → False (too deep)
    """
    if base == candidate:
        return True
    # candidate is a parent of base
    if base.startswith(candidate + "."):
        return True
    # candidate is a direct child of base (exactly one level deeper)
    if candidate.startswith(base + "."):
        remainder = candidate[len(base) + 1:]
        if "." not in remainder:
            return True
    return False


# ── corpus index (loaded once) ────────────────────────────────────────────────

_corpus: list[dict] | None = None


def _get_corpus() -> list[dict]:
    global _corpus
    if _corpus is None:
        _corpus = [
            json.loads(l)
            for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        ]
    return _corpus


# ── expansion ─────────────────────────────────────────────────────────────────

def expand_chunk(chunk: dict, corpus: list[dict], window: int = EXPAND_WINDOW) -> list[dict]:
    """
    Find the chunk's position in the corpus and return it with up to `window`
    eligible neighbours on each side.
    """
    cid = chunk["chunk_id"]
    spec = chunk["spec"]
    section = chunk["section"]
    page_start = chunk.get("page_start") or chunk.get("page", 0)
    page_end = chunk.get("page_end") or chunk.get("page", 0)

    # find the chunk's index in the ordered corpus
    idx = next((i for i, c in enumerate(corpus) if c["chunk_id"] == cid), None)
    if idx is None:
        return [chunk]

    neighbours = [chunk]

    # look left and right
    for direction in (-1, +1):
        for step in range(1, window + 1):
            pos = idx + direction * step
            if pos < 0 or pos >= len(corpus):
                break
            candidate = corpus[pos]

            # must be same spec
            if candidate["spec"] != spec:
                break

            # must be a related section
            if not _is_related_section(section, candidate["section"]):
                break

            # pages must be adjacent or overlapping
            gap = (candidate["page_start"] - page_end
                   if direction == +1
                   else page_start - candidate["page_end"])
            if gap > PAGE_GAP:
                break

            neighbours.append(candidate)

    return neighbours


def _normalise(chunk: dict, corpus_by_id: dict | None = None) -> dict:
    """Ensure chunk always has page_start / page_end and full metadata."""
    c = dict(chunk)
    if "page_start" not in c:
        c["page_start"] = c.get("page", 0)
    if "page_end" not in c:
        c["page_end"] = c.get("page", c["page_start"])
    # fill missing fields from corpus if available
    if corpus_by_id and c.get("chunk_id") in corpus_by_id:
        full = corpus_by_id[c["chunk_id"]]
        for field in ("section_title", "parent_section", "release", "version",
                      "page_start", "page_end"):
            if field not in c or not c[field]:
                c[field] = full.get(field, "")
    # fallback defaults
    c.setdefault("section_title", c.get("section", ""))
    c.setdefault("parent_section", "")
    return c


def build_evidence(reranked: list[dict], window: int = EXPAND_WINDOW) -> str:
    """
    Expand each reranked chunk, deduplicate, sort by page, and concatenate
    into a single evidence string for the LLM.

    Returns a formatted string with section headers.
    """
    corpus = _get_corpus()
    corpus_by_id = {c["chunk_id"]: c for c in corpus}

    seen_ids: set[str] = set()
    all_chunks: list[dict] = []

    for chunk in reranked:
        for c in expand_chunk(_normalise(chunk, corpus_by_id), corpus, window):
            c = _normalise(c, corpus_by_id)
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                all_chunks.append(c)

    # sort by spec, then page
    all_chunks.sort(key=lambda c: (c["spec"], c.get("page_start") or c.get("page", 0)))

    # build evidence text with word cap
    parts: list[str] = []
    total_words = 0

    for c in all_chunks:
        header = (f"[{c['spec']} §{c['section']} — {c['section_title']} "
                  f"(p.{c['page_start']})]")
        body = c["text"].strip()
        chunk_words = len(body.split())

        if total_words + chunk_words > MAX_EVIDENCE_WORDS:
            # include a truncated version rather than nothing
            remaining = MAX_EVIDENCE_WORDS - total_words
            if remaining > 50:
                body = " ".join(body.split()[:remaining]) + " [...]"
                parts.append(f"{header}\n{body}")
            break

        parts.append(f"{header}\n{body}")
        total_words += chunk_words

    return "\n\n".join(parts)


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
        print(f"\n{'=' * 65}")
        print(f"QUERY: {query}")
        print("=" * 65)

        reranked = retrieve_and_rerank(query, top_k=5)
        evidence = build_evidence(reranked)

        words = len(evidence.split())
        chunks_in = len(reranked)
        # count headers to get expanded chunk count
        expanded = evidence.count("\n[")  + (1 if evidence.startswith("[") else 0)
        print(f"  Input chunks : {chunks_in}")
        print(f"  After expand : {expanded} passages")
        print(f"  Total words  : {words}")
        print(f"\n--- Evidence preview (first 300 words) ---")
        preview = " ".join(evidence.split()[:300])
        print(preview)
        print()


if __name__ == "__main__":
    main()
