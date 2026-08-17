"""
src/ingestion/chunker.py
------------------------
Chunking stage: converts parsed 3GPP structural units (parsed.jsonl) into
retrieval-ready chunks (chunks.jsonl).

Strategy
--------
1. Group sections by (spec, section-prefix) so nearby sub-sections stay together.
2. Within each group, accumulate sections until the chunk hits TARGET_WORDS.
3. When a single section already exceeds MAX_WORDS, split it into overlapping
   sub-chunks of TARGET_WORDS words with OVERLAP_WORDS words of overlap.
4. Every chunk retains: spec, release, version, section, section_title,
   parent_section, page_start, page_end, text.

Word-count is used as a proxy for token count (≈ 1.3 words/token),
which avoids a tokenizer dependency while staying accurate enough.

Usage:
    python -m src.ingestion.chunker              # all three PDFs
    python -m src.ingestion.chunker 23.501       # one spec only
"""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ── tuneable constants ────────────────────────────────────────────────────────
TARGET_WORDS = 450      # aim for this many words per chunk
MAX_WORDS = 700         # hard ceiling before we force a split
OVERLAP_WORDS = 50      # words of overlap when splitting large sections
MIN_WORDS = 5           # only merge if truly tiny (was 40 — caused section identity loss)

INPUT_PATH = Path("data/parsed.jsonl")
OUTPUT_PATH = Path("data/chunks.jsonl")
STATS_PATH = Path("data/chunk_stats.json")


@dataclass
class Chunk:
    chunk_id: str
    spec: str
    release: str
    version: str
    section: str
    section_title: str
    parent_section: str
    page_start: int
    page_end: int
    text: str
    source_type: str = "3gpp_official"  # "3gpp_official" or "uploaded"
    document: str = ""  # filename (for uploaded PDFs)


def _chunk_id(spec: str, section: str, page_start: int, text: str, offset: int = 0) -> str:
    """Deterministic chunk ID — same content always gets the same ID.
    Includes word offset so split sub-chunks of the same section get distinct IDs."""
    key = f"{spec}|{section}|{page_start}|{offset}|{text[:80]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _words(text: str) -> int:
    return len(text.split())
    return len(text.split())


def _split_large(section: dict, target: int, overlap: int) -> list[Chunk]:
    """Split one oversized section into overlapping word-window chunks."""
    words = section["text"].split()
    chunks = []
    start = 0
    
    # Determine source_type
    source_type = section.get("source_type", "3gpp_official")
    document = section.get("document", "")
    
    while start < len(words):
        end = min(start + target, len(words))
        chunk_text = " ".join(words[start:end])
        header = f"[{section['spec']} §{section['section']} — {section['section_title']}]"
        txt = f"{header}\n{chunk_text}"
        chunks.append(Chunk(
            chunk_id=_chunk_id(section["spec"], section["section"],
                               section["page_start"], txt, offset=start),
            spec=section["spec"],
            release=section["release"],
            version=section["version"],
            section=section["section"],
            section_title=section["section_title"],
            parent_section=section["parent_section"],
            page_start=section["page_start"],
            page_end=section["page_end"],
            text=txt,
            source_type=source_type,
            document=document,
        ))
        if end == len(words):
            break
        start = end - overlap  # step back by overlap for context continuity
    return chunks


def _make_chunk(sections: list[dict]) -> Chunk:
    """Merge a list of sections into a single chunk."""
    first = sections[0]
    last = sections[-1]
    # Prefix with section header so the reranker can match section-specific queries
    header = f"[{first['spec']} §{first['section']} — {first['section_title']}]"
    body = "\n\n".join(s["text"] for s in sections if s["text"].strip())
    combined_text = f"{header}\n{body}"
    
    # Determine source_type: preserve if in input, otherwise default to 3gpp_official
    source_type = first.get("source_type", "3gpp_official")
    document = first.get("document", "")
    
    return Chunk(
        chunk_id=_chunk_id(first["spec"], first["section"], first["page_start"], combined_text, offset=0),
        spec=first["spec"],
        release=first["release"],
        version=first["version"],
        section=first["section"],
        section_title=first["section_title"],
        parent_section=first["parent_section"],
        page_start=first["page_start"],
        page_end=last["page_end"],
        text=combined_text,
        source_type=source_type,
        document=document,
    )


def chunk_sections(sections: list[dict]) -> list[Chunk]:
    """
    Main chunking logic.

    Pass 1 – filter noise: drop ToC lines (lines that are mostly dots).
    Pass 2 – accumulate sections into TARGET_WORDS buckets; split oversized ones.
    """
    # Pass 1: clean section text (remove table-of-contents dot lines)
    def clean(text: str) -> str:
        lines = [ln for ln in text.splitlines()
                 if ln.count(".") < len(ln) * 0.4]  # skip lines >40% dots
        return "\n".join(lines).strip()

    cleaned = []
    for s in sections:
        s = dict(s)
        s["text"] = clean(s["text"])
        if s["text"]:
            cleaned.append(s)

    chunks: list[Chunk] = []
    bucket: list[dict] = []
    bucket_words = 0

    for sec in cleaned:
        sec_words = _words(sec["text"])

        # Oversized single section — split independently
        if sec_words > MAX_WORDS:
            # Flush current bucket first
            if bucket:
                chunks.append(_make_chunk(bucket))
                bucket, bucket_words = [], 0
            chunks.extend(_split_large(sec, TARGET_WORDS, OVERLAP_WORDS))
            continue

        # Adding this section would exceed the target — flush first
        if bucket and (bucket_words + sec_words) > TARGET_WORDS:
            chunks.append(_make_chunk(bucket))
            bucket, bucket_words = [], 0

        bucket.append(sec)
        bucket_words += sec_words

        # Flush when we've hit the target
        if bucket_words >= MIN_WORDS and bucket_words >= TARGET_WORDS:
            chunks.append(_make_chunk(bucket))
            bucket, bucket_words = [], 0

    # Flush remainder
    if bucket:
        # If tiny, try to merge with previous chunk instead of emitting alone
        if bucket_words < MIN_WORDS and chunks:
            prev = chunks[-1]
            merged_text = prev.text + "\n\n" + "\n\n".join(
                s["text"] for s in bucket if s["text"].strip()
            )
            chunks[-1] = Chunk(
                chunk_id=_chunk_id(prev.spec, prev.section, prev.page_start, merged_text, offset=0),
                spec=prev.spec, release=prev.release, version=prev.version,
                section=prev.section, section_title=prev.section_title,
                parent_section=prev.parent_section,
                page_start=prev.page_start,
                page_end=bucket[-1]["page_end"],
                text=merged_text.strip(),
                source_type=prev.source_type,
                document=prev.document,
            )
        else:
            chunks.append(_make_chunk(bucket))

    return chunks


def build_stats(chunks: list[Chunk]) -> dict:
    sizes = [_words(c.text) for c in chunks]
    per_spec: dict[str, int] = {}
    for c in chunks:
        per_spec[c.spec] = per_spec.get(c.spec, 0) + 1
    return {
        "total_chunks": len(chunks),
        "chunks_per_spec": per_spec,
        "avg_words": round(sum(sizes) / len(sizes)) if sizes else 0,
        "min_words": min(sizes) if sizes else 0,
        "max_words": max(sizes) if sizes else 0,
    }


def main(filter_spec: str | None = None):
    sections = [json.loads(l) for l in INPUT_PATH.read_text(encoding="utf-8").splitlines()]

    if filter_spec:
        sections = [s for s in sections if s["spec"] == filter_spec]
        print(f"Filtered to spec {filter_spec}: {len(sections)} sections")

    print(f"Chunking {len(sections)} sections ...")
    chunks = chunk_sections(sections)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    stats = build_stats(chunks)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Total chunks : {stats['total_chunks']}")
    print(f"Per spec     : {stats['chunks_per_spec']}")
    print(f"Avg words    : {stats['avg_words']}")
    print(f"Min/Max words: {stats['min_words']} / {stats['max_words']}")
    print(f"Output       : {OUTPUT_PATH}")
    print(f"Stats        : {STATS_PATH}")


if __name__ == "__main__":
    spec_filter = sys.argv[1] if len(sys.argv) > 1 else None
    main(spec_filter)
