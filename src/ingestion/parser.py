"""
src/ingestion/parser.py
-----------------------
Structure-aware parser for 3GPP Release 17 PDFs.

Strategy:
  - Read each page with PyMuPDF.
  - Strip the recurring ETSI page header (boilerplate at the top of every page).
  - Detect 3GPP section headings by their numbering pattern (4  /  4.2  /  4.2.1 …).
  - Accumulate body text under the current heading; emit a record when a new
    heading starts OR when a page boundary forces a split.
  - Preserve page_start / page_end, section number, title, and parent_section.

Output: data/parsed.jsonl  (one JSON object per structural unit)

Usage:
    python -m src.ingestion.parser
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# ── paths ─────────────────────────────────────────────────────────────────────
PDF_DIR = Path("data/pdfs")
OUTPUT_PATH = Path("data/parsed.jsonl")

# ── patterns ──────────────────────────────────────────────────────────────────
# Matches lines like:  "4"  "4.2"  "4.2.1"  "4.2.1.1"  "4.2.1.1.1"
# The section number must be at the start of a line (after stripping).
HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+){0,4})\s{1,10}(\S[^\n]{0,120})$",
    re.MULTILINE,
)

# ETSI page header that appears on every page — strip it before processing.
# Pattern: optional whitespace, "ETSI", version line, page-number line, spec line
HEADER_RE = re.compile(
    r"^\s*ETSI\s*\nETSI TS \d{3} \d{3} V[\d.]+ \(\d{4}-\d{2}\)\s*\n\d+\s*\n3GPP TS [\d.]+ version [\d.]+ Release \d+\s*\n",
    re.MULTILINE,
)

# ── metadata extracted from the cover page ────────────────────────────────────
# e.g. "ETSI TS 123 501 V17.13.0 (2024-07)"
COVER_RE = re.compile(
    r"ETSI TS (\d{3} \d{3})\s+V([\d.]+)\s+\(\d{4}-\d{2}\).*?Release (\d+)",
    re.DOTALL,
)


@dataclass
class Section:
    document: str
    spec: str          # e.g. "23.501"
    release: str       # e.g. "17"
    version: str       # e.g. "17.13.0"
    page_start: int
    page_end: int
    section: str       # e.g. "4.2.1"
    section_title: str
    parent_section: str
    text: str = field(default="", repr=False)


def _parse_spec_meta(cover_text: str) -> tuple[str, str, str]:
    """Extract (spec_number, version, release) from the cover page."""
    m = COVER_RE.search(cover_text)
    if not m:
        return ("unknown", "unknown", "unknown")
    raw_spec = m.group(1).replace(" ", "")   # "123501"
    # "123501" → drop leading "1", then insert dot after first two digits → "23.501"
    digits = raw_spec.lstrip("1")             # "23501"
    spec = digits[:2] + "." + digits[2:]      # "23.501"
    version = m.group(2)
    release = m.group(3)
    return spec, version, release


def _strip_header(text: str) -> str:
    """Remove the ETSI boilerplate header that appears on every page."""
    return HEADER_RE.sub("", text).strip()


def _parent(section: str) -> str:
    """Return the immediate parent section number, or '' for top-level."""
    parts = section.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else ""


def parse_pdf(pdf_path: Path) -> list[Section]:
    doc = fitz.open(pdf_path)
    filename = pdf_path.name

    # extract metadata from cover page (page 0)
    cover_text = doc[0].get_text()
    spec, version, release = _parse_spec_meta(cover_text)

    sections: list[Section] = []
    current: Section | None = None

    for page_idx in range(len(doc)):
        raw = doc[page_idx].get_text()
        text = _strip_header(raw)

        # Split the page text on every heading occurrence
        # so that a page containing multiple headings emits multiple sections.
        pos = 0
        for m in HEADING_RE.finditer(text):
            section_num = m.group(1)
            section_title = m.group(2).strip()
            match_start = m.start()

            # Body text before this heading belongs to the current section
            preceding = text[pos:match_start].strip()
            if preceding and current is not None:
                current.text += ("\n" + preceding) if current.text else preceding
                current.page_end = page_idx + 1  # 1-indexed

            # Flush the current section if we hit a new one
            if current is not None:
                # Only save sections with some content
                if current.text.strip():
                    sections.append(current)

            current = Section(
                document=filename,
                spec=spec,
                release=release,
                version=version,
                page_start=page_idx + 1,
                page_end=page_idx + 1,
                section=section_num,
                section_title=section_title,
                parent_section=_parent(section_num),
                text="",
            )
            pos = m.end()

        # Remaining text on this page goes into the current section
        remaining = text[pos:].strip()
        if remaining:
            if current is not None:
                current.text += ("\n" + remaining) if current.text else remaining
                current.page_end = page_idx + 1

    # Flush the last section
    if current is not None and current.text.strip():
        sections.append(current)

    doc.close()
    return sections


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for pdf_path in pdf_files:
            print(f"Parsing {pdf_path.name} ...")
            sections = parse_pdf(pdf_path)
            for sec in sections:
                out.write(json.dumps(asdict(sec), ensure_ascii=False) + "\n")
            print(f"  → {len(sections)} sections extracted")
            total += len(sections)

    print(f"\nTotal sections: {total}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
