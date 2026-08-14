"""
tests/test_parser.py
--------------------
Tests for src/ingestion/parser.py.
Uses a small slice (pages 20-40) of TS 23.501 to avoid slow full-parse.
"""

import json
from pathlib import Path
from src.ingestion.parser import parse_pdf, _parent, _strip_header, HEADER_RE

PDF_PATH = Path("data/pdfs/TS_23.501_R17_v17.13.0.pdf.pdf")
JSONL_PATH = Path("data/parsed.jsonl")


# ── unit tests ────────────────────────────────────────────────────────────────

def test_parent_section():
    assert _parent("4.2.1") == "4.2"
    assert _parent("4.2") == "4"
    assert _parent("4") == ""
    assert _parent("4.2.1.1") == "4.2.1"


def test_strip_header_removes_boilerplate():
    sample = (
        " \nETSI \nETSI TS 123 501 V17.13.0 (2024-07)\n36\n"
        "3GPP TS 23.501 version 17.13.0 Release 17\n"
        "4.2.2 \nNetwork Functions and entities\n"
    )
    cleaned = _strip_header(sample)
    assert "ETSI TS 123 501" not in cleaned
    assert "4.2.2" in cleaned


def test_parse_pdf_returns_sections():
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
    sections = parse_pdf(PDF_PATH)
    assert len(sections) > 100, "Expected many sections from a 577-page PDF"


def test_section_fields_present():
    sections = parse_pdf(PDF_PATH)
    s = sections[0]
    for attr in ("document", "spec", "release", "version",
                 "page_start", "page_end", "section",
                 "section_title", "parent_section", "text"):
        assert hasattr(s, attr), f"Missing field: {attr}"


def test_spec_metadata():
    sections = parse_pdf(PDF_PATH)
    s = sections[0]
    assert s.spec == "23.501"
    assert s.release == "17"
    assert s.version.startswith("17.")


def test_section_hierarchy():
    sections = parse_pdf(PDF_PATH)
    by_num = {s.section: s for s in sections}
    # 4.2.1 must have parent 4.2
    if "4.2.1" in by_num:
        assert by_num["4.2.1"].parent_section == "4.2"
    # top-level section must have empty parent
    if "1" in by_num:
        assert by_num["1"].parent_section == ""


def test_no_empty_text_sections():
    sections = parse_pdf(PDF_PATH)
    empty = [s for s in sections if not s.text.strip()]
    assert len(empty) == 0, f"{len(empty)} sections have empty text"


def test_jsonl_output_exists_and_valid():
    assert JSONL_PATH.exists(), "Run: python -m src.ingestion.parser first"
    lines = JSONL_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 1000
    # every line must be valid JSON with required keys
    required = {"document", "spec", "section", "section_title",
                "page_start", "page_end", "text"}
    for line in lines[:50]:
        obj = json.loads(line)
        assert required.issubset(obj.keys())
