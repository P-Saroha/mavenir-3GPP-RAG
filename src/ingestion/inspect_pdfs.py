"""
src/ingestion/inspect_pdfs.py
------------------------------
Inspect the three 3GPP PDFs and save a structural report to data/pdf_report.json.

Usage:
    python -m src.ingestion.inspect_pdfs
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

# ── paths ─────────────────────────────────────────────────────────────────────
PDF_DIR = Path("data/pdfs")
REPORT_PATH = Path("data/pdf_report.json")

# A 3GPP section heading looks like:  "4.2.3  Some title"
HEADING_RE = re.compile(r"^\d+(\.\d+){0,4}\s{2,}.+", re.MULTILINE)

# A page is considered "sparse" if it has fewer than this many characters
SPARSE_THRESHOLD = 100


def inspect_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    # ── first and last page text ───────────────────────────────────────────────
    first_text = doc[0].get_text().strip()[:500]
    last_text = doc[-1].get_text().strip()[:500]

    # ── scan all pages ─────────────────────────────────────────────────────────
    sparse_pages = 0
    all_headings = []
    page_numbers_found = []

    for page in doc:
        text = page.get_text()

        if len(text.strip()) < SPARSE_THRESHOLD:
            sparse_pages += 1

        # collect heading examples (up to 20 total)
        if len(all_headings) < 20:
            for m in HEADING_RE.finditer(text):
                if len(all_headings) < 20:
                    all_headings.append(m.group().strip())

        # check if a page number is visible in the footer/header area
        # PyMuPDF page numbers are 0-indexed; look for a digit-only line
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                page_numbers_found.append(int(stripped))
                break

    doc.close()

    return {
        "filename": pdf_path.name,
        "page_count": page_count,
        "first_page_text": first_text,
        "last_page_text": last_text,
        "heading_examples": all_headings,
        "page_numbers_detectable": len(page_numbers_found) > page_count * 0.5,
        "text_extraction_quality": "good" if sparse_pages < page_count * 0.1 else "poor",
        "sparse_page_count": sparse_pages,
    }


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        return

    report = []
    for pdf_path in pdf_files:
        print(f"Inspecting {pdf_path.name} ...")
        result = inspect_pdf(pdf_path)
        report.append(result)
        print(f"  pages={result['page_count']}  sparse={result['sparse_page_count']}  "
              f"quality={result['text_extraction_quality']}  "
              f"page_numbers={result['page_numbers_detectable']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
