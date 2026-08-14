"""
src/ingestion/validate_chunks.py
---------------------------------
Validates data/chunks.jsonl and prints a compact quality report.
Does NOT modify any data.

Usage:
    python -m src.ingestion.validate_chunks
"""

import json
import random
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")

# Thresholds
MIN_WORDS = 10
MAX_WORDS = 700


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate(chunks: list[dict]) -> dict:
    issues: dict[str, list] = {
        "empty_text":          [],
        "duplicate_ids":       [],
        "duplicate_text":      [],
        "missing_section":     [],
        "missing_pages":       [],
        "too_small":           [],
        "too_large":           [],
        "missing_spec_meta":   [],
    }

    seen_ids: dict[str, int] = {}
    seen_text: dict[str, int] = {}

    for i, c in enumerate(chunks):
        cid = c.get("chunk_id", f"<row {i}>")

        # empty text
        if not c.get("text", "").strip():
            issues["empty_text"].append(cid)

        # duplicate IDs
        if cid in seen_ids:
            issues["duplicate_ids"].append((cid, seen_ids[cid], i))
        else:
            seen_ids[cid] = i

        # duplicate text (use first 120 chars as fingerprint)
        fingerprint = c.get("text", "")[:120].strip()
        if fingerprint in seen_text:
            issues["duplicate_text"].append((cid, seen_text[fingerprint]))
        else:
            seen_text[fingerprint] = i

        # missing section number
        if not c.get("section", "").strip():
            issues["missing_section"].append(cid)

        # missing page info
        if not c.get("page_start") or not c.get("page_end"):
            issues["missing_pages"].append(cid)

        # chunk size
        words = len(c.get("text", "").split())
        if words < MIN_WORDS:
            issues["too_small"].append((cid, words))
        if words > MAX_WORDS:
            issues["too_large"].append((cid, words))

        # spec metadata
        for field in ("spec", "release", "version"):
            if not c.get(field, "").strip():
                issues["missing_spec_meta"].append((cid, field))
                break

    return issues


def print_report(chunks: list[dict], issues: dict):
    total = len(chunks)
    print("=" * 55)
    print(f"  CHUNK QUALITY REPORT  —  {total} chunks total")
    print("=" * 55)

    checks = [
        ("empty_text",        "Empty text"),
        ("duplicate_ids",     "Duplicate IDs"),
        ("duplicate_text",    "Duplicate text (fingerprint)"),
        ("missing_section",   "Missing section number"),
        ("missing_pages",     "Missing page_start/page_end"),
        ("too_small",         f"Too small (< {MIN_WORDS} words)"),
        ("too_large",         f"Too large (> {MAX_WORDS} words)"),
        ("missing_spec_meta", "Missing spec/release/version"),
    ]

    all_ok = True
    for key, label in checks:
        count = len(issues[key])
        status = "OK" if count == 0 else "WARN"
        flag = "  " if count == 0 else "!"
        print(f"  {flag} {label:<36} {status:4}  ({count})")
        if count:
            all_ok = False

    print("-" * 55)
    if all_ok:
        print("  All checks passed.")
    else:
        print("  Some issues found — see counts above.")
    print("=" * 55)

    # size distribution
    words_list = [len(c.get("text", "").split()) for c in chunks]
    avg = round(sum(words_list) / len(words_list)) if words_list else 0
    print(f"\n  Word-count distribution:")
    print(f"    min={min(words_list)}  avg={avg}  max={max(words_list)}")

    per_spec: dict[str, int] = {}
    for c in chunks:
        k = c.get("spec", "unknown")
        per_spec[k] = per_spec.get(k, 0) + 1
    print(f"\n  Chunks per spec:")
    for spec, n in sorted(per_spec.items()):
        print(f"    {spec}: {n}")


def print_samples(chunks: list[dict], n: int = 5):
    print(f"\n{'=' * 55}")
    print(f"  {n} RANDOM SAMPLE CHUNKS")
    print("=" * 55)
    for c in random.sample(chunks, min(n, len(chunks))):
        print(f"\n  chunk_id  : {c['chunk_id']}")
        print(f"  spec      : {c.get('spec')}  v{c.get('version')}  R{c.get('release')}")
        print(f"  section   : {c.get('section')}  —  {c.get('section_title')}")
        print(f"  parent    : {c.get('parent_section') or '(top-level)'}")
        print(f"  pages     : {c.get('page_start')} – {c.get('page_end')}")
        words = len(c.get("text", "").split())
        preview = " ".join(c.get("text", "").split()[:30])
        print(f"  words     : {words}")
        print(f"  text      : {preview} ...")


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run: python -m src.ingestion.chunker")
        return

    chunks = load_chunks(CHUNKS_PATH)
    issues = validate(chunks)
    print_report(chunks, issues)
    print_samples(chunks)


if __name__ == "__main__":
    main()
