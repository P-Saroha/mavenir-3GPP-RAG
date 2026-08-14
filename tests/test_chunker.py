"""
tests/test_chunker.py
---------------------
Tests for src/ingestion/chunker.py.
"""

import json
from pathlib import Path
from src.ingestion.chunker import chunk_sections, build_stats, TARGET_WORDS, MAX_WORDS, MIN_WORDS

CHUNKS_PATH = Path("data/chunks.jsonl")
STATS_PATH = Path("data/chunk_stats.json")


def _make_section(spec="23.501", section="4.2", title="General", text="word " * 100,
                  parent="4", page=10):
    return dict(spec=spec, release="17", version="17.13.0",
                section=section, section_title=title, parent_section=parent,
                page_start=page, page_end=page, text=text)


# ── unit tests ────────────────────────────────────────────────────────────────

def test_chunk_fields_present():
    sections = [_make_section(text="word " * 200)]
    chunks = chunk_sections(sections)
    assert len(chunks) >= 1
    c = chunks[0]
    for attr in ("chunk_id", "spec", "release", "version", "section",
                 "section_title", "parent_section", "page_start", "page_end", "text"):
        assert hasattr(c, attr), f"Missing field: {attr}"


def test_chunk_ids_are_unique():
    sections = [_make_section(text="word " * 100, section=str(i)) for i in range(20)]
    chunks = chunk_sections(sections)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"


def test_small_sections_are_merged():
    # 10 tiny sections of 5 words each → should produce far fewer chunks than sections
    sections = [_make_section(text="word " * 5, section=f"4.{i}") for i in range(10)]
    chunks = chunk_sections(sections)
    assert len(chunks) < len(sections), "Tiny sections should be merged"


def test_large_section_is_split():
    # One section of 1500 words → must produce multiple chunks, all <= MAX_WORDS
    sections = [_make_section(text="word " * 1500)]
    chunks = chunk_sections(sections)
    assert len(chunks) > 1, "Large section should be split"
    for c in chunks:
        assert len(c.text.split()) <= MAX_WORDS, f"Chunk exceeds MAX_WORDS: {len(c.text.split())}"


def test_max_chunk_size_respected():
    # Mix of sizes — no chunk should exceed MAX_WORDS
    import random
    random.seed(42)
    sections = [_make_section(text="word " * random.randint(10, 2000), section=str(i))
                for i in range(50)]
    chunks = chunk_sections(sections)
    for c in chunks:
        assert len(c.text.split()) <= MAX_WORDS, f"Chunk too large: {len(c.text.split())}"


def test_section_metadata_preserved():
    sec = _make_section(spec="23.502", section="4.2.1", title="Registration",
                        parent="4.2", text="word " * 200)
    chunks = chunk_sections([sec])
    c = chunks[0]
    assert c.spec == "23.502"
    assert c.section == "4.2.1"
    assert c.section_title == "Registration"
    assert c.parent_section == "4.2"


def test_build_stats():
    sections = [_make_section(text="word " * 300, section=str(i)) for i in range(5)]
    chunks = chunk_sections(sections)
    stats = build_stats(chunks)
    assert "total_chunks" in stats
    assert "chunks_per_spec" in stats
    assert "avg_words" in stats
    assert "min_words" in stats
    assert "max_words" in stats
    assert stats["total_chunks"] == len(chunks)


# ── integration: check generated files ───────────────────────────────────────

def test_chunks_jsonl_exists():
    assert CHUNKS_PATH.exists(), "Run: python -m src.ingestion.chunker first"


def test_chunks_jsonl_valid():
    lines = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 500
    required = {"chunk_id", "spec", "section", "section_title",
                "page_start", "page_end", "text"}
    for line in lines[:100]:
        obj = json.loads(line)
        assert required.issubset(obj.keys())
        assert obj["text"].strip(), "Empty text in chunk"


def test_stats_json_valid():
    assert STATS_PATH.exists()
    stats = json.loads(STATS_PATH.read_text())
    assert stats["total_chunks"] > 500
    assert set(stats["chunks_per_spec"].keys()) == {"23.501", "23.502", "23.503"}
    assert 200 <= stats["avg_words"] <= 700
