"""
tests/test_citations.py
------------------------
Tests for src/generation/citations.py
"""

from unittest.mock import patch, MagicMock
import pytest
from src.generation.citations import (
    build_sourced_evidence,
    parse_citations,
    validate_citations,
    answer_with_citations,
    CITATION_RE,
)


def _chunk(spec="23.501", section="4.2", page=36, title="General", text="The AMF handles mobility."):
    return {
        "chunk_id": f"{spec}-{section}",
        "spec": spec, "release": "17", "version": "17.13.0",
        "section": section, "section_title": title,
        "parent_section": "4",
        "page": page, "page_end": page,
        "rerank_score": 5.0, "text": text,
    }


def _mock_llm(text: str):
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


# ── build_sourced_evidence ────────────────────────────────────────────────────

def test_source_ids_assigned():
    chunks = [_chunk(section=f"4.{i}") for i in range(3)]
    evidence_str, source_map = build_sourced_evidence(chunks)
    assert "S1" in source_map
    assert "S2" in source_map
    assert "S3" in source_map
    assert "S4" not in source_map


def test_source_ids_in_evidence_string():
    chunks = [_chunk()]
    evidence_str, _ = build_sourced_evidence(chunks)
    assert "[S1]" in evidence_str


def test_source_map_has_required_fields():
    chunks = [_chunk(spec="23.502", section="5.1", page=100, title="PDU Session")]
    _, source_map = build_sourced_evidence(chunks)
    s = source_map["S1"]
    for field in ("spec", "release", "section", "page", "title"):
        assert field in s
    assert s["spec"] == "23.502"
    assert s["section"] == "5.1"
    assert s["page"] == 100


def test_empty_chunks_returns_empty():
    evidence_str, source_map = build_sourced_evidence([])
    assert evidence_str == ""
    assert source_map == {}


# ── parse_citations ───────────────────────────────────────────────────────────

def test_parse_finds_citations():
    text = "The AMF handles mobility [S1]. The UPF routes packets [S2]."
    cited = parse_citations(text)
    assert cited == ["S1", "S2"]


def test_parse_deduplicates():
    text = "[S1] and [S1] again and [S2]."
    cited = parse_citations(text)
    assert cited == ["S1", "S2"]


def test_parse_empty_text():
    assert parse_citations("No citations here.") == []


def test_parse_preserves_order():
    text = "[S3] first, then [S1], then [S2]."
    cited = parse_citations(text)
    assert cited == ["S3", "S1", "S2"]


# ── validate_citations ────────────────────────────────────────────────────────

def test_validate_all_valid():
    source_map = {
        "S1": {"spec": "23.501", "release": "17", "section": "4.2", "page": 36, "title": "General"},
        "S2": {"spec": "23.502", "release": "17", "section": "5.1", "page": 100, "title": "PDU"},
    }
    answer = "AMF role [S1]. PDU sessions [S2]."
    clean, valid, invalid = validate_citations(answer, source_map)
    assert invalid == []
    assert len(valid) == 2
    assert clean == answer   # no changes needed


def test_validate_removes_invalid():
    source_map = {"S1": {"spec": "23.501", "release": "17", "section": "4.2", "page": 36, "title": ""}}
    answer = "AMF role [S1]. Invented fact [S9]."
    clean, valid, invalid = validate_citations(answer, source_map)
    assert "S9" in invalid
    assert "[INVALID]" in clean
    assert "[S9]" not in clean
    assert len(valid) == 1


def test_validate_citation_has_metadata():
    source_map = {"S1": {"spec": "23.501", "release": "17", "section": "4.2", "page": 36, "title": "General"}}
    clean, valid, _ = validate_citations("fact [S1].", source_map)
    assert valid[0]["id"] == "[S1]"
    assert valid[0]["spec"] == "23.501"
    assert valid[0]["section"] == "4.2"


# ── answer_with_citations (mocked) ───────────────────────────────────────────

def test_answer_with_citations_structure():
    chunks = [_chunk(section=f"4.{i}") for i in range(3)]

    with patch("src.generation.citations.retrieve_and_rerank") as mock_r, \
         patch("src.generation.citations.check") as mock_g, \
         patch("src.generation.citations._get_client") as mock_c:

        mock_r.return_value = chunks
        mock_g.return_value = {"supported": True, "reason": "ok", "evidence": chunks}
        mock_c.return_value.chat.completions.create.return_value = \
            _mock_llm("The AMF manages mobility [S1]. PDU sessions [S2].")

        result = answer_with_citations("What is AMF?")

    assert "answer" in result
    assert "citations" in result
    assert "invalid_ids" in result
    assert result["supported"] is True


def test_answer_with_citations_gate_fail():
    with patch("src.generation.citations.retrieve_and_rerank") as mock_r, \
         patch("src.generation.citations.check") as mock_g:

        mock_r.return_value = []
        mock_g.return_value = {"supported": False, "reason": "weak", "evidence": []}

        result = answer_with_citations("What is chocolate cake?")

    assert result["supported"] is False
    assert result["citations"] == []
    assert result["invalid_ids"] == []


# ── integration ───────────────────────────────────────────────────────────────

def test_live_no_invalid_citations():
    result = answer_with_citations("What is the role of the AMF in 5G core network?")
    assert result["supported"] is True
    assert result["invalid_ids"] == [], f"LLM hallucinated: {result['invalid_ids']}"
    assert len(result["citations"]) > 0
