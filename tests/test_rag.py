"""
tests/test_rag.py
-----------------
Tests for src/rag.py — the complete end-to-end RAG pipeline.

Uses Groq API (free tier).
"""

from unittest.mock import patch, MagicMock
import pytest
from src.rag import answer_question


def _mock_llm(text: str):
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


# ── return structure ───────────────────────────────────────────────────────────

def test_returns_required_keys():
    result = answer_question("What is the capital of France?")
    assert "answer" in result
    assert "sources" in result
    assert "supported" in result


def test_supported_is_bool():
    result = answer_question("What is the capital of France?")
    assert isinstance(result["supported"], bool)


def test_sources_is_list():
    result = answer_question("What is the capital of France?")
    assert isinstance(result["sources"], list)


# ── out-of-domain queries ──────────────────────────────────────────────────────

def test_out_of_domain_not_supported():
    result = answer_question("What is the capital of France?")
    assert result["supported"] is False
    assert result["sources"] == []


def test_unrelated_returns_cannot_answer():
    from src.generation.grok import CANNOT_ANSWER
    result = answer_question("How do I bake a chocolate cake?")
    assert result["supported"] is False
    assert result["answer"] == CANNOT_ANSWER


# ── answerable queries (mocked LLM) ───────────────────────────────────────

def test_answerable_query_supported():
    with patch("src.rag._get_client") as mock_c:
        mock_c.return_value.chat.completions.create.return_value = \
            _mock_llm("The AMF handles access and mobility [S1].")
        result = answer_question("What is the role of the AMF?")
    assert result["supported"] is True
    assert result["answer"] != ""


def test_sources_have_required_fields():
    with patch("src.rag._get_client") as mock_c:
        mock_c.return_value.chat.completions.create.return_value = \
            _mock_llm("AMF manages mobility [S1].")
        result = answer_question("What is the AMF?")
    for s in result["sources"]:
        for field in ("id", "spec", "section", "page"):
            assert field in s


def test_invalid_citation_replaced():
    with patch("src.rag._get_client") as mock_c:
        # LLM invents [S99] which doesn't exist
        mock_c.return_value.chat.completions.create.return_value = \
            _mock_llm("Invented fact [S99].")
        result = answer_question("What is the AMF?")
    if result["supported"]:
        assert "[S99]" not in result["answer"]
        assert "[INVALID]" in result["answer"] or result["sources"] == []


# ── integration: live queries (Groq) ─────────────────────────────────────

@pytest.mark.skip(reason="Requires Groq API key")
@pytest.mark.parametrize("query", [
    "What is the role of the AMF in 5G core network?",
    "Explain the UPF function in the user plane.",
    "What is network slicing and how is NSSAI used?",
    "What are the differences between SSC mode 1, 2, and 3?",
    "What is the NSSF role in network slice selection?",
])
def test_live_answerable(query):
    result = answer_question(query)
    assert result["supported"] is True
    assert len(result["answer"]) > 50
    assert len(result["sources"]) > 0


@pytest.mark.skip(reason="Requires Groq API key")
@pytest.mark.parametrize("query", [
    "What is the capital of France?",
    "How do I train a neural network from scratch?",
])
def test_live_out_of_domain(query):
    result = answer_question(query)
    assert result["supported"] is False
    assert result["sources"] == []
