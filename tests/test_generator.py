"""
tests/test_generator.py
------------------------
Tests for src/generation/generator.py.

The generator now uses ONLY xAI Grok API. No fallbacks to Groq or Ollama.
All external calls are mocked — no xAI credits or running API required.
"""

from unittest.mock import patch, MagicMock
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_response(text: str):
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


# ── generator.py: generate_answer with Grok-only ──────────────────────────────

def test_generate_answer_calls_grok():
    """generate_answer should call grok.generate() which is re-exported."""
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate") as mock_grok_gen:
        mock_grok_gen.return_value = "Grok answer"
        result = generate_answer("question", "evidence")
    assert result == "Grok answer"
    mock_grok_gen.assert_called_once_with("question", "evidence")


def test_generate_answer_passes_query_and_evidence():
    """generate_answer should pass query and evidence to grok.generate()."""
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate") as mock_grok_gen:
        mock_grok_gen.return_value = "answer"
        generate_answer("What is the AMF?", "AMF handles mobility.")
        
        # Check that grok.generate was called with correct args
        mock_grok_gen.assert_called_once()
        args = mock_grok_gen.call_args
        assert args[0] == ("What is the AMF?", "AMF handles mobility.")


def test_generate_answer_returns_string():
    """generate_answer must return a string."""
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate") as mock_grok_gen:
        mock_grok_gen.return_value = "test answer"
        result = generate_answer("q", "e")
    assert isinstance(result, str)
    assert result == "test answer"


def test_generate_answer_raises_on_grok_error():
    """If Grok API fails, the exception propagates (no fallback)."""
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate", side_effect=ValueError("GROK_API_KEY not set")):
        with pytest.raises(ValueError, match="GROK_API_KEY"):
            generate_answer("q", "e")


def test_generate_answer_raises_on_grok_api_failure():
    """If Grok API call fails, exception propagates."""
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate", side_effect=Exception("API error")):
        with pytest.raises(Exception):
            generate_answer("q", "e")


# ── exports ───────────────────────────────────────────────────────────────────

def test_generator_exports_answer():
    """generator should re-export answer from grok for full pipeline."""
    from src.generation.generator import answer
    assert callable(answer)


def test_generator_exports_cannot_answer_constant():
    """generator should re-export CANNOT_ANSWER constant."""
    from src.generation.generator import CANNOT_ANSWER
    assert isinstance(CANNOT_ANSWER, str)
    assert "cannot" in CANNOT_ANSWER.lower()


# ── integration: live Grok call ───────────────────────────────────────────────

def test_live_generate_answer():
    """
    Integration test: call actual Grok API (requires GROK_API_KEY in .env).
    Skip if API key not available.
    """
    import os
    if not os.getenv("GROK_API_KEY"):
        pytest.skip("GROK_API_KEY not set")
    
    from src.generation.generator import generate_answer
    evidence = "[23.501 §4.2.2 p.37]\nThe AMF handles access and mobility management."
    result = generate_answer("What does AMF do?", evidence)
    assert isinstance(result, str)
    assert len(result) > 10
    assert "AMF" in result or "cannot" in result.lower()
