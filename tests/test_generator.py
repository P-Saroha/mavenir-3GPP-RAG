"""
tests/test_generator.py
------------------------
Tests for src/generation/local.py and src/generation/generator.py.
All external calls are mocked — no GPU or running Ollama required.
"""

from unittest.mock import patch, MagicMock
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_response(text: str):
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


# ── local.py: generate_with_local_llm ────────────────────────────────────────

def test_local_generate_returns_string():
    from src.generation.local import generate_with_local_llm
    with patch("src.generation.local.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("local answer")
        mock_openai.return_value = client

        result = generate_with_local_llm("test prompt")
    assert result == "local answer"


def test_local_generate_includes_system_message():
    from src.generation.local import generate_with_local_llm
    with patch("src.generation.local.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("ok")
        mock_openai.return_value = client

        generate_with_local_llm("prompt", system="you are a helper")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "you are a helper"


def test_local_generate_raises_on_failure():
    from src.generation.local import generate_with_local_llm
    with patch("src.generation.local.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("connection refused")
        mock_openai.return_value = client

        with pytest.raises(RuntimeError, match="Ollama call failed"):
            generate_with_local_llm("prompt")


def test_is_ollama_available_true():
    from src.generation.local import is_ollama_available
    with patch("httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert is_ollama_available() is True


def test_is_ollama_available_false():
    from src.generation.local import is_ollama_available
    with patch("httpx.get", side_effect=Exception("refused")):
        assert is_ollama_available() is False


# ── generator.py: generate_answer fallback chain ─────────────────────────────

def test_generate_answer_uses_cloud_when_available():
    from src.generation.generator import generate_answer
    with patch("src.generation.generator.generate") as mock_generate, \
         patch("src.generation.generator.GROQ_API_KEY", "gsk_test"):
        mock_generate.return_value = "cloud answer"
        result = generate_answer("question", "evidence")
    assert result == "cloud answer"


def test_generate_answer_falls_back_to_ollama():
    from src.generation.generator import generate_answer, CANNOT_ANSWER
    with patch("src.generation.generator.generate", side_effect=Exception("API down")), \
         patch("src.generation.generator.GROQ_API_KEY", "gsk_test"), \
         patch("src.generation.local.is_ollama_available", return_value=True), \
         patch("src.generation.local.generate_with_local_llm", return_value="ollama answer"):
        result = generate_answer("question", "evidence")
    assert result == "ollama answer"


def test_generate_answer_returns_cannot_answer_when_all_fail():
    from src.generation.generator import generate_answer, CANNOT_ANSWER
    with patch("src.generation.generator.generate", side_effect=Exception("API down")), \
         patch("src.generation.generator.GROQ_API_KEY", "gsk_test"), \
         patch("src.generation.local.is_ollama_available", return_value=False):
        result = generate_answer("question", "evidence")
    assert result == CANNOT_ANSWER


def test_generate_answer_skips_cloud_when_no_keys():
    from src.generation.generator import generate_answer, CANNOT_ANSWER
    with patch("src.generation.generator.GROQ_API_KEY", ""), \
         patch("src.generation.generator.XAI_API_KEY", ""), \
         patch("src.generation.local.is_ollama_available", return_value=False):
        result = generate_answer("question", "evidence")
    assert result == CANNOT_ANSWER


# ── integration: live Groq call ───────────────────────────────────────────────

def test_live_generate_answer():
    from src.generation.generator import generate_answer
    evidence = "[23.501 §4.2.2 p.37]\nThe AMF handles access and mobility management."
    result = generate_answer("What does AMF stand for?", evidence)
    assert isinstance(result, str)
    assert len(result) > 10
    assert "AMF" in result or "cannot" in result.lower()
