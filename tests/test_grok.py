"""
tests/test_grok.py
-------------------
Tests for src/generation/grok.py.

API calls are mocked so tests pass without xAI credits.
One integration test is marked with a skip flag — remove the skip
when credits are available.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_response(text: str):
    """Build a fake OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _fake_candidates(n=3, score=5.0):
    return [
        {
            "chunk_id": f"id-{i}",
            "rerank_score": score,
            "spec": "23.501",
            "section": f"4.{i}",
            "page": i + 1,
            "text": f"The AMF handles access and mobility management procedure {i}.",
        }
        for i in range(n)
    ]


# ── unit: generate() ─────────────────────────────────────────────────────────

def test_generate_calls_api_and_returns_string():
    from src.generation.grok import generate

    fake_answer = "The AMF manages access and mobility per TS 23.501 §4.2."
    with patch("src.generation.grok._get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(fake_answer)
        mock_client_fn.return_value = mock_client

        result = generate("What is the AMF?", "evidence text here")

    assert isinstance(result, str)
    assert result == fake_answer


def test_generate_passes_system_prompt():
    from src.generation.grok import generate, SYSTEM_PROMPT

    with patch("src.generation.grok._get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response("answer")
        mock_client_fn.return_value = mock_client

        generate("question", "evidence")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "3GPP" in messages[0]["content"]


def test_generate_uses_temperature_zero():
    from src.generation.grok import generate

    with patch("src.generation.grok._get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response("ans")
        mock_client_fn.return_value = mock_client

        generate("q", "evidence")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.0


def test_generate_uses_configured_model():
    from src.generation.grok import generate, ACTIVE_MODEL

    with patch("src.generation.grok._get_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response("ans")
        mock_client_fn.return_value = mock_client

        generate("q", "evidence")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == ACTIVE_MODEL


# ── unit: answer() ────────────────────────────────────────────────────────────

def test_answer_returns_cannot_answer_when_gate_fails():
    from src.generation.grok import answer, CANNOT_ANSWER

    with patch("src.generation.grok.retrieve_and_rerank") as mock_rerank, \
         patch("src.generation.grok.check") as mock_gate:

        mock_rerank.return_value = _fake_candidates(score=-5.0)
        mock_gate.return_value = {
            "supported": False,
            "reason": "Evidence too weak",
            "evidence": [],
        }

        result = answer("What is the capital of France?")

    assert result["supported"] is False
    assert result["answer"] == CANNOT_ANSWER
    assert result["evidence"] == []


def test_answer_returns_llm_response_when_gate_passes():
    from src.generation.grok import answer

    fake_llm_answer = "The AMF is responsible for access and mobility management."

    with patch("src.generation.grok.retrieve_and_rerank") as mock_rerank, \
         patch("src.generation.grok.check") as mock_gate, \
         patch("src.generation.grok.build_evidence") as mock_evidence, \
         patch("src.generation.grok.generate") as mock_generate:

        mock_rerank.return_value = _fake_candidates()
        mock_gate.return_value = {
            "supported": True,
            "reason": "OK",
            "evidence": _fake_candidates(),
        }
        mock_evidence.return_value = "evidence text"
        mock_generate.return_value = fake_llm_answer

        result = answer("What is the AMF?")

    assert result["supported"] is True
    assert result["answer"] == fake_llm_answer
    assert len(result["evidence"]) > 0


def test_answer_result_has_required_keys():
    from src.generation.grok import answer

    with patch("src.generation.grok.retrieve_and_rerank") as mock_rerank, \
         patch("src.generation.grok.check") as mock_gate:

        mock_rerank.return_value = []
        mock_gate.return_value = {"supported": False, "reason": "x", "evidence": []}

        result = answer("test")

    for key in ("query", "supported", "answer", "evidence"):
        assert key in result


# ── integration (requires xAI credits) ───────────────────────────────────────

def test_live_amf_question():
    from src.generation.grok import answer
    result = answer("What is the role of the AMF in 5G core network?")
    assert result["supported"] is True
    assert "AMF" in result["answer"]
    assert len(result["evidence"]) > 0
