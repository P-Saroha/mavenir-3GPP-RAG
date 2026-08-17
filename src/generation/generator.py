"""
src/generation/generator.py
-----------------------------
Unified LLM generation using xAI Grok API (ONLY provider).

No fallback providers. If Grok API is unavailable or quota exhausted,
fails gracefully with clear error message.

This module is the single import used by the API and UI layers.
It re-exports the full `answer()` pipeline from grok.py and provides
generate_answer() for lower-level usage with pre-built evidence.

Usage:
    from src.generation.generator import answer, generate_answer
"""

from __future__ import annotations

from src.generation.grok import (
    SYSTEM_PROMPT,
    CANNOT_ANSWER,
    answer,           # full pipeline (retrieve → gate → generate)
    generate,         # low-level: query + evidence → Grok response
)


def generate_answer(query: str, evidence: str) -> str:
    """
    Generate an answer given a pre-built evidence string.

    Uses xAI Grok API (ONLY provider).
    
    Raises:
        ValueError: if GROK_API_KEY not set
        Exception: if Grok API call fails (no fallback)

    Args:
        query: user question
        evidence: pre-built evidence string

    Returns:
        answer string from Grok, or CANNOT_ANSWER if generation fails
    """
    return generate(query, evidence)


# Re-export for convenience
__all__ = ["answer", "generate_answer", "CANNOT_ANSWER"]
