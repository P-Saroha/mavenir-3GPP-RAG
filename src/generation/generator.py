"""
src/generation/generator.py
-----------------------------
Unified LLM generation with automatic fallback.

Priority order:
  1. Groq API  (if GROQ_API_KEY is set)
  2. xAI Grok  (if XAI_API_KEY is set)
  3. Ollama    (if running locally)
  4. Error     (if all three fail)

This module is the single import used by the API and UI layers.
It re-exports the full `answer()` pipeline from grok.py and adds
generate_answer() as a lower-level function that accepts pre-built evidence.

Usage:
    from src.generation.generator import answer, generate_answer
"""

from __future__ import annotations

from src.utils.config import GROQ_API_KEY, XAI_API_KEY
from src.generation.grok import (
    SYSTEM_PROMPT,
    CANNOT_ANSWER,
    answer,           # full pipeline (retrieve → gate → generate)
    generate,         # low-level: query + evidence → LLM response
)


def generate_answer(query: str, evidence: str) -> str:
    """
    Generate an answer given a pre-built evidence string.

    Tries providers in priority order:
      1. Groq  (if GROQ_API_KEY set)
      2. xAI   (if XAI_API_KEY set)
      3. Ollama (local fallback)

    Returns the answer string, or CANNOT_ANSWER if all providers fail.
    """
    # ── Try cloud provider (Groq or xAI via grok.generate) ────────────────────
    if GROQ_API_KEY or XAI_API_KEY:
        try:
            return generate(query, evidence)
        except Exception as e:
            print(f"[generator] Cloud provider failed: {e}")
            print("[generator] Falling back to Ollama ...")

    # ── Try local Ollama ───────────────────────────────────────────────────────
    from src.generation.local import generate_with_local_llm, is_ollama_available

    if is_ollama_available():
        try:
            prompt = f"Question: {query}\n\nEvidence:\n{evidence}"
            return generate_with_local_llm(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            print(f"[generator] Ollama failed: {e}")

    # ── All providers failed ───────────────────────────────────────────────────
    return CANNOT_ANSWER


# Re-export for convenience
__all__ = ["answer", "generate_answer", "CANNOT_ANSWER"]
