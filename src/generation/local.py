"""
src/generation/local.py
------------------------
Local LLM fallback using Ollama.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
so we reuse the openai SDK — no extra dependency needed.

Recommended model: mistral  (non-Chinese, open-weight, runs on CPU)
  ollama pull mistral

Usage:
    python -m src.generation.local
"""

from __future__ import annotations

from openai import OpenAI

from src.utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Ollama does not require a real API key but the openai client expects one
_DUMMY_KEY = "ollama"


def generate_with_local_llm(prompt: str, system: str = "") -> str:
    """
    Send a prompt to the local Ollama server and return the response text.

    Args:
        prompt: the full user message (query + evidence already formatted)
        system: optional system message

    Raises:
        RuntimeError if Ollama is not running or returns an error.
    """
    client = OpenAI(api_key=_DUMMY_KEY, base_url=OLLAMA_BASE_URL)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e


def is_ollama_available() -> bool:
    """Return True if Ollama is reachable at OLLAMA_BASE_URL."""
    import httpx
    try:
        # Ollama's health endpoint (strip /v1 suffix for the base check)
        base = OLLAMA_BASE_URL.replace("/v1", "")
        httpx.get(base, timeout=2.0)
        return True
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Ollama URL  : {OLLAMA_BASE_URL}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Available   : {is_ollama_available()}\n")

    if not is_ollama_available():
        print("Ollama is not running.")
        print("Start it with:  ollama serve")
        print(f"Pull the model: ollama pull {OLLAMA_MODEL}")
        return

    prompt = "What is the role of the AMF in 5G?\n\nEvidence: The AMF handles access and mobility."
    print(f"Test prompt: {prompt}\n")
    result = generate_with_local_llm(prompt)
    print(f"Response: {result}")


if __name__ == "__main__":
    main()
