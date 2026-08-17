"""
src/generation/grok.py
-----------------------
Grounded LLM generation using the Groq API (ONLY LLM provider).

The Groq API is OpenAI-compatible — we use the openai SDK pointed at
https://api.groq.com/openai/v1.  No separate Groq SDK required.

The LLM receives ONLY the user question and the evidence package.
It is instructed never to invent facts outside the supplied evidence.

If Groq API is unavailable or quota exhausted, fails gracefully with clear error.

Usage:
    python -m src.generation.grok
"""

from __future__ import annotations

from openai import OpenAI

from src.utils.config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
)
from src.retrieval.reranker import retrieve_and_rerank
from src.retrieval.quality_gate import check
from src.retrieval.context_builder import build_evidence

# Export the active model name for tests
ACTIVE_MODEL = GROQ_MODEL

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a 3GPP Release 17 5G Core standards assistant.

Rules:
- Use ONLY the evidence provided below. Do not use outside knowledge.
- Do not invent technical facts, section numbers, specification numbers,
  procedures, network functions, or references.
- Every factual claim must cite the evidence supplied (spec and section).
- If the evidence is insufficient to answer, respond exactly:
  "I cannot reliably answer this from the provided 3GPP Release 17 5G Core specifications."
- Be concise and technical."""

CANNOT_ANSWER = (
    "I cannot reliably answer this from the provided "
    "3GPP Release 17 5G Core specifications."
)

# ── client singleton ──────────────────────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set in .env. "
                "Groq is the ONLY LLM provider. "
                "Get a free API key at https://console.groq.com"
            )
        _client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _client


# ── generation ────────────────────────────────────────────────────────────────

def generate(query: str, evidence: str) -> str:
    """
    Call Grok with the query and pre-built evidence string.
    Returns the model's answer as a plain string.
    
    Raises:
        ValueError: if GROK_API_KEY is not set
        Exception: if Grok API call fails (no fallback)
    """
    client = _get_client()

    user_message = f"""Question: {query}

Evidence:
{evidence}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,   # deterministic — grounded answers only
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def answer(query: str, top_k: int = 5) -> dict:
    """
    Full pipeline: retrieve → quality gate → build evidence → generate.

    Returns:
        {
            "query":     str,
            "supported": bool,
            "answer":    str,
            "evidence":  list[dict]  (empty if gate failed)
        }
    """
    candidates = retrieve_and_rerank(query, top_k=top_k)
    gate = check(candidates)

    if not gate["supported"]:
        return {
            "query":     query,
            "supported": False,
            "answer":    CANNOT_ANSWER,
            "evidence":  [],
        }

    evidence_text = build_evidence(gate["evidence"])
    llm_answer = generate(query, evidence_text)

    return {
        "query":     query,
        "supported": True,
        "answer":    llm_answer,
        "evidence":  gate["evidence"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What is the role of the AMF in 5G core network?",
    "How does PDU session establishment work?",
    "What is network slicing and how is NSSAI used?",
    "Explain the UPF function in the user plane.",
    "How does the SMF interact with the PCF for policy control?",
]


def main():
    print(f"LLM Provider: Groq  |  Model: {GROQ_MODEL}\n")

    for query in TEST_QUERIES:
        print("=" * 65)
        print(f"Q: {query}")
        print("=" * 65)
        result = answer(query)
        print(f"Supported: {result['supported']}")
        print(f"\nA: {result['answer']}")
        if result["supported"]:
            print(f"\nSources:")
            for e in result["evidence"]:
                print(f"  [{e['spec']} §{e['section']} p.{e['page']}]")
        print()


if __name__ == "__main__":
    main()
