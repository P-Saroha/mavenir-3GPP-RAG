"""
src/retrieval/quality_gate.py
------------------------------
Evidence quality gate — deterministic checks before calling the LLM.

Checks (in order):
  1. At least MIN_EVIDENCE_COUNT candidates present.
  2. At least one candidate with rerank_score >= MIN_RERANK_SCORE.
  3. Every candidate has the required metadata fields (spec, section, page).

Returns a dict:
    {
        "supported": bool,
        "reason":    str,
        "evidence":  list[dict]   # the passing candidates (empty if not supported)
    }

All thresholds come from config.py and can be overridden via .env.

Usage:
    python -m src.retrieval.quality_gate
"""

from src.utils.config import MIN_RERANK_SCORE, MIN_EVIDENCE_COUNT, REQUIRED_METADATA
from src.retrieval.reranker import retrieve_and_rerank


def check(candidates: list[dict]) -> dict:
    """
    Run the quality gate on a list of reranked candidates.

    Args:
        candidates: output of rerank() — each dict must have 'rerank_score'.

    Returns:
        {"supported": bool, "reason": str, "evidence": list[dict]}
    """
    print(f"[QG DEBUG] Quality gate check: {len(candidates)} candidates received")
    for i, c in enumerate(candidates[:5]):  # Log top 5
        score = c.get("rerank_score", 0.0)
        print(f"[QG DEBUG]   Candidate {i+1}: score={score:.3f}, chunk_id={c.get('chunk_id', '?')}, section={c.get('section', '?')}")
    
    # 1. enough candidates?
    if len(candidates) < MIN_EVIDENCE_COUNT:
        msg = f"Insufficient evidence: only {len(candidates)} candidate(s) retrieved (minimum {MIN_EVIDENCE_COUNT})."
        print(f"[QG DEBUG] REJECTED: {msg}")
        return {
            "supported": False,
            "reason": msg,
            "evidence": [],
        }

    # 2. at least one strong enough score?
    best_score = max(c.get("rerank_score", 0.0) for c in candidates)
    print(f"[QG DEBUG] Best rerank score: {best_score:.3f}, threshold: {MIN_RERANK_SCORE}")
    if best_score < MIN_RERANK_SCORE:
        msg = f"Evidence too weak: best reranker score {best_score:.3f} is below threshold {MIN_RERANK_SCORE}."
        print(f"[QG DEBUG] REJECTED: {msg}")
        return {
            "supported": False,
            "reason": msg,
            "evidence": [],
        }

    # 3. all candidates have required metadata?
    for c in candidates:
        missing = [f for f in REQUIRED_METADATA if not c.get(f)]
        if missing:
            msg = f"Candidate '{c.get('chunk_id', '?')}' missing metadata: {missing}."
            print(f"[QG DEBUG] REJECTED: {msg}")
            return {
                "supported": False,
                "reason": msg,
                "evidence": [],
            }

    msg = f"Evidence supported: {len(candidates)} candidate(s), best score {best_score:.3f}."
    print(f"[QG DEBUG] PASSED: {msg}")
    return {
        "supported": True,
        "reason": msg,
        "evidence": candidates,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_CASES = [
    ("known 3GPP",         "What is the role of the AMF in 5G core network?"),
    ("unrelated",          "What is the capital of France?"),
    ("outside corpus",     "How do I bake a chocolate cake?"),
]


def main():
    print(f"Thresholds: MIN_RERANK_SCORE={MIN_RERANK_SCORE}  "
          f"MIN_EVIDENCE_COUNT={MIN_EVIDENCE_COUNT}\n")

    for label, query in TEST_CASES:
        print(f"── [{label}] ─────────────────────────────────────────────────")
        print(f"   Query: {query}")

        candidates = retrieve_and_rerank(query, top_k=5)
        result = check(candidates)

        print(f"   Supported : {result['supported']}")
        print(f"   Reason    : {result['reason']}")
        if result["supported"]:
            for i, c in enumerate(result["evidence"], 1):
                print(f"   [{i}] score={c['rerank_score']:.3f}  "
                      f"spec={c['spec']}  sec={c['section']}  p={c['page']}")
        print()


if __name__ == "__main__":
    main()
