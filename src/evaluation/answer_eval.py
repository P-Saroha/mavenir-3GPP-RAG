"""
src/evaluation/answer_eval.py
------------------------------
Evaluates answer quality against the gold eval dataset.

All checks are deterministic — no LLM used for scoring.

Rate-limit handling:
  Waits INTER_QUESTION_DELAY seconds between questions to stay under the
  Groq free-tier tokens-per-minute limit. On 429, retries with exponential
  backoff. Saves after every question so a run can be resumed.

Usage:
    python -m src.evaluation.answer_eval
    python -m src.evaluation.answer_eval --report   # print from saved file
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.rag import answer_question

EVAL_PATH   = Path("data/eval_questions.json")
OUTPUT_PATH = Path("data/answer_results.json")

MIN_KEYWORD_OVERLAP  = 0.25   # threshold for "correct" answer
MAX_RETRIES          = 8      # retries on 429
RETRY_BASE_SECS      = 60     # first wait; doubles each retry (60,120,240,480,600...)
INTER_QUESTION_DELAY = 15     # seconds between questions to avoid TPM limit

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","to","of","in","for","on","with","by",
    "at","from","as","it","its","this","that","these","those","and","or",
    "but","if","when","where","how","what","which","who","not","no","also",
    "3gpp","release","17","ts","clause","section","figure",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def keyword_overlap(answer: str, expected_summary: str) -> float:
    exp = _tokens(expected_summary)
    if not exp:
        return 0.0
    return len(exp & _tokens(answer)) / len(exp)


def citation_hits_gold(citations: list[dict], spec: str, section: str) -> bool:
    return any(c.get("spec") == spec and c.get("section") == section
               for c in citations)


def _call_with_retry(question: str) -> dict:
    wait = RETRY_BASE_SECS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return answer_question(question)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"    [rate limit] waiting {wait}s  (retry {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                wait = min(wait * 2, 600)
            else:
                raise
    raise RuntimeError(f"All {MAX_RETRIES} retries failed for: {question}")


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate() -> dict:
    questions    = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    answerable   = [q for q in questions if q.get("expected_spec")]
    unanswerable = [q for q in questions if not q.get("expected_spec")]

    # resume from partial results if available
    completed_ids: set[str] = set()
    per_question: list[dict] = []
    if OUTPUT_PATH.exists():
        try:
            prev = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            per_question  = prev.get("per_question", [])
            completed_ids = {r["id"] for r in per_question}
            if completed_ids:
                print(f"Resuming — {len(completed_ids)}/30 already done.\n")
        except Exception:
            pass

    print(f"Evaluating {len(questions)} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)...\n")

    # ── answerable ─────────────────────────────────────────────────────────────
    print("── Answerable questions ─────────────────────────────────────────")
    for q in answerable:
        if q["id"] in completed_ids:
            print(f"  {q['id']:12s}  (cached)")
            continue

        result    = _call_with_retry(q["question"])
        time.sleep(INTER_QUESTION_DELAY)

        abstained = not result["supported"]
        overlap   = 0.0
        cit_hit   = False
        if result["supported"]:
            overlap  = keyword_overlap(result["answer"], q["expected_answer_summary"])
            cit_hit  = citation_hits_gold(
                result["sources"], q["expected_spec"], q["expected_section"]
            )

        row = {
            "id":                q["id"],
            "category":          q["category"],
            "question":          q["question"],
            "expected_spec":     q["expected_spec"],
            "expected_section":  q["expected_section"],
            "expected_summary":  q["expected_answer_summary"],
            "answer":            result["answer"],
            "supported":         result["supported"],
            "sources":           result["sources"],
            "keyword_overlap":   round(overlap, 3),
            "citation_hit":      cit_hit,
            "wrongly_abstained": abstained,
        }
        per_question.append(row)
        _save_partial(per_question)

        status = ("ABSTAINED" if abstained
                  else f"kw={overlap:.2f}  cit={'HIT' if cit_hit else 'MISS'}")
        print(f"  {q['id']:12s}  {q['expected_spec']} §{q['expected_section']:12s}  {status}")

    # ── unanswerable ───────────────────────────────────────────────────────────
    print("\n── Unanswerable questions ───────────────────────────────────────")
    for q in unanswerable:
        if q["id"] in completed_ids:
            print(f"  {q['id']:12s}  (cached)")
            continue

        result  = _call_with_retry(q["question"])
        time.sleep(INTER_QUESTION_DELAY)

        refused = not result["supported"]
        row = {
            "id":                q["id"],
            "category":          "unanswerable",
            "question":          q["question"],
            "expected_spec":     None,
            "expected_section":  None,
            "expected_summary":  q["expected_answer_summary"],
            "answer":            result["answer"],
            "supported":         result["supported"],
            "sources":           result["sources"],
            "keyword_overlap":   None,
            "citation_hit":      None,
            "correctly_refused": refused,
        }
        per_question.append(row)
        _save_partial(per_question)
        print(f"  {q['id']:12s}  {'REFUSED [OK]' if refused else 'ANSWERED (wrong!)'}")

    return _compute_output(per_question)


def _save_partial(rows: list[dict]):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"metrics": {}, "per_question": rows}, indent=2, ensure_ascii=False)
    )


def _compute_output(per_question: list[dict]) -> dict:
    ans_rows   = [r for r in per_question if r.get("expected_spec")]
    unans_rows = [r for r in per_question if not r.get("expected_spec")]

    n_ans   = len(ans_rows)
    n_unans = len(unans_rows)

    kw       = [r["keyword_overlap"] or 0.0 for r in ans_rows]
    cit      = [r["citation_hit"]           for r in ans_rows]
    abstained= [r["wrongly_abstained"]       for r in ans_rows]
    refused  = [r.get("correctly_refused", False) for r in unans_rows]

    metrics = {
        "n_answerable":                  n_ans,
        "n_unanswerable":                n_unans,
        "correctness":                   round(sum(1 for s in kw if s >= MIN_KEYWORD_OVERLAP) / n_ans, 4) if n_ans else 0.0,
        "mean_keyword_overlap":          round(sum(kw) / n_ans, 4) if n_ans else 0.0,
        "citation_accuracy":             round(sum(1 for h in cit if h) / n_ans, 4) if n_ans else 0.0,
        "abstention_accuracy":           round(sum(1 for r in refused if r) / n_unans, 4) if n_unans else None,
        "unsupported_rate":              round(sum(1 for a in abstained if a) / n_ans, 4) if n_ans else 0.0,
        "min_keyword_overlap_threshold": MIN_KEYWORD_OVERLAP,
    }
    return {"metrics": metrics, "per_question": per_question}


# ── report ────────────────────────────────────────────────────────────────────

def print_report(metrics: dict):
    thr = metrics["min_keyword_overlap_threshold"]
    print("\n" + "=" * 57)
    print("  ANSWER QUALITY EVALUATION REPORT")
    print("=" * 57)
    print(f"  Questions    : {metrics['n_answerable']} answerable, {metrics['n_unanswerable']} unanswerable")
    print(f"  KW threshold : >= {thr}")
    print("-" * 57)
    print(f"  Correctness          : {metrics['correctness']:.4f}  (kw overlap >= {thr})")
    print(f"  Mean keyword overlap : {metrics['mean_keyword_overlap']:.4f}")
    print(f"  Citation accuracy    : {metrics['citation_accuracy']:.4f}  (gold section cited)")
    if metrics["abstention_accuracy"] is not None:
        print(f"  Abstention accuracy  : {metrics['abstention_accuracy']:.4f}  (unanswerable refused)")
    print(f"  Unsupported rate     : {metrics['unsupported_rate']:.4f}  (answerable wrongly refused)")
    print("=" * 57)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="Print report from saved file without calling the LLM")
    args = parser.parse_args()

    if args.report:
        if not OUTPUT_PATH.exists():
            print(f"No results at {OUTPUT_PATH}. Run without --report first.")
            return
        data   = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        output = _compute_output(data["per_question"])
    else:
        t0     = time.time()
        output = evaluate()
        print(f"\nTotal time : {time.time() - t0:.1f}s")

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print_report(output["metrics"])
    print(f"Results    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
