"""
src/evaluation/answer_eval.py
------------------------------
Evaluates answer quality against the gold eval dataset.

All checks are deterministic — no LLM used for scoring.

Metrics:
  correctness        — fraction of answerable questions where the answer
                       contains >= MIN_KEYWORD_OVERLAP keywords from the
                       expected_answer_summary (normalised token overlap)
  citation_accuracy  — fraction of answerable questions where at least one
                       cited source matches (gold_spec, gold_section)
  abstention_accuracy— fraction of unanswerable questions correctly refused
  unsupported_rate   — fraction of answerable questions where the pipeline
                       returned supported=False (should be low)

Rate-limit handling:
  If the LLM API returns 429, the evaluator waits (exponential backoff up to
  10 minutes) and retries automatically. Progress is saved after each question
  so a run can be resumed after a hard stop.

Usage:
    python -m src.evaluation.answer_eval            # run full evaluation
    python -m src.evaluation.answer_eval --report   # print report from existing results
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

MIN_KEYWORD_OVERLAP = 0.25      # threshold for "correct" answer
MAX_RETRIES         = 6         # retries on 429
RETRY_BASE_SECS     = 15        # first wait; doubles each retry (15,30,60,120,240,480)

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
    """Fraction of expected summary tokens present in answer (0–1)."""
    exp = _tokens(expected_summary)
    if not exp:
        return 0.0
    return len(exp & _tokens(answer)) / len(exp)


def citation_hits_gold(citations: list[dict], spec: str, section: str) -> bool:
    """True if any cited source matches (spec, section)."""
    return any(c.get("spec") == spec and c.get("section") == section
               for c in citations)


def _call_with_retry(question: str) -> dict:
    """Call answer_question() with exponential backoff on rate-limit errors."""
    wait = RETRY_BASE_SECS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return answer_question(question)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"    [rate limit] waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
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

    # load partial results if we were interrupted previously
    completed_ids: set[str] = set()
    per_question: list[dict] = []
    if OUTPUT_PATH.exists():
        try:
            prev = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            per_question = prev.get("per_question", [])
            completed_ids = {r["id"] for r in per_question}
            if completed_ids:
                print(f"Resuming — {len(completed_ids)} questions already done.\n")
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

        result   = _call_with_retry(q["question"])
        abstained = not result["supported"]
        overlap  = 0.0
        cit_hit  = False

        if result["supported"]:
            overlap = keyword_overlap(result["answer"], q["expected_answer_summary"])
            cit_hit = citation_hits_gold(
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

        # save after every question so we can resume on interruption
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

        print(f"  {q['id']:12s}  {'REFUSED ✓' if refused else 'ANSWERED (wrong!)'}")

    return _compute_output(per_question)


def _save_partial(per_question: list[dict]):
    """Save partial results so we can resume on rate-limit interruption."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"metrics": {}, "per_question": per_question},
                   indent=2, ensure_ascii=False)
    )


def _compute_output(per_question: list[dict]) -> dict:
    """Compute aggregate metrics from a completed per_question list."""
    answerable_rows   = [r for r in per_question if r.get("expected_spec")]
    unanswerable_rows = [r for r in per_question if not r.get("expected_spec")]

    n_ans   = len(answerable_rows)
    n_unans = len(unanswerable_rows)

    kw_scores   = [r["keyword_overlap"] or 0.0 for r in answerable_rows]
    cit_correct = [r["citation_hit"]          for r in answerable_rows]
    abstained   = [r["wrongly_abstained"]      for r in answerable_rows]
    refused     = [r.get("correctly_refused", False) for r in unanswerable_rows]

    metrics = {
        "n_answerable":          n_ans,
        "n_unanswerable":        n_unans,
        "correctness":           round(
            sum(1 for s in kw_scores if s >= MIN_KEYWORD_OVERLAP) / n_ans, 4
        ) if n_ans else 0.0,
        "mean_keyword_overlap":  round(sum(kw_scores) / n_ans, 4) if n_ans else 0.0,
        "citation_accuracy":     round(
            sum(1 for h in cit_correct if h) / n_ans, 4
        ) if n_ans else 0.0,
        "abstention_accuracy":   round(
            sum(1 for r in refused if r) / n_unans, 4
        ) if n_unans else None,
        "unsupported_rate":      round(
            sum(1 for a in abstained if a) / n_ans, 4
        ) if n_ans else 0.0,
        "min_keyword_overlap_threshold": MIN_KEYWORD_OVERLAP,
    }

    return {"metrics": metrics, "per_question": per_question}


# ── report ────────────────────────────────────────────────────────────────────

def print_report(metrics: dict):
    n_ans   = metrics["n_answerable"]
    n_unans = metrics["n_unanswerable"]
    thr     = metrics["min_keyword_overlap_threshold"]

    print("\n" + "=" * 57)
    print("  ANSWER QUALITY EVALUATION REPORT")
    print("=" * 57)
    print(f"  Questions  : {n_ans} answerable, {n_unans} unanswerable")
    print(f"  KW threshold : >= {thr}")
    print("-" * 57)
    print(f"  Correctness          : {metrics['correctness']:.4f}  "
          f"(kw overlap >= {thr})")
    print(f"  Mean keyword overlap : {metrics['mean_keyword_overlap']:.4f}")
    print(f"  Citation accuracy    : {metrics['citation_accuracy']:.4f}  "
          f"(gold section cited)")
    if metrics["abstention_accuracy"] is not None:
        print(f"  Abstention accuracy  : {metrics['abstention_accuracy']:.4f}  "
              f"(unanswerable refused)")
    print(f"  Unsupported rate     : {metrics['unsupported_rate']:.4f}  "
          f"(answerable wrongly refused)")
    print("=" * 57)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", action="store_true",
        help="Print report from existing results file without calling the LLM"
    )
    args = parser.parse_args()

    if args.report:
        if not OUTPUT_PATH.exists():
            print(f"No results file found at {OUTPUT_PATH}. Run without --report first.")
            return
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        # recompute metrics in case file was saved mid-run
        output = _compute_output(data["per_question"])
    else:
        t0 = time.time()
        output = evaluate()
        print(f"\nTotal time : {time.time() - t0:.1f}s")

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print_report(output["metrics"])
    print(f"Results    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
