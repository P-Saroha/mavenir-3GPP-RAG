"""
src/evaluation/retrieval_eval.py
---------------------------------
Evaluates retrieval quality against the gold eval dataset.

Metrics (computed on the 28 answerable questions):
  Hit@k  — fraction of questions where the gold section appears in top-k results
  MRR    — Mean Reciprocal Rank (1/rank of first gold hit, 0 if not found)

Systems compared:
  1. dense_only   — Qdrant vector search
  2. bm25_only    — BM25 sparse search
  3. hybrid_rrf   — BM25 + dense fused with RRF
  4. hybrid_rerank — hybrid_rrf + cross-encoder reranking

Gold match: result is a hit if result['spec'] == gold_spec
                              AND result['section'] == gold_section

Output:
  data/retrieval_results.json  — full per-question breakdown
  stdout                       — comparison table

Usage:
    python -m src.evaluation.retrieval_eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.retrieval.dense_search  import search as dense_search
from src.retrieval.bm25          import search as bm25_search
from src.retrieval.hybrid        import hybrid_search
from src.retrieval.reranker      import rerank

EVAL_PATH    = Path("data/eval_questions.json")
OUTPUT_PATH  = Path("data/retrieval_results.json")
TOP_K        = 5    # evaluate Hit@1, Hit@3, Hit@5 and MRR up to rank 5


# ── metrics ───────────────────────────────────────────────────────────────────

def _rank_of_hit(results: list[dict], gold_spec: str, gold_section: str) -> int | None:
    """Return 1-based rank of first gold hit, or None if not found in results."""
    for i, r in enumerate(results, start=1):
        if r.get("spec") == gold_spec and r.get("section") == gold_section:
            return i
    return None


def compute_metrics(ranks: list[int | None], k_values: list[int]) -> dict:
    n = len(ranks)
    metrics: dict[str, float] = {}
    for k in k_values:
        hit_k = sum(1 for r in ranks if r is not None and r <= k) / n
        metrics[f"hit@{k}"] = round(hit_k, 4)
    mrr = sum((1 / r) for r in ranks if r is not None) / n
    metrics["mrr"] = round(mrr, 4)
    return metrics


# ── retrieval wrappers ────────────────────────────────────────────────────────

def run_dense(query: str) -> list[dict]:
    return dense_search(query, top_k=TOP_K)


def run_bm25(query: str) -> list[dict]:
    return bm25_search(query, top_k=TOP_K)


def run_hybrid(query: str) -> list[dict]:
    return hybrid_search(query, top_k=TOP_K)


def run_hybrid_rerank(query: str) -> list[dict]:
    candidates = hybrid_search(query, top_k=20)
    return rerank(query, candidates, top_k=TOP_K)


SYSTEMS = [
    ("dense_only",     run_dense),
    ("bm25_only",      run_bm25),
    ("hybrid_rrf",     run_hybrid),
    ("hybrid_rerank",  run_hybrid_rerank),
]


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate() -> dict:
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q.get("expected_spec")]
    print(f"Evaluating {len(answerable)} answerable questions across {len(SYSTEMS)} systems...\n")

    all_results: dict[str, list] = {name: [] for name, _ in SYSTEMS}
    per_question: list[dict] = []

    for q in answerable:
        qid   = q["id"]
        query = q["question"]
        spec  = q["expected_spec"]
        sec   = q["expected_section"]
        row   = {"id": qid, "question": query, "gold_spec": spec, "gold_section": sec}

        for name, fn in SYSTEMS:
            results = fn(query)
            rank = _rank_of_hit(results, spec, sec)
            row[f"{name}_rank"] = rank
            all_results[name].append(rank)

        per_question.append(row)
        print(f"  {qid:12s}  gold={spec} §{sec:12s}  "
              + "  ".join(
                  f"{name}=r{row[f'{name}_rank'] or '-'}"
                  for name, _ in SYSTEMS
              ))

    # compute metrics for each system
    k_values = [1, 3, 5]
    metrics: dict[str, dict] = {}
    for name, _ in SYSTEMS:
        metrics[name] = compute_metrics(all_results[name], k_values)

    return {"metrics": metrics, "per_question": per_question}


# ── printing ──────────────────────────────────────────────────────────────────

def print_table(metrics: dict[str, dict]):
    systems = list(metrics.keys())
    keys    = ["hit@1", "hit@3", "hit@5", "mrr"]

    col = 16
    header = f"{'System':<{col}}" + "".join(f"{k:>{col}}" for k in keys)
    print("\n" + "=" * (col * (len(keys) + 1)))
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * (col * (len(keys) + 1)))
    print(header)
    print("-" * (col * (len(keys) + 1)))
    for name in systems:
        row = f"{name:<{col}}" + "".join(f"{metrics[name][k]:>{col}.4f}" for k in keys)
        print(row)
    print("=" * (col * (len(keys) + 1)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    output = evaluate()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print_table(output["metrics"])
    print(f"\nTotal time : {time.time() - t0:.1f}s")
    print(f"Results    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
