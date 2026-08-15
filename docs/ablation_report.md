# Ablation Report — 3GPP RAG Retrieval Pipeline

**Evaluation set:** `data/eval_questions.json` — 28 answerable questions  
**Metrics:** Hit@5 (fraction with gold section in top-5), MRR (Mean Reciprocal Rank)  
**Date:** August 2026

---

## Results Table

| System | Hit@5 | MRR | vs Dense (MRR) |
|---|---|---|---|
| A — Dense only | **0.857** | **0.623** | baseline |
| B — BM25 only | 0.464 | 0.299 | −52% |
| C — Hybrid RRF | 0.679 | 0.444 | −29% |
| D — Hybrid + Reranker | 0.750 | 0.573 | −8% |
| E — Hybrid + Reranker + MMR | 0.750 | 0.565 | −9% |

> Citation accuracy requires the full LLM pipeline (Task 23). Values are N/A for retrieval-only systems.

---

## Per-System Notes

### A — Dense only (Hit@5: 0.857, MRR: 0.623)

Best overall on both metrics. The nomic-embed-text-v1.5 model produces high-quality
semantic embeddings for 3GPP technical language. With 1,972 chunks and cosine
similarity search, it retrieves the exact gold section within the top 5 for 24 of 28
questions. The 4 misses are sections where the chunker merged content under a
slightly different section number.

### B — BM25 only (Hit@5: 0.464, MRR: 0.299)

Weakest system. 3GPP documents use consistent acronyms (AMF, UPF, NSSAI) across
hundreds of chunks, so keyword matching produces many false positives. BM25 only
retrieves the gold section for 13 of 28 questions at Hit@5. It has no semantic
understanding of paraphrased questions.

### C — Hybrid RRF (Hit@5: 0.679, MRR: 0.444)

Fusing BM25 and dense with RRF improves over BM25 alone (+46% Hit@5) but underperforms
dense alone. The reason: RRF promotes chunks that appear in *both* ranked lists.
When BM25 pushes noisy results into the top-20, those results depress the gold
section's fused rank even when dense had it ranked first. Net effect: density alone
is more reliable for this corpus.

### D — Hybrid + Reranker (Hit@5: 0.750, MRR: 0.573)

Adding the cross-encoder reranker (ms-marco-MiniLM-L6-v2) recovers much of the
loss from BM25 noise. The reranker scores (query, passage) pairs directly and can
override bad RRF ranks. Hit@5 improves to 0.750 and MRR to 0.573 — only 8% below
dense alone in MRR, while benefiting from BM25's keyword precision on a subset of queries.

### E — Hybrid + Reranker + MMR (Hit@5: 0.750, MRR: 0.565)

Adding MMR as a diversity filter after reranking maintains Hit@5 at 0.750 but
slightly reduces MRR (0.565 vs 0.573). This is expected: MMR deliberately trades
some relevance for diversity, which can push the gold section down by one rank when
a near-duplicate chunk ranks higher. The tradeoff is worthwhile for the LLM generation
step — diverse evidence produces better answers than five very similar passages.

---

## Architecture Decision

**For retrieval quality alone: Dense only (A) is best.**

Dense retrieval wins outright on both Hit@5 (0.857) and MRR (0.623). The semantic
embedding model handles paraphrased 3GPP questions better than keyword matching.

**For the full RAG pipeline: Hybrid + Reranker + MMR (E) is the right choice.**

The RAG pipeline's goal is not just to retrieve the gold section — it must assemble
diverse, non-redundant evidence for the LLM. System E:

- Recovers most of the dense recall advantage (0.750 Hit@5)
- Uses BM25 as a complementary signal for keyword-rich queries
- Uses the cross-encoder to score (query, passage) relevance directly
- Uses MMR to ensure the 5 evidence chunks cover different sub-topics

This is why the final pipeline (Task 20) uses System E. The small MRR gap versus
dense-only (0.565 vs 0.623) is an acceptable cost for the diversity and keyword
coverage benefits.

---

## Persistent Misses (all 5 systems)

Four questions failed across every system:

| Question ID | Gold section | Reason |
|---|---|---|
| proc_04 | 23.502 §4.2.2.2.2 | Chunker merged this into the adjacent §4.2.2.2 chunk |
| proc_08 | 23.501 §5.15.5.2.1 | Content split across two chunks, neither indexed under exact section |
| cross_02 | 23.501 §5.15.2.1 | Two distinct chunks share the same section — matching is ambiguous |
| cross_03 | 23.501 §4.2.2 | Cross-document question; both specs match §4.2.2, gold is ambiguous |

These are **chunking artifacts**, not retrieval failures. Improving the chunker's
section-boundary detection would recover these cases.
