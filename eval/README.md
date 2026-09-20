# Evaluation: what the scorecard does and does not prove

Run the default comparison first, then `uv run python -m eval.run_eval`.
Use `--skip-generation` to evaluate delta and retrieval without calling a provider.
Each scorecard is emitted as JSON to the console and `logs/project.log`.

The existing ground-truth file is preserved. It contains three expected changes
and one QA case over the default synthetic revision pair, not all sample pairs.
Its note describes earlier re-keying after extraction changed to lines. This
cleanup has not independently verified the historical human-review claim.
Do not regenerate labels from predictions to improve scores.

## Definitions

| Metric | Meaning | Limitation |
| --- | --- | --- |
| Delta precision/recall/F1 | Exact match of predicted and labelled change hashes | Hashes include description, confidence and location; equivalent changes can mismatch after extraction changes |
| Retrieval Recall@K | Fraction of labelled relevant IDs retrieved in the first K hits | Only one question currently has a relevance label |
| MRR | Reciprocal rank of the first labelled relevant result | Does not judge unlabelled relevant results |
| Answer correctness | All expected keywords occur in generated text | Keyword proxy; a wrong sentence can contain the right word |
| Citation accuracy | Fraction of returned citations matching a labelled source fragment | Source precision, not claim-level entailment |
| Citation coverage | Fraction of expected source fragments cited | Broad fragments do not uniquely identify evidence |

Citation accuracy previously measured coverage. These are now separate metrics;
historical numbers under that name are not directly comparable. The answer
service also rejects absent or invented citations, but a valid source reference
does not prove a sentence is true.

## Human review before expanding the benchmark

1. Inspect both source PDFs and record the actual changes independently.
2. Review optional `--write-candidates` output against those observations. It is
   a proposal, not ground truth, and never replaces the dataset automatically.
3. Label questions for A, B, changes, comparisons, exact tags, OCR errors, and
   unsupported requests. Include precise relevant source and element IDs.
4. Write full expected answers and check each claim against cited evidence.
5. Freeze reviewed cases before comparing retrieval settings or thresholds.

No new human-labelled answers are claimed in this cleanup. Tests use explicit
synthetic examples to verify metric arithmetic and edge cases; those tests are
not additional human-reviewed drawing labels.

## Known limitations

- Dense OCR can split or misread labels, creating false deltas.
- A moved and substantially edited label can become an addition and a removal.
- Shared boilerplate can make unrelated drawings pass the compatibility check.
- Query preparation cannot resolve pronouns or infer unstated equipment tags.
- Reranker thresholds can reject valid evidence; tune precision and recall on
  a larger reviewed set, not just the supplied positive question.
- No live generated-answer score is valid until a configured provider run succeeds.
