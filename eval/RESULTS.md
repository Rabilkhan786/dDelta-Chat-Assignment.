# Local validation — 2026-09-20

These are observed smoke-test results, not a claim of general document accuracy.
No ground-truth labels were changed during this cleanup.
Code checks use Ruff and pytest. The suite covers ingestion, query preparation,
hybrid retrieval, reranking, delta confidence/alignment, citations, API, logging,
metric arithmetic, and a simple native/OCR pipeline. One upstream
Starlette/httpx deprecation warning remains; it is not a test failure.

Final checks: **64 tests passed**, Ruff lint/format checks passed, and
`uv sync --locked` succeeded. These validate code behavior, not the unresolved
dense-scan or live-generation quality requirements below.

## Default native revision pair

Command: `uv run python main.py run --question "What changed on PSV 9066?"`

- 875 native lines in A, 874 in B.
- Three changes: one added, one removed, one modified; 872 unchanged.
- Compatibility score: 0.989.
- 2,624 indexed excerpts across both documents and the report.
- Retrieved `[delta_report | PID revision_b | page 1 | delta-95]`.
- One request ID followed ingestion, alignment, reporting, indexing, retrieval,
  reranking, and the visible provider failure.
- Generation status: `provider_error`, because `GROQ_API_KEY` was not configured.
  This is not a successful generated chat exchange.

`uv run python -m eval.run_eval` produced:

```json
{"delta": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted": 3, "expected": 3}}
{"retrieval": {"cases": 1, "recall_at_5": 1.0, "mrr": 1.0}}
{"generation": {"status": "not_run", "reason": "provider unavailable"}}
```

These scores concern only the existing three-change/one-question dataset. Its
historical human-review status has not been independently verified. They do not
establish answer correctness, groundedness, or OCR accuracy.

## Retrieval smoke checks with real local models

| Question | Observed result |
| --- | --- |
| What changed on PSV-9066? | `delta-95`, cross-encoder score approximately 3.034 |
| What changed on PSV 9066? | Same `delta-95`, score approximately 3.220 |
| Who won the World Cup? | No evidence after rejection threshold |

Scores are raw model outputs, not probabilities. Unknown identifier suffixes,
source constraints, keyword/vector fusion, and citation rejection also have
controlled unit tests; those are not additional real-document benchmark cases.

## Failures and negative cases

| Case | Observation | Decision |
| --- | --- | --- |
| Supplied different-system PDFs | Overlap 0.465; old 0.12 threshold wrongly accepted them | Threshold now 0.60, with a regression test; still only a heuristic warning |
| Dense scanned copy versus original native page | OCR recovered 131 lines versus 875 native; 853 reported changes despite the same content; overlap 0.386 | Treat as a failed comparison-quality stress case, not meaningful revisions |
| Dense scan retrieval | “What does revision A say about compressor?” returned no evidence at the default reranker threshold | False-negative retrieval case; threshold is not calibrated |
| Sparse-layout OCR experiment (`--psm 11`) | 835 lines and 447 matches, but 901 reported changes on the same-content pair | Do not adopt globally just for higher extraction count; default unchanged |
| Groq answer generation | Missing key, visible error and evidence references | Needs configured provider and human answer/citation review |

The dense scan still completed routing, OCR, canonical serialization, report,
markup, and real model indexing (1,861 excerpts). Successful execution is not
successful comparison quality. A separate simple scanned/native test fixture
correctly detects a single pressure change from 10 to 12 bar; it does not erase
the dense drawing failure.

Full raw traces remain local in ignored `logs/project.log` because they can
contain source text and prompts. Timing depends on cold model downloads and
concurrent work, so this run is not a latency benchmark.
