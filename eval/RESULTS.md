# Validation notes

The repository uses GitHub Actions CI to run the locked environment sync, Ruff
lint/format checks, and pytest on a clean Linux runner with Tesseract installed.
Use the latest CI result as the source of truth for code checks.

The measurements below are historical smoke observations from the supplied
samples. They are useful failure evidence, not a current benchmark and not a
claim of general document accuracy.

## Primary synthetic revision pair

The supplied primary pair previously produced:

- 875 extracted native lines in Revision A and 874 in Revision B.
- Three meaningful changes: one modified tag, one removed callout, and one
  added note.
- 872 unchanged aligned elements.
- Revision compatibility score 0.989.

The current implementation keeps the unchanged count in the report summary but
writes and indexes only the three meaningful changes as `delta-1`,
`delta-2`, and `delta-3`.

The reviewed change labels remain:

1. `9066A` changed to `9066C`.
2. `MECHANICAL INTERLOCK` was removed.
3. `NOTE 24: NEW BLOWDOWN VALVE ADDED PER REV B.` was added.

The current QA dataset contains six retrieval/chat cases covering Revision A,
Revision B, and each of the three delta-report entries. Rerun
`uv run python -m eval.run_eval` after `uv run python main.py run` to obtain
current metrics.

## Latest local validation

The configured Groq provider was exercised successfully on 21 September 2026.
On the supplied primary pair and the small checked-in labels, the run produced:

- Code checks: `106` tests passed with `95.69%` branch-aware coverage.
- Delta precision/recall/F1: `1.0 / 1.0 / 1.0` over three labelled changes.
- Retrieval Recall@5: `1.0` over six questions.
- Retrieval MRR: `1.0` over six questions.
- Answer keyword correctness: `1.0` over six generated answers.
- Citation accuracy: `1.0` over six generated answers.
- Citation coverage: `1.0` over six generated answers.

All six generation cases completed with status `answered`. Token counts and
estimated cost were recorded in the local request traces. These are regression
smoke results for a tiny dataset, not a general accuracy claim.

## Known failure and stress cases

| Case | Observation | Decision |
| --- | --- | --- |
| Different-system PDFs | Historical token overlap was 0.465 | Keep the compatibility warning; it is a heuristic, not proof of document identity |
| Dense scanned copy vs native original | Historical OCR extraction produced many false deltas on the dense drawing | Keep as a candid OCR/layout failure case, not an accuracy benchmark |
| Moved and substantially edited label | May become remove + add because second-pass alignment requires high text similarity | Document limitation rather than hiding it |
| Groq answer generation without a key | Provider cannot run | Report generation as not run/provider error; never count it as a successful eval |

A small synthetic scanned/native test fixture is included in pytest and checks
that the common pipeline can detect a simple pressure change. That controlled
test does not erase the dense-drawing OCR limitation.

## What the results do not prove

- Three labelled changes are not a broad delta benchmark.
- Six QA cases are not a broad retrieval benchmark.
- Keyword answer coverage is not semantic factual correctness.
- Citation-source matching is not claim-level entailment.
- Compatibility token overlap is not document-identity verification.
- Successful OCR execution is not proof of OCR quality.

Runtime traces and prompts remain local under ignored `logs/` because they can
contain source-document text.
