# Assignment review

Reference: `2026-07-23-applied-ai-engineer-delta-chat-assignment.html`, candidate
sections 01–11 supplied by the user. The reference architecture and scaffold are
guides, not a requirement to add more frameworks. This implementation deliberately
uses native PDF + scanned PDF/OCR; DWG remains an explicit adapter stub.

## Acceptance criteria and evidence

| HTML requirement | Implementation and validation | Remaining limit |
| --- | --- | --- |
| At least two formats behind one interface | `FormatAdapter`, native and OCR adapters, routing tests, scanned/native pipeline test | Dense drawings still have serious OCR/layout errors |
| Canonical format seam | Shared document/page/line models with boxes, source and OCR confidence | Equal granularity does not guarantee equal line segmentation |
| Typed, located, confidence-scored delta | Two-pass deterministic alignment; added/removed/modified/moved; regression tests | Same-page matching only; confidence is heuristic |
| Human/machine report | Markdown + JSON; default real run found three changes | Not a CAD geometry or table-cell diff |
| Cited chat over both PIDs and report | BM25+ + Chroma + RRF + cross-encoder; source preference; exact citation validation tests | Six live generation cases are only a small smoke set |
| One-command ingest → report → chat | `uv run python main.py run --question "What changed on PSV-9066?"` | Validated locally with the configured Groq provider; still a trusted-local demo |
| Request traces, timing, tokens/cost | Shared request ID, shared JSON log, per-request JSONL trace, stage timings, retrieval hit counts, live provider token/cost validation | Cost remains an estimate based on configured prices |
| Runnable evaluation scorecard | `make eval`; delta, retrieval, generation scorecards; six QA cases covering PID A, PID B, and delta entries | Still a small single-pair reviewed benchmark |
| Failure reporting | `eval/RESULTS.md` records dense-OCR limits and explicit provider-error behavior | These limits are not fixed by passing unit tests |
| Secrets and environment template | `.env.example`, ignored `.env`/logs; tracked-key-pattern check found no matches | Not an exhaustive history/PII audit; users must review their own input documents |
| Samples and provenance | Three test pairings documented in `data/samples/README.md` | Only one is labelled; one is an unrelated negative pair |
| Bonus markup | PDF box annotations for changes located in Revision B; removed content is deliberately report-only | Not a pixel/geometry diff and does not visualize removals on Revision A |
| Simple runnable interface | CLI and optional single-process FastAPI | Trusted local demo; one active pair, no concurrent comparison support |

## Deliberate design choices

- Keep ingestion, canonical models, delta, chat, logging, and evaluation separate.
  Do not add service layers, agent frameworks, or a second UI.
- Use the user's question directly for retrieval and reranking. There is no
  query rewriting or expansion. A small deterministic preference puts change
  evidence or an explicitly requested revision first without filtering sources.
- Keep keyword search for identifiers and vector search for natural-language
  similarity; search all indexed sources uniformly and rerank a short candidate
  list with the existing cross-encoder.
- Use one final result limit and explicit, configurable quality thresholds.
- Keep factual answer generation as the only generative-LLM step. Embeddings and
  cross-encoder scoring remain local retrieval models.
- Preserve existing labels, describe their uncertain review provenance, and
  distinguish citation-source checks from claim-level factual grounding.

## Remaining validation limits

A human should independently review and expand labels across multiple revision
pairs. Dense scanned drawings require better OCR/layout work; this repository
does not claim production-quality comparison on them. The successful live Groq
run validates integration and grounded-answer controls, not broad answer quality.

Read [DEMO.md](DEMO.md) for the walkthrough and [eval/RESULTS.md](eval/RESULTS.md)
for measured results, including failures.
