# Offline Ticket Support Service

A small Python service that classifies support tickets, retrieves local help-center articles, and produces validated, cited reply drafts. It uses only the two supplied JSON input files. No packages, model key, network access, or downloads are required.

## Run and validate

Requires Python 3.10 or later. From this directory:

```bash
python main.py
python validate.py
```

The first command writes `results.json` and `debug_report.json`. The second validates the saved files, reruns the pipeline twice to check reproducibility, and runs the unit tests. Both commands return a nonzero exit code on failure. Paths default to the directory containing the scripts, so the commands also work when invoked by absolute path from another working directory.

Optional input/output directories and retrieval count:

```bash
python main.py --data-dir ./data --output-dir ./output --top-k 2
python validate.py --data-dir ./data --output-dir ./output --top-k 2
```

Use the same options for both commands. You can also run tests alone with `python -m unittest discover -s tests -v`.

## Pipeline and component boundaries

```text
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT
          -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
```

| Stage | Implementation |
| --- | --- |
| Load | `service/pipeline.py`: UTF-8/BOM-aware JSON loading, required types, unique IDs, nonempty KB |
| Index and retrieve | `service/retrieval.py`: word TF-IDF vectors and cosine similarity, ticket message as query |
| Classify | `service/classification.py`: conservative English rules, explicit trading-advice priority, conflict detection |
| Generate | `service/generation.py`: offline extractive generator with cited article bodies |
| Validate | `service/validation.py`: strict schema, refusal, exact-source grounding, confidence and handoff checks |
| Write | `service/pipeline.py`: validated JSON, atomic replacement of each artifact |

`Classifier`, `Retriever`, and `Generator` protocols in `service/domain.py` allow independent component replacement through `Pipeline(classifier=..., retriever=..., generator=...)`. Generation receives retrieved snippets, not access to a separate knowledge store. The default offline path is always available. A hosted LLM adapter is not included; a paraphrasing adapter would need a grounding validator that supports its evidence format. Until then, unrecognized generated prose is replaced with a conservative fallback.

## Retrieval and grounded generation

The implementation uses logarithmic term frequency and smoothed inverse document frequency: `tf = 1 + log(count)`, `idf = 1 + log((1 + N) / (1 + df))`. Both query and article vectors are normalized. A small explicit alias map handles forms such as `deposited/deposits` and `sign-in/login`. There is no intent-dependent retrieval boost. Equal scores sort by article ID; no random state, timestamps, or remote calls enter the output.

Retrieval returns the top 1–3 IDs, bounded by KB size. It can return zero-score candidates when no relevant article exists. Generation only quotes a candidate with cosine score at least 0.12 and a title aligned with the predicted intent. Zero-score candidates are recorded for transparency, but never used as evidence. `debug_report.json` distinguishes retrieved candidates from quoted articles.

Replies combine fixed framing with exact article-body quotations labeled `[article_id]`. They answer the ticket category with the supplied help-center guidance rather than hardcoded final answers per ticket. This makes factual content traceable and avoids invented account facts, processing windows, or guarantees. The supplied KB does not specify the actual card processing window or lock cooldown duration, so the service does not invent either. Withdrawal replies list possible causes and request a secure human review; they do not claim which cause applies to the customer's account. Trading requests receive an explicit refusal.

## Confidence, safety, and escalation

Confidence is a heuristic, **not a calibrated probability**:

```text
retrieval_quality = min(1, best_usable_cosine_score / 0.45)
confidence = min(classifier_certainty, retrieval_quality)
```

Single matched intent rules use certainty 0.95; coherent lock/password recovery uses 0.80; conflicting rules use 0.55; unknown English requests use 0.25. Answers without usable grounding have confidence capped at 0.20 (normally zero because retrieval quality is zero). Non-English requests are flagged and capped at 0.20. The generic English fallback explicitly hands off to support.

Human escalation occurs for confidence below 0.55, insufficient grounding, unsupported generation, conflicting intents, unsupported language, or a withdrawal review. Trading advice is refused automatically; it does not need a human handoff when the refusal is sufficiently grounded and confident.

Flags include `advice_request`, `insufficient_grounding`, `policy_sensitive`, `low_confidence`, `account_specific_request`, `conflicting_intents`, `unsupported_language`, and `unsupported_claim`. The debug report contains predicted intent, matched signals, candidate titles and scores, quoted source IDs, confidence components, and the reasons for the handoff decision.

## Validation and tests

Validation checks required artifacts, valid JSON, one result per input ticket, exact output fields and types, allowed intents, 1–3 unique known article IDs, finite confidence in [0, 1], explicit trading refusal, grounding evidence, and escalation rules. Saved artifacts must equal two fresh deterministic pipeline runs. The unit tests exercise the sample deposit/T4 refusal behaviors, paraphrases, replacement IDs, unknown queries, conflicts, language fallback, prompt injection, unsafe generator replacement, schema rejection, empty or invalid input, retrieval ordering, and artifact tampering.

## Scope and limitations

This is a bounded batch service, invoked with Python; an HTTP server is outside this challenge's required interface. The English rule classifier and lexical retrieval are intentionally small and can miss uncommon paraphrases or implicit trading requests. Unrecognized requests are handed off. Exact quotation proves provenance from the supplied KB, not the truth or completeness of the KB. The KB is treated as trusted local content. No account lookup, transaction verification, translation model, model training, or production authentication is included. Atomic file replacement protects each individual JSON file; the two output files are not a single transaction, and validation detects a stale or mismatched pair.

The inputs can be replaced with the same schemas. Classification never uses ticket IDs, and generation never stores full final replies keyed by ID.
