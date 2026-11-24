# 01 — System Architecture

## Design principle

Separate **policy enforcement** from **language generation**. The model is a component inside KatAI; it is not the authority that decides whether it may answer.

## Request lifecycle

1. KatLearn authenticates the learner and sends a bounded request to the KatAI API.
2. API validates schema, request size, rate limit, and supported locale.
3. Input safety checks detect disallowed or high-risk requests.
4. Scope classifier and deterministic policy decide whether the task is English learning.
5. Out-of-scope requests receive the exact fixed Vietnamese response without calling the model.
6. In-scope requests are mapped to a supported English-learning task.
7. Native model generates a candidate answer.
8. Output checks assess safety, scope drift, format, and obvious malformed output.
9. API returns the answer with model/version metadata; it does not expose internal prompts, policy internals, or private training data.

## Logical modules

### API boundary
- Authentication and authorization.
- Rate limiting and request-size limits.
- Strict request/response schemas.
- Timeouts and graceful failure.
- No model access to KatLearn database credentials.

### Safety layer
- Input risk checks and policy classifier.
- Scope decision and mixed-intent handling.
- Output screening.
- Regression test corpus.
- Human review process for policy changes.

### Scope router
Initial task labels:
- `vocabulary`
- `translation`
- `grammar`
- `writing`
- `reading`
- `pronunciation`
- `speaking_practice`
- `english_study_planning`
- `out_of_scope`
- `ambiguous`

A classifier is not sufficient by itself. Use deterministic rules for explicit out-of-scope cases, a separately evaluated classifier for natural-language intent, and a conservative decision policy. Track false-accept and false-reject rates.

### Native model
The model is responsible for language-learning responses only. Its initial size and capabilities must be determined by experiments. A small model may first support constrained tasks such as vocabulary explanation or grammar feedback rather than open-ended tutoring.

### Data and training
- Dataset manifests with source, license/permission, language, task, quality review, and version.
- Train/validation/test separation by source or document where possible.
- Deduplication and contamination checks.
- Reproducible tokenizer and model configuration.
- Checkpoint metadata and evaluation reports.

### Integration adapter
KatLearn should call a narrow internal API. The adapter translates KatLearn's request format into KatAI's schema and maps the response back. It must not alter unrelated LMS behavior, coins, synchronization, or learning-pack logic.

## Suggested service contract

Request:
```json
{
  "request_id": "opaque-id",
  "learner_level": "A2",
  "task": "grammar",
  "message": "Why do we use the present perfect here?"
}
```

Response:
```json
{
  "request_id": "opaque-id",
  "status": "ok",
  "task": "grammar",
  "answer": "…",
  "model_id": "katai-native-prototype",
  "model_version": "0.1.0"
}
```

For out-of-scope requests, the `answer` field contains the exact fixed response in the product requirements. The API may include a machine-readable `status` such as `out_of_scope`, but the user-visible answer must remain exact.

## Failure behavior

- Model unavailable: return a concise service-unavailable message; do not silently call Gemini.
- Scope decision uncertain: do not let the generator answer an unrelated request. Use a conservative clarification only for plausibly English-learning requests.
- Output check fails: retry at most a bounded number of times with constrained generation; if still failing, return a safe failure response.
- Dataset or checkpoint missing: fail closed; do not load arbitrary files or download untrusted artifacts automatically.

## Security boundaries

- The model process has no shell or arbitrary tool execution.
- Model artifacts are loaded only from configured, versioned paths.
- User input is treated as untrusted data, never as system policy.
- No secrets in prompts, logs, datasets, or checkpoints.
- Keep training, evaluation, and production inference as separate commands/processes.
