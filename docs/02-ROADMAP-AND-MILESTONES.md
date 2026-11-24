# 02 — Roadmap and Milestones

This roadmap is ordered by dependency. Dates are intentionally omitted until scope, compute, and available time are confirmed.

## Phase 0 — Plan approval

**Deliverables**
- Approved vision and supported-task list.
- Fixed out-of-scope response.
- Architecture and API boundary.
- Dataset and privacy policy.
- Safety test plan and acceptance thresholds.
- Hardware/compute decision.

**Exit criteria:** no unresolved product-critical ambiguity.

## Phase 1 — Engineering foundation

**Deliverables**
- Python project layout and pinned dependencies.
- Configuration management with no committed secrets.
- Data schemas and dataset manifest format.
- Unit-test and evaluation harness.
- CLI commands for validation and reproducible runs.
- API interface stub with health endpoint.

**Exit criteria:** clean setup on the local Windows machine; tests run deterministically.

## Phase 2 — Scope and safety foundation

**Deliverables**
- Exact out-of-scope response implementation.
- English-learning task taxonomy.
- Rule-based baseline plus separately evaluated intent classifier.
- Input/output policy checks.
- Adversarial and regression test set, including Vietnamese and English phrasing, misspellings, mixed requests, and prompt-injection attempts.

**Exit criteria:** documented precision/recall and false-accept rate on a held-out test set; all critical policy tests pass. A perfect guarantee is not claimed.

## Phase 3 — Dataset pipeline

**Deliverables**
- Data collection checklist and source register.
- Normalization, deduplication, filtering, and quality-review scripts.
- Train/validation/test split.
- Dataset cards and version hashes.
- Small initial dataset focused on a narrow task.

**Exit criteria:** every training record has provenance and permitted-use status; no private learner conversations are included without a separately approved consent process.

## Phase 4 — Native model prototype

**Deliverables**
- Tokenizer implementation or documented selected tokenizer.
- Small decoder-only transformer or another justified architecture.
- Training loop, checkpoint save/load, and inference script.
- Loss curves and reproducible experiment records.
- Baseline comparisons against simple non-neural approaches.

**Exit criteria:** model trains from initialized project-owned weights, can be reloaded, and produces measurable results on held-out English-learning examples. This is a research prototype, not production tutor quality.

## Phase 5 — Task-specific tutoring quality

**Deliverables**
- Curated instruction-style examples for selected tasks.
- Fine-tuning experiments only if compute and dataset quality support them.
- Evaluation by task, learner level, correctness, clarity, hallucination, and safety.
- Error analysis and model-card limitations.

**Exit criteria:** selected tasks meet pre-agreed quality thresholds. Unsupported tasks remain disabled.

## Phase 6 — KatLearn integration

**Deliverables**
- Authenticated KatAI service.
- KatLearn adapter and feature flag.
- Timeouts, rate limits, request caps, and health checks.
- No Gemini fallback in native mode.
- Regression tests proving unrelated KatLearn features are untouched.

**Exit criteria:** staging integration works; failures are graceful; no secrets or learner content leak into logs.

## Phase 7 — Controlled release

**Deliverables**
- Security and privacy review.
- Load and latency tests.
- Monitoring and rollback procedure.
- Versioned model release and evaluation report.
- Feedback process that does not automatically turn chats into training data.

**Exit criteria:** owner approves release based on evidence, not merely successful generation.

## Priority order

1. Policy and scope correctness.
2. Data quality and provenance.
3. Reproducibility.
4. Narrow task quality.
5. Latency and deployment.
6. Broader capabilities.

Do not expand to more tasks just because the model can produce fluent-looking text.
