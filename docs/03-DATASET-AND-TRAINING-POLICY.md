# 03 — Dataset and Training Policy

## Goal

Build a small, high-quality, legally usable English-learning dataset before attempting larger-scale training. Data quality and provenance matter more than raw record count.

## Data categories

Potential categories:
- Vocabulary definitions and usage examples.
- Grammar explanations and minimal pairs.
- Error-correction examples with explanations.
- Short reading passages with comprehension questions.
- Writing feedback examples.
- CEFR-tagged exercises, only where level labels are justified.
- English-learning dialogues that remain educational and age-appropriate.

## Provenance requirements

Every source must record:
- source name and origin;
- URL or local acquisition reference where applicable;
- author/publisher if known;
- license or explicit permission basis;
- date acquired;
- transformations performed;
- reviewer and quality status;
- dataset version and split assignment.

Do not scrape or copy copyrighted textbooks, paid courses, dictionaries, or websites without permission or a valid license. Do not assume that publicly accessible means free to train on.

## Learner data and privacy

- Do not use KatLearn learner chats as training data by default.
- Never include passwords, tokens, account identifiers, contact details, or other secrets.
- If user-contributed examples are considered later, require a clear opt-in, explain intended use, minimize data, and provide a deletion process.
- Prefer synthetic examples authored for the project or sources with clear reuse permissions.
- Keep private data out of Git history.

## Data quality

Each record should be:
- relevant to a supported English-learning task;
- linguistically correct or explicitly marked as disputed/variant;
- understandable for its target learner level;
- free of duplicate or near-duplicate leakage;
- free of unsupported factual claims;
- safe and age-appropriate.

Quality labels may include `approved`, `needs_review`, and `rejected`. Only approved records enter training.

## Suggested record schema

```json
{
  "id": "vocab-000001",
  "task": "vocabulary",
  "level": "A2",
  "input": "What does 'comfortable' mean?",
  "target": "Comfortable means…",
  "source_id": "project-authored-001",
  "license": "project-authored",
  "review_status": "approved",
  "language": "en",
  "dataset_version": "0.1.0"
}
```

## Training sequence

1. Validate schema and provenance.
2. Remove invalid, duplicate, and contaminated records.
3. Split by source/document before tokenization to reduce leakage.
4. Fit tokenizer on training data only.
5. Train baseline model with recorded configuration and random seed.
6. Evaluate on validation data during development.
7. Use test data only for final comparisons.
8. Save checkpoint, tokenizer, config, data manifest hash, and metrics together.

## From-scratch versus fine-tuning

A model trained from random initialization is genuinely project-trained, but needs substantial diverse text and compute to develop broad language ability. On the known CPU-only development machine, use a tiny model to validate the pipeline and learn from experiments.

A future pretrained starting point would be a different strategy: it may be useful, but it must be labeled as a pretrained/fine-tuned model and its license, provenance, and restrictions must be reviewed. It must not be described as trained from scratch.

## Avoiding misleading metrics

Training loss alone does not demonstrate tutoring quality. Report:
- held-out task accuracy or rubric score;
- grammar and vocabulary correctness;
- helpfulness and clarity;
- unsupported-answer rate;
- scope false-accept rate;
- safety regression results;
- latency and memory usage.

Every result must identify the dataset and model versions used.
