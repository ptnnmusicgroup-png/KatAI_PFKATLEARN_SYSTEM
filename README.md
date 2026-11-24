# KatAI — Native English Learning AI for KatLearn

> **Project status:** Planning / Specification  
> **Owner:** PTNN Music Group  
> **Repository:** `KatAI_PFKATLEARN_SYSTEM`  
> **Primary product:** KatLearn  
> **Current principle:** Build an independently trained, English-learning-focused AI system. Do not disguise a third-party chatbot API as a native model.

## 1. Vision

KatAI is a task-specific AI system designed to support English learning inside KatLearn. It is not intended to be a general-purpose chatbot and must not answer unrelated school subjects or general knowledge requests as if they were supported features.

The long-term goal is to develop and train KatAI's own language model. The first milestones will establish the data pipeline, evaluation, safety and scope enforcement, then train increasingly capable models. A small model trained from scratch on a CPU is an engineering prototype—not a substitute for a production-scale language model.

## 2. Non-negotiable product requirements

1. **English-learning scope only.** Supported tasks include vocabulary, definitions, pronunciation guidance, grammar, sentence construction, reading, writing, listening practice, speaking practice, and English-learning study plans.
2. **Fixed out-of-scope response.** For requests outside English learning, return exactly:
   `Hiện tại, chúng tôi không hỗ trợ các môn học khác ngoài Tiếng Anh. Xin cảm ơn`
3. **Native-model direction.** The target model must be trained and owned by this project. Gemini or another hosted chatbot must not be a hidden generation fallback.
4. **Layered safety.** Input checks, task/scope classification, output checks, rate limits, and evaluation work alongside the model. A prompt or regex alone is not a safety guarantee.
5. **No arbitrary tool access.** The model does not receive shell, filesystem, network, account, or KatLearn administrative capabilities.
6. **Honest capability boundaries.** If the model is uncertain or a task is unsupported, it should say so rather than invent an answer.
7. **Privacy and data provenance.** Training data must have documented sources and permitted use. Do not train on private learner conversations by default.

## 3. System at a glance

```text
KatLearn UI
    |
KatLearn server / authenticated API
    |
KatAI service boundary
    |
Input normalization
    |
Safety pre-check -> English-learning scope gate
    |                         |
blocked / out of scope        allowed task
    |                         |
fixed response                task router
                              |
                       native KatAI model
                              |
                       output safety check
                              |
                       response + metadata
```

The scope and safety gates are independent of the generative model. The model cannot override them.

## 4. Planned repository structure

```text
KatAI_PFKATLEARN_SYSTEM/
├── README.md
├── docs/
│   ├── 00-VISION-AND-REQUIREMENTS.md
│   ├── 01-SYSTEM-ARCHITECTURE.md
│   ├── 02-ROADMAP-AND-MILESTONES.md
│   ├── 03-DATASET-AND-TRAINING-POLICY.md
│   ├── 04-SAFETY-AND-EVALUATION.md
│   └── 05-LOCAL-DEVELOPMENT-AND-DEPLOYMENT.md
├── kat_ai/                 # implementation, introduced after design review
├── datasets/               # metadata/manifests; no private data or secrets
├── experiments/            # reproducible experiment configs and reports
├── tests/
└── scripts/
```

## 5. Development phases

- **Phase 0 — Specification:** agree on scope, architecture, safety policy, data policy, and measurable acceptance criteria.
- **Phase 1 — Foundation:** create a reproducible Python environment, data schemas, test harness, and service interface.
- **Phase 2 — Safety and scope:** implement deterministic policy gates and a separately evaluated intent/scope classifier.
- **Phase 3 — Data pipeline:** curate permitted English-learning examples, split datasets without leakage, and document provenance.
- **Phase 4 — Native model prototype:** build tokenizer/model/training loop and train a small experimental model from initialized weights.
- **Phase 5 — Tutor quality:** evaluate and improve task-specific performance; add supervised instruction examples only when the data and compute justify it.
- **Phase 6 — KatLearn integration:** expose a narrow authenticated API and connect it to KatLearn without changing unrelated LMS features.
- **Phase 7 — Release readiness:** security, privacy, regression, load, and quality evaluation before any real-user rollout.

See the documents in `docs/` for the full plan.

## 6. Hardware reality

The currently known development machine is a Dell OptiPlex 7050 Micro with Intel Core i5-7500T, 16 GB RAM, and Intel HD 630 integrated graphics. It can support Python development, small classifiers, data preprocessing, and very small CPU training experiments. It is not a practical machine for pretraining a capable modern LLM. Initial model work must be sized accordingly; larger experiments may require rented or otherwise available GPU compute.

## 7. What this project is not

- Not a Gemini wrapper renamed as KatAI.
- Not a universal homework solver.
- Not a promise that a tiny CPU-trained model will match commercial assistants.
- Not a system that learns automatically from student chats without explicit governance.
- Not a replacement for teacher judgment or verified learning materials.

## 8. Current status

This repository is currently a planning repository. The documents define intended behavior and implementation direction; they do **not** mean a model has already been trained or that the system is production-ready.

**Next decision:** review and approve the specification before implementation begins.
