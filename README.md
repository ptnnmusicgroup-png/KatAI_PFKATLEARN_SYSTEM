# KatAI Native v0.3.0

KatAI is a local-first bilingual (English/Vietnamese) English-learning AI project for KatLearn. The native decoder-only Transformer is randomly initialized and trained by this project; there is no Gemini or hosted-chatbot fallback.

## Implemented in this code baseline

- **Native model foundation:** decoder-only Transformer, SentencePiece tokenizer pipeline, training loop, checkpoint/resume, inference engine, and CLI.
- **Safety baseline v2:** Unicode normalization, invisible-format-character removal, prompt-injection pattern checks, conservative English-learning scope gate, and tool payload limits.
- **Skill Router + permission allowlist:** routes quiz-generation requests to KatWork and progress requests to KatGames; skill actions use default-deny permissions.
- **KatWork:** runs inside KatLearn's existing **PHÒNG HỌC AI** chat flow. It returns a structured multiple-choice quiz pack and has a direct `POST /v1/katwork/quiz` endpoint. Starter bank: Present Perfect (10 distinct questions), Simple Past (5), Articles (3). Unsupported topics fail closed until a native generator is connected.
- **KatGames:** read-only progress summarizer. It has no method for changing scores, rewards, game state, or KatCoin. Live KatLearn data adapter is not configured.
- **Automated tests:** safety, scope, routing, quiz validation, API, and read-only progress tests.

## Run locally

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -e .
python -m pytest -q
uvicorn kat_ai.api.main:app --host 127.0.0.1 --port 8000
```

Train a tokenizer and model using an authorized JSONL corpus:

```bash
katai train-tokenizer --input datasets/train.jsonl --output artifacts/tokenizer --vocab-size 8000
katai train --data datasets/train.jsonl --tokenizer artifacts/tokenizer/tokenizer.model --output checkpoints/katai.pt
```

Configure `KATAI_CHECKPOINT`, `KATAI_TOKENIZER`, and `KATAI_API_KEY` before connecting a client. The service should remain bound to localhost during development.

## KatWork request example

```http
POST /v1/chat
Content-Type: application/json

{"message":"Tạo 10 câu trắc nghiệm về Present Perfect"}
```

The response contains `skill: "KatWork"` and a structured `tool_result` pack for the KatLearn adapter. KatLearn remains responsible for rendering, collecting answers, and scoring.

## Honest limitations

- No trained checkpoint or tokenizer is bundled. Free-form model chat returns HTTP 503 until a model is trained and configured.
- KatWork's starter bank is intentionally limited; it does not invent questions for unsupported topics.
- KatGames currently summarizes supplied progress data only. Real user progress must be fetched through a confirmed, authenticated server-side KatLearn adapter; client-provided user IDs or scores must never be trusted.
- Safety checks are defense-in-depth prototypes, not a guarantee. Public deployment still requires authentication enforcement, rate limiting, threat modeling, security review, and integration tests.
- KatLearn UI integration is not yet verified against the actual quiz schema.

## Release state

**v0.3.0 code prototype.** Automated tests in this repository must pass before release. Do not treat this version as production-ready.


## K.AI 1 development

The `katai-v1-development` branch contains the next model architecture under the product name **KatLearn K.AI 1**. It is a development candidate, not yet a trained or production-ready release.

### K.AI 1 architecture
- RoPE positional encoding, RMSNorm, SwiGLU, and PyTorch scaled dot-product causal attention.
- Independent versioned training and inference modules.
- Separate last and best-validation checkpoints.
- API runtime selection with `KATAI_MODEL_VERSION=1`.
- CLI commands: `katai train-kai1` and `katai chat-kai1`.

Example training command:

```powershell
katai train-kai1 --data datasets/train.jsonl --tokenizer artifacts/tokenizer/tokenizer.model --output checkpoints/kai1.pt --steps 2000
```

Example local inference:

```powershell
katai chat-kai1 --checkpoint checkpoints/kai1.best.pt --tokenizer artifacts/tokenizer/tokenizer.model
```

To select K.AI 1 in the API, configure `KATAI_MODEL_VERSION=1`, `KATAI1_CHECKPOINT`, and `KATAI1_TOKENIZER`. The service returns 503 if the required trained checkpoint/tokenizer are missing; it never falls back to Gemini.

See [K.AI 1 Model Card](docs/KAI-1-MODEL-CARD.md) for architecture, data requirements, evaluation, and release gates.

**Important:** This branch contains model code and tests, not trained model weights. Do not call it a final K.AI 1 release until the release gates in the model card have been met.
