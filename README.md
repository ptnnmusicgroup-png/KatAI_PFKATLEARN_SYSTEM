# KatAI Native v0.3.0

KatAI is a local-first bilingual (English/Vietnamese) English-learning AI project for KatLearn. The native decoder-only Transformer is randomly initialized and trained by this project; there is no Gemini or hosted-chatbot fallback.

## Implemented in this code baseline

- **Native model foundation:** decoder-only Transformer, SentencePiece tokenizer pipeline, training loop, checkpoint/resume, inference engine, and CLI.
- **Safety baseline v2:** Unicode normalization, invisible-format-character removal, prompt-injection pattern checks, conservative English-learning scope gate, and tool payload limits.
- **Skill Router:** routes quiz-generation requests to KatWork and progress requests to KatGames.
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
