# 05 — Local Development and Deployment Plan

## Known development hardware

- Computer: Dell OptiPlex 7050 Micro
- CPU: Intel Core i5-7500T
- RAM: 16 GB
- GPU: Intel HD 630 integrated graphics
- OS: Windows 11 Pro

This is sufficient for project development, preprocessing, tests, small intent classifiers, and tiny CPU model experiments. It is not suitable for training a capable modern LLM from scratch.

## Development environment

Preferred initial stack:
- Python 3.x (select and pin an exact supported version during Phase 1).
- PyTorch CPU build for initial experiments.
- Standard Python tooling for linting, formatting, tests, and configuration.
- Git for source, configuration, and experiment metadata.
- JSONL for simple, inspectable dataset records.

Avoid adding large dependencies until there is a concrete need. Do not put model weights or large datasets into Git; use versioned artifact storage or local ignored directories.

## Proposed local commands

The exact command names will be finalized when the implementation exists. Intended interface:

```powershell
# Create and activate an isolated environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install -r requirements.txt

# Validate dataset
python -m kat_ai.data.validate --manifest datasets/manifest.json

# Run tests
python -m pytest

# Train a small experiment
python -m kat_ai.training.train --config experiments/cpu_tiny.yaml

# Evaluate a checkpoint
python -m kat_ai.evaluation.run --checkpoint artifacts/model

# Start local inference API
python -m kat_ai.api
```

These are planned interfaces, not executable commands until the modules are implemented.

## CPU training strategy

Start with a small intent classifier and tiny language-model experiment. Keep batch size, sequence length, vocabulary size, and model width conservative; monitor RAM, CPU temperature, and training time. Save checkpoints frequently.

Do not begin with multi-billion-parameter targets. Do not promise that CPU training will produce a fluent or competitive tutor. If experiments show that the model is not learning useful language behavior, stop scaling blindly and reassess data, architecture, and compute.

## Deployment topology

Recommended initial topology:
- KatLearn server authenticates the learner.
- KatAI runs as a separate local or hosted service process.
- KatLearn calls only the narrow KatAI API.
- Training is never run inside the production request process.
- Model artifacts are immutable and versioned per release.
- No Gemini fallback is configured.

For an early prototype, KatAI may run on the same machine as a development instance of KatLearn. Production hosting requires separate resource and security review.

## Secrets and configuration

- Keep API keys, signing secrets, and database credentials out of source control.
- Use environment variables or the deployment platform's secret manager.
- Commit only an example environment file with placeholder values.
- Do not log raw learner messages by default.
- Restrict model artifact paths and uploaded file sizes.

## Integration constraints

When integrating with KatLearn:
- add a small adapter rather than moving LMS code into this repository;
- preserve existing authentication and authorization checks;
- do not change KatCoin logic, coin synchronization, learning packs, or unrelated UI;
- add a feature flag and rollback path;
- test timeout, unavailable model, out-of-scope, and malformed response behavior.
