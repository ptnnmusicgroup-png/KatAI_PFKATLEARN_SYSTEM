# KatLearn K.AI 1 — Model Card (Development)

## Identity
- Product name: KatLearn K.AI 1
- Internal model ID: `katlearn-kai-1`
- Architecture version: `1.0.0`
- Development branch: `katai-v1-development`
- Training origin: random initialization; no pretrained checkpoint or hosted model fallback.

## Architecture changes from KatAI Native v0.3
- Rotary positional embeddings (RoPE).
- RMSNorm pre-normalization.
- SwiGLU feed-forward blocks.
- PyTorch scaled dot-product causal attention.
- Tied input/output token embeddings.
- Separate versioned inference engine and training pipeline.

## Initial compute profile
The initial configuration remains intentionally small: 128 hidden dimensions, 4 attention heads, 4 layers, 256-token context, and an 8k tokenizer vocabulary. This is a pipeline-validation configuration, not a claim of competitive language-model quality. Larger configurations must be benchmarked against available RAM, CPU throughput, and training data before selection.

## Training requirements
- Use only original, public-domain, licensed, or explicitly authorized English-learning data.
- Keep source/license provenance and dataset version.
- Split by source/document into train and validation sets; reserve a separate test set for final evaluation.
- Train tokenizer only on the training split.
- Save the best checkpoint based on held-out validation loss; retain the last checkpoint for resuming.
- Report data hash, model configuration, random seed, training steps, and evaluation results.

## Release gate
K.AI 1 is not a production release until:
1. CI and model tests pass.
2. A tokenizer and checkpoint are trained from an authorized corpus.
3. Held-out evaluation covers grammar, vocabulary, translation, reading, writing feedback, bilingual consistency, and unsupported-answer rate.
4. Safety/scope regression tests pass.
5. CPU RAM, latency, and generation throughput are measured on the target machine.
6. KatLearn integration is tested against the real PHÒNG HỌC AI request/response contract.

## Known limitations
- No trained tokenizer or model checkpoint is included in this development branch.
- Small from-scratch models may produce weak or nonsensical language until adequately trained.
- Training on a CPU-only 16 GB machine is suitable for experiments, not large-scale pretraining.
- Safety and scope filters remain defense-in-depth prototypes and do not guarantee correctness.
