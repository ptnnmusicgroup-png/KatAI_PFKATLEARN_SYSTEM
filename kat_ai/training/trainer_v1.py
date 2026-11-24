"""Training pipeline for the K.AI 1 architecture."""
from pathlib import Path
import json
import math
import random
import hashlib

import torch
from torch.utils.data import Dataset, DataLoader

from kat_ai.data.jsonl import read_jsonl_texts
from kat_ai.model.transformer_v1 import KatAI1Config, KatAI1Transformer


class TokenBlockDataset(Dataset):
    def __init__(self, texts, processor, block_size):
        ids = []
        for text in texts:
            ids.extend(processor.encode(text, out_type=int))
            ids.append(processor.eos_id())
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        if self.data.numel() < block_size + 1:
            raise ValueError(
                f"Need at least {block_size + 1} tokens; got {self.data.numel()}"
            )

    def __len__(self):
        return max(1, (self.data.numel() - 1) // self.block_size)

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = self.data[start:start + self.block_size + 1]
        if chunk.numel() < self.block_size + 1:
            missing = self.block_size + 1 - chunk.numel()
            chunk = torch.cat((chunk, self.data[:missing]))
        return chunk[:-1], chunk[1:]


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _evaluate(model, loader, device, max_batches):
    model.eval()
    losses = []
    with torch.inference_mode():
        for index, (x, y) in enumerate(loader):
            if index >= max_batches:
                break
            _, loss = model(x.to(device), y.to(device))
            losses.append(float(loss))
    model.train()
    if not losses:
        raise ValueError("Validation dataset produced no batches")
    return sum(losses) / len(losses)


def train_kai1(data_path, tokenizer_path, output_path, steps=2000, batch_size=4,
               learning_rate=3e-4, seed=42, d_model=128, n_heads=4, n_layers=4,
               max_seq_len=256, validation_interval=100, warmup_steps=100,
               validation_batches=20, resume=False, num_workers=0):
    import sentencepiece as spm

    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if validation_interval < 0 or warmup_steps < 0 or validation_batches <= 0:
        raise ValueError("Invalid validation or warmup settings")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    cfg = KatAI1Config(
        vocab_size=processor.get_piece_size(), d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_seq_len=max_seq_len
    )
    cfg.validate()

    texts = read_jsonl_texts(data_path)
    if len(texts) < 2:
        raise ValueError("K.AI 1 training requires at least two records for a held-out validation split")
    random.shuffle(texts)
    split = min(len(texts) - 1, max(1, int(len(texts) * 0.95)))
    train_texts, val_texts = texts[:split], texts[split:]

    train_ds = TokenBlockDataset(train_texts, processor, max_seq_len)
    val_ds = TokenBlockDataset(val_texts, processor, max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KatAI1Transformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    step = 0
    best_val = float("inf")
    output_path = Path(output_path)
    best_path = output_path.with_name(output_path.stem + ".best" + output_path.suffix)

    if resume:
        checkpoint = torch.load(output_path, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != "1.0.0":
            raise ValueError("Checkpoint is not a K.AI 1 checkpoint")
        if checkpoint["config"] != cfg.to_dict():
            raise ValueError("Checkpoint config mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_validation_loss", float("inf")))

    def lr_at(current):
        if warmup_steps and current < warmup_steps:
            return learning_rate * (current + 1) / warmup_steps
        progress = min(max((current - warmup_steps) /
                           max(1, steps - warmup_steps), 0.0), 1.0)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    history_path = output_path.with_suffix(".metrics.jsonl")
    model.train()
    while step < steps:
        for x, y in train_dl:
            if step >= steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            record = {"step": step, "train_loss": float(loss.detach()),
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            if step == 1 or step % 25 == 0:
                print(json.dumps(record))

            if validation_interval and step % validation_interval == 0:
                val_loss = _evaluate(model, val_dl, device, validation_batches)
                record["validation_loss"] = val_loss
                print(json.dumps(record))
                payload = {
                    "architecture_version": "1.0.0",
                    "model_id": "katlearn-kai-1",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "config": cfg.to_dict(),
                    "tokenizer_path": str(tokenizer_path),
                    "training_seed": seed,
                    "best_validation_loss": min(best_val, val_loss),
                    "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                }
                if val_loss < best_val:
                    best_val = val_loss
                    payload["best_validation_loss"] = best_val
                    _atomic_save(payload, best_path)

            if step == 1 or step % 25 == 0 or (validation_interval and step % validation_interval == 0):
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            _atomic_save({
                "architecture_version": "1.0.0",
                "model_id": "katlearn-kai-1",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "config": cfg.to_dict(),
                "tokenizer_path": str(tokenizer_path),
                "training_seed": seed,
                "best_validation_loss": best_val,
                "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            }, output_path)

    return {"checkpoint": str(output_path), "best_checkpoint": str(best_path),
            "steps": step, "best_validation_loss": best_val}
