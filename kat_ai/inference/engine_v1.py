"""Inference engine for K.AI 1 checkpoints."""
import torch
import sentencepiece as spm

from kat_ai.model.transformer_v1 import KatAI1Config, KatAI1Transformer


class KatAI1InferenceEngine:
    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("architecture_version") != "1.0.0":
            raise ValueError("Unsupported checkpoint: expected K.AI 1 architecture version 1.0.0")
        self.config = KatAI1Config(**checkpoint["config"])
        self.model = KatAI1Transformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        if self.tokenizer.get_piece_size() != self.config.vocab_size:
            raise ValueError("Tokenizer vocabulary mismatch")
        self.step = int(checkpoint.get("step", 0))
        self.model_id = checkpoint.get("model_id", "katlearn-kai-1")

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.9):
        ids = self.tokenizer.encode(prompt, out_type=int)
        if not ids:
            bos = self.tokenizer.bos_id()
            if bos < 0:
                raise ValueError("Tokenizer does not define a BOS token")
            ids = [bos]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, eos_id=self.tokenizer.eos_id()
        )
        return self.tokenizer.decode(output[0, len(ids):].tolist()).strip()
