"""Inference wrapper for a trained jutsu classifier (local directory or Hub repo id)."""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ...core import pick_device


class JutsuClassifier:
    def __init__(self, model_path: str, max_length: int = 256):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.max_length = max_length

    @torch.inference_mode()
    def predict_proba(self, texts: str | list[str]) -> list[dict[str, float]]:
        texts = [texts] if isinstance(texts, str) else texts
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                             return_tensors="pt").to(self.device)
        probs = self.model(**enc).logits.float().softmax(-1).cpu().tolist()
        return [dict(zip(self.labels, p)) for p in probs]

    def predict(self, texts: str | list[str]) -> list[str]:
        return [max(p, key=p.get) for p in self.predict_proba(texts)]


@lru_cache(maxsize=2)
def load_classifier(model_path: str) -> JutsuClassifier:
    return JutsuClassifier(model_path)
