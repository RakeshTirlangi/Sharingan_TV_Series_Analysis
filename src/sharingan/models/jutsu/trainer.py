"""Fine-tune a transformer encoder for jutsu-type classification.

* class-weighted cross-entropy (via ``Trainer(compute_loss_func=...)``, no subclassing),
* model selection on validation **macro-F1** (accuracy hides the rare Genjutsu class),
* early stopping + best-checkpoint restore, then a single final pass on the held-out test set.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ...core import get_logger, pick_device, seed_everything
from .data import class_weights, evaluate

log = get_logger(__name__)


class JutsuTrainer:
    def __init__(self, model_name: str, labels: list[str], output_dir: Path, *, max_length: int = 256,
                 epochs: int = 4, learning_rate: float = 5e-5, batch_size: int = 16, seed: int = 42):
        self.model_name, self.labels, self.output_dir = model_name, list(labels), Path(output_dir)
        self.max_length, self.epochs, self.lr, self.batch_size, self.seed = max_length, epochs, learning_rate, batch_size, seed
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_dir = self.output_dir / "model"

    def _dataset(self, df) -> Dataset:
        ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["label"].map(self.label2id).tolist()})
        return ds.map(lambda b: self.tokenizer(b["text"], truncation=True, max_length=self.max_length),
                      batched=True, remove_columns=["text"])

    def fit(self, splits: dict, *, save: bool = True) -> dict:
        """Train with early stopping on ``val``, evaluate once on ``test``; ``save`` keeps the model."""
        seed_everything(self.seed)
        weights = torch.tensor(class_weights(splits["train"]["label"], self.labels), dtype=torch.float)
        log.info(f"Fine-tuning {self.model_name} on {self.device}; class weights "
                 f"{dict(zip(self.labels, [round(w, 2) for w in weights.tolist()]))}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)), label2id=self.label2id,
        )

        def weighted_loss(outputs, labels, num_items_in_batch=None):
            logits = outputs["logits"].float()
            return torch.nn.functional.cross_entropy(logits, labels, weight=weights.to(logits.device))

        def macro_f1(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"macro_f1": f1_score(labels, preds, average="macro"),
                    "accuracy": float((preds == labels).mean())}

        args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.epochs,
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            weight_decay=0.01,
            warmup_steps=0.1,  # float = ratio of total steps
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=20,
            bf16=self.device == "cuda" and torch.cuda.is_bf16_supported(),
            report_to="none",
            seed=self.seed,
        )
        trainer = Trainer(
            model=model, args=args,
            train_dataset=self._dataset(splits["train"]), eval_dataset=self._dataset(splits["val"]),
            processing_class=self.tokenizer, data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_loss_func=weighted_loss, compute_metrics=macro_f1,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        t0 = time.perf_counter()
        trainer.train()
        train_seconds = time.perf_counter() - t0

        test_logits = trainer.predict(self._dataset(splits["test"])).predictions
        preds = [self.labels[i] for i in np.argmax(test_logits, axis=-1)]
        report = evaluate(splits["test"]["label"].tolist(), preds, self.labels)
        report.update(model=self.model_name, train_seconds=round(train_seconds, 1),
                      best_val_macro_f1=round(float(trainer.state.best_metric or 0), 4),
                      epochs_run=round(float(trainer.state.epoch or 0), 2))

        log.info(f"Transformer: test macro-F1 {report['macro_f1']:.3f} accuracy {report['accuracy']:.3f}")
        if save:
            trainer.save_model(str(self.model_dir))
            self.tokenizer.save_pretrained(str(self.model_dir))
            (self.model_dir / "training_report.json").write_text(json.dumps(report, indent=2))
            shutil.rmtree(self.output_dir / "checkpoints", ignore_errors=True)  # optimizer state, ~1.7 GB
            log.info(f"Saved fine-tuned classifier to {self.model_dir}")

        del trainer, model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return report

    def push(self, repo_id: str) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model.push_to_hub(repo_id)
        self.tokenizer.push_to_hub(repo_id)
        log.info(f"Pushed jutsu classifier to https://huggingface.co/{repo_id}")
