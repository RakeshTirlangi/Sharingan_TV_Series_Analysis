"""LoRA supervised fine-tuning of a chat model on one character's lines (TRL ``SFTTrainer``).

Examples are *conversational prompt-completion* records:
    prompt     = [system: character card, user: the line said to the character]
    completion = [assistant: what the character actually replied]
so the loss is computed on the character's reply only (``completion_only_loss``).
On a CUDA GPU with bitsandbytes this becomes QLoRA (4-bit NF4 base weights).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from ...core import get_logger, pick_device, seed_everything
from .cards import CARDS
from .engine import load_causal_lm

log = get_logger(__name__)


def build_sft_dataset(voice: pd.DataFrame, character: str, *, min_words: int = 3) -> Dataset:
    card = CARDS[character]
    rows = voice[(voice["character"] == character) & (voice["context"].str.len() > 0)
                 & (voice["line"].str.split().str.len() >= min_words)]
    system = card.system_prompt(voice=[], scenes=[])
    records = [
        {"prompt": [{"role": "system", "content": system}, {"role": "user", "content": ctx}],
         "completion": [{"role": "assistant", "content": line}]}
        for ctx, line in zip(rows["context"], rows["line"])
    ]
    if not records:
        raise ValueError(f"No training exchanges for {character}")
    return Dataset.from_list(records)


def train_persona_adapter(base_model: str, dataset: Dataset, output_dir: Path, *, epochs: int = 3,
                          learning_rate: float = 2e-4, rank: int = 16, batch_size: int = 4,
                          grad_accum: int = 2, max_length: int = 512, seed: int = 42) -> dict:
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(seed)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(base_model, device)
    model.config.use_cache = False

    split = dataset.train_test_split(test_size=0.1, seed=seed) if len(dataset) >= 40 else None
    config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # float = ratio of total steps
        max_grad_norm=0.3,
        max_length=max_length,
        completion_only_loss=True,
        logging_steps=5,
        eval_strategy="epoch" if split else "no",
        save_strategy="no",
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        report_to="none",
        seed=seed,
    )
    peft_config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                             target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(
        model=model, args=config, processing_class=tokenizer, peft_config=peft_config,
        train_dataset=split["train"] if split else dataset,
        eval_dataset=split["test"] if split else None,
    )
    # Held-out loss of the untouched base model: the bar the adapter has to beat.
    base_eval = trainer.evaluate()["eval_loss"] if split else None
    result = trainer.train()
    metrics = {"train_loss": round(result.training_loss, 4), "examples": len(dataset),
               "base_model": base_model, "epochs": epochs, "lora_rank": rank}
    if split:
        metrics["base_eval_loss"] = round(base_eval, 4)
        metrics["eval_loss"] = round(trainer.evaluate()["eval_loss"], 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "sft_report.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"Persona adapter saved to {output_dir}: {metrics}")

    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return metrics
