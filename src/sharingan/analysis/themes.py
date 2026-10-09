"""Zero-shot theme analysis with a Natural Language Inference model.

Each theme label becomes a hypothesis ("This dialogue is about friendship.") and the
NLI model scores P(entailment) against every chunk of an episode's dialogue. Labels
are scored independently (multi-label), so themes do not compete for probability mass.

Engineering choices over a plain ``pipeline("zero-shot-classification")`` call:
  * token-budgeted chunks built from whole subtitle lines (no mid-sentence truncation),
  * all (chunk, hypothesis) pairs are length-sorted and batched -> minimal padding,
  * works with 2-way (entail/not) and 3-way (entail/neutral/contradict) NLI heads,
  * chunk-level scores are kept, so the most representative excerpt per theme can be shown,
  * resumable: progress is checkpointed every few episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.progress import track
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..core import get_logger, pick_device

log = get_logger(__name__)


@dataclass
class ThemeResult:
    episodes: pd.DataFrame   # one row per episode, one column per theme (mean P(entail))
    chunks: pd.DataFrame     # one row per chunk with its text and per-theme scores

    @property
    def labels(self) -> list[str]:
        meta = {"episode", "season", "arc", "n_chunks", "chunk_id", "text"}
        return [c for c in self.episodes.columns if c not in meta]


class ThemeAnalyzer:
    def __init__(self, model_name: str, labels: list[str], *,
                 hypothesis_template: str = "This dialogue is about {}.",
                 chunk_max_tokens: int = 320, batch_size: int = 32, device: str | None = None):
        self.labels = list(labels)
        self.hypotheses = [hypothesis_template.format(label) for label in self.labels]
        self.chunk_max_tokens = chunk_max_tokens
        self.batch_size = batch_size
        self.device = device or pick_device()

        log.info(f"Loading NLI model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.entail_idx, self.contra_idx = self._resolve_label_ids(self.model.config.label2id)

    @staticmethod
    def _resolve_label_ids(label2id: dict[str, int]) -> tuple[int, int]:
        norm = {k.lower(): v for k, v in label2id.items()}
        entail = next(v for k, v in norm.items() if k.startswith("entail"))
        contra = next((v for k, v in norm.items() if k.startswith(("contradict", "not_entail", "not entail"))), None)
        if contra is None:  # fall back to "the other" class of a binary head
            contra = next(v for v in norm.values() if v != entail)
        return entail, contra

    # ------------------------------------------------------------------ chunking
    def chunk_lines(self, lines: list[str]) -> list[str]:
        """Greedily pack consecutive subtitle lines into chunks of <= chunk_max_tokens."""
        lengths = [len(ids) for ids in self.tokenizer(lines, add_special_tokens=False)["input_ids"]]
        chunks, current, size = [], [], 0
        for line, n in zip(lines, lengths):
            if current and size + n > self.chunk_max_tokens:
                chunks.append(" ".join(current))
                current, size = [], 0
            current.append(line)
            size += n
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ------------------------------------------------------------------- scoring
    @torch.inference_mode()
    def score_chunks(self, chunks: list[str]) -> np.ndarray:
        """Return an ``[n_chunks, n_labels]`` matrix of P(entailment | chunk, hypothesis)."""
        pairs = [(c, h) for c in chunks for h in self.hypotheses]
        order = np.argsort([len(c) for c, _ in pairs])
        probs = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                truncation="only_first", max_length=self.chunk_max_tokens + 48,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits[:, [self.contra_idx, self.entail_idx]]
            probs[idx] = logits.float().softmax(-1)[:, 1].cpu().numpy()
        return probs.reshape(len(chunks), len(self.labels))

    def analyze(self, dialogue: pd.DataFrame, *, checkpoint: Path | None = None) -> ThemeResult:
        """Score every episode in a line-level dialogue frame (see ``ingest.load_dialogue``)."""
        done = pd.read_parquet(checkpoint) if checkpoint and checkpoint.exists() else None
        finished = set(done["episode"]) if done is not None else set()
        frames = [done] if done is not None else []
        if finished:
            log.info(f"Resuming theme analysis: {len(finished)} episodes already scored")

        groups = [(ep, g) for ep, g in dialogue.groupby("episode", sort=True) if ep not in finished]
        for i, (episode, group) in enumerate(track(groups, description="Scoring themes"), 1):
            chunks = self.chunk_lines(group["text"].tolist())
            scores = self.score_chunks(chunks)
            frame = pd.DataFrame(scores, columns=self.labels)
            frame.insert(0, "text", chunks)
            frame.insert(0, "chunk_id", range(len(chunks)))
            frame.insert(0, "arc", group["arc"].iat[0])
            frame.insert(0, "season", group["season"].iat[0])
            frame.insert(0, "episode", episode)
            frames.append(frame)
            if checkpoint and (i % 20 == 0 or i == len(groups)):
                pd.concat(frames, ignore_index=True).to_parquet(checkpoint, index=False)

        chunks_df = pd.concat(frames, ignore_index=True)
        episodes = (
            chunks_df.groupby(["episode", "season", "arc"], sort=True)
            .agg(n_chunks=("chunk_id", "size"), **{label: (label, "mean") for label in self.labels})
            .reset_index()
        )
        return ThemeResult(episodes=episodes, chunks=chunks_df)


# ------------------------------------------------------------------ summaries
def series_profile(episodes: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Overall theme strength, normalised so the strongest theme = 1."""
    score = episodes[labels].mean()
    return (
        pd.DataFrame({"theme": score.index, "score": score.values, "relative": score.values / score.max()})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def arc_profile(episodes: pd.DataFrame, labels: list[str], arc_order: list[str]) -> pd.DataFrame:
    """Per-arc theme lift: arc mean divided by series mean (>1 = more prominent than usual)."""
    by_arc = episodes.groupby("arc")[labels].mean()
    lift = by_arc / episodes[labels].mean()
    return lift.reindex([a for a in arc_order if a in lift.index])


def top_excerpts(chunks: pd.DataFrame, theme: str, k: int = 3) -> pd.DataFrame:
    return chunks.nlargest(k, theme)[["episode", "arc", theme, "text"]]
