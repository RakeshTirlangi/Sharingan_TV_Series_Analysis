"""Retrieval memory for character personas.

Two dense indices share one sentence-embedding model:

* **voice** - (context, reply) exchanges spoken by the character: gold lines from the
  speaker-attributed transcript plus weakly attributed subtitle lines (card ``signature``).
  Retrieved by similarity to the user's message -> style anchors in the prompt.
* **scenes** - overlapping windows of subtitle dialogue from all 220 episodes.
  Retrieved by similarity (boosted when the character is named) -> factual grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import get_logger
from .cards import CARDS

log = get_logger(__name__)

SCENE_LINES, SCENE_STRIDE = 8, 4


def build_voice_table(dialogue: pd.DataFrame, transcript: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if transcript is not None:
        for card in CARDS.values():
            mine = transcript[(transcript["speaker"] == card.name) & (transcript["line"].str.split().str.len() >= 4)]
            rows += [(card.name, ctx or "", line, "transcript")
                     for ctx, line in zip(mine["prev_line"].fillna(""), mine["line"])]
    prev = dialogue.groupby("episode")["text"].shift(1).fillna("")
    for card in CARDS.values():
        if not card.signature:
            continue
        hit = dialogue["text"].str.contains(card.signature, flags=re.I, regex=True)
        rows += [(card.name, ctx, line, "subtitle-signature")
                 for ctx, line in zip(prev[hit], dialogue.loc[hit, "text"])]
    voice = pd.DataFrame(rows, columns=["character", "context", "line", "source"])
    return voice.drop_duplicates(subset=["character", "line"]).reset_index(drop=True)


def build_scene_table(dialogue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (episode, arc), group in dialogue.groupby(["episode", "arc"], sort=True):
        lines = group["text"].tolist()
        for start in range(0, max(len(lines) - SCENE_LINES, 0) + 1, SCENE_STRIDE):
            rows.append((episode, arc, " ".join(lines[start:start + SCENE_LINES])))
    return pd.DataFrame(rows, columns=["episode", "arc", "text"])


@dataclass
class PersonaMemory:
    voice: pd.DataFrame
    scenes: pd.DataFrame
    voice_emb: np.ndarray
    scene_emb: np.ndarray
    encoder: object

    @classmethod
    def load_or_build(cls, cache_dir: Path, embedding_model: str, dialogue_fn, transcript_fn,
                      *, refresh: bool = False) -> "PersonaMemory":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model)
        cache_dir.mkdir(parents=True, exist_ok=True)
        files = [cache_dir / n for n in ("voice.parquet", "scenes.parquet", "embeddings.npz")]
        if all(f.exists() for f in files) and not refresh:
            emb = np.load(files[2])
            return cls(pd.read_parquet(files[0]), pd.read_parquet(files[1]), emb["voice"], emb["scenes"], encoder)

        dialogue = dialogue_fn()
        voice = build_voice_table(dialogue, transcript_fn())
        scenes = build_scene_table(dialogue)
        log.info(f"Embedding {len(voice):,} voice exchanges and {len(scenes):,} scene windows")
        encode = lambda texts: encoder.encode(texts, batch_size=128, normalize_embeddings=True,  # noqa: E731
                                              show_progress_bar=False).astype(np.float32)
        voice_emb = encode((voice["context"] + " " + voice["line"]).str.strip().tolist())
        scene_emb = encode(scenes["text"].tolist())
        voice.to_parquet(files[0], index=False)
        scenes.to_parquet(files[1], index=False)
        np.savez_compressed(files[2], voice=voice_emb, scenes=scene_emb)
        return cls(voice, scenes, voice_emb, scene_emb, encoder)

    def _query(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def recall_exchanges(self, character: str, query: str, k: int = 4) -> list[tuple[str, str]]:
        """Most similar (context, reply) exchanges spoken by ``character``."""
        mask = (self.voice["character"] == character).to_numpy()
        if not mask.any() or k <= 0:
            return []
        idx = np.flatnonzero(mask)
        sims = self.voice_emb[idx] @ self._query(query)
        best = idx[np.argsort(-sims)[:k]]
        return list(zip(self.voice.loc[best, "context"], self.voice.loc[best, "line"]))

    def recall_voice(self, character: str, query: str, k: int = 4) -> list[str]:
        return [line for _, line in self.recall_exchanges(character, query, k)]

    def recall_scenes(self, character: str, query: str, k: int = 3, name_boost: float = 0.08) -> list[str]:
        if k <= 0:
            return []
        sims = self.scene_emb @ self._query(f"{character}: {query}")
        sims = sims + name_boost * self.scenes["text"].str.contains(character.split()[0], regex=False).to_numpy()
        best = np.argsort(-sims)[:k]
        return [f"(Episode {r.episode}, {r.arc}) {r.text}" for r in self.scenes.iloc[best].itertuples()]
