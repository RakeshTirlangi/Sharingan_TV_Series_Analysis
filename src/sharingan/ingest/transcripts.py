"""Speaker-attributed transcript ingestion (``name,line`` CSV)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_STAGE_DIRECTION_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")


def clean_line(text: str) -> str:
    text = _STAGE_DIRECTION_RE.sub(" ", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS_RE.sub(" ", text).strip(" -")


def load_transcript(path: Path, min_words: int = 1) -> pd.DataFrame:
    """Return ``turn, speaker, line, prev_speaker, prev_line`` with stage directions removed."""
    df = pd.read_csv(path).dropna(subset=["name", "line"])
    df = df.rename(columns={"name": "speaker"})
    df["speaker"] = df["speaker"].str.strip()
    df["line"] = df["line"].map(clean_line)
    df = df[df["line"].str.split().str.len() >= min_words].reset_index(drop=True)
    df["turn"] = range(len(df))
    df["prev_speaker"] = df["speaker"].shift(1)
    df["prev_line"] = df["line"].shift(1)
    return df[["turn", "speaker", "line", "prev_speaker", "prev_line"]]
