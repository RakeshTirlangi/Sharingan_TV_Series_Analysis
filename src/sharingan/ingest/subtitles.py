"""Subtitle ingestion: ASS/SRT -> tidy, line-level dialogue table.

Compared with naive "skip the first 27 lines and split on commas" parsing this:
  * reads the ``[Events]`` ``Format:`` header so column positions are never assumed,
  * supports both ``.ass`` and ``.srt`` (two Naruto episodes ship as SRT),
  * strips ASS override tags (``{\\i1}``), HTML tags, and ``\\N``/``\\h`` escapes,
  * keeps timestamps, so later stages can reason about time and position,
  * flags opening/ending song lyrics, which otherwise dominate theme scores.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..core import get_logger
from .arcs import arc_for_episode

log = get_logger(__name__)

_EPISODE_RE = re.compile(r"Season\s*(?P<season>\d+)\s*-\s*(?P<episode>\d+)", re.I)
_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SRT_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def _ass_time(value: str) -> float:
    h, m, s = value.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def clean_text(text: str) -> str:
    text = _ASS_TAG_RE.sub("", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS_RE.sub(" ", text).strip()


def parse_ass(raw: str) -> list[tuple[float, float, str]]:
    rows, fields, in_events = [], None, False
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.startswith("Format:"):
            fields = [f.strip().lower() for f in line[7:].split(",")]
        elif line.startswith("Dialogue:") and fields:
            values = line[9:].split(",", maxsplit=len(fields) - 1)
            rec = dict(zip(fields, values))
            text = clean_text(rec.get("text", ""))
            if text:
                rows.append((_ass_time(rec["start"]), _ass_time(rec["end"]), text))
    return rows


def parse_srt(raw: str) -> list[tuple[float, float, str]]:
    rows = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        m = _SRT_TIME_RE.search(block)
        if not m:
            continue
        g = list(map(int, m.groups()))
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = clean_text(" ".join(block[m.end():].strip().splitlines()))
        if text:
            rows.append((start, end, text))
    return rows


def parse_subtitle_file(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = parse_srt(raw) if path.suffix.lower() == ".srt" else parse_ass(raw)
    m = _EPISODE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot infer season/episode from {path.name!r}")
    df = pd.DataFrame(rows, columns=["start", "end", "text"]).sort_values("start", kind="stable")
    df.insert(0, "season", int(m["season"]))
    df.insert(0, "episode", int(m["episode"]))
    df["line_idx"] = range(len(df))
    return df


def flag_song_lines(df: pd.DataFrame, min_episodes: int = 5, window_s: float = 150) -> pd.Series:
    """Mark OP/ED lyrics: verbatim repeats across episodes that sit in the head/tail window."""
    norm = df["text"].str.lower().str.replace(r"[^a-z' ]", "", regex=True).str.strip()
    spread = norm.groupby(norm).transform(lambda s: df.loc[s.index, "episode"].nunique())
    duration = df.groupby("episode")["end"].transform("max")
    in_window = (df["start"] <= window_s) | (df["start"] >= duration - window_s)
    multiword = norm.str.count(" ") >= 2
    return (spread >= min_episodes) & in_window & multiword


def load_dialogue(subtitles_dir: Path, *, min_episodes: int = 5, window_s: float = 150,
                  drop_songs: bool = True) -> pd.DataFrame:
    """Load every subtitle file into one line-level frame (one row per subtitle event)."""
    files = sorted(p for p in Path(subtitles_dir).iterdir() if p.suffix.lower() in {".ass", ".srt"})
    if not files:
        raise FileNotFoundError(f"No .ass/.srt subtitles found in {subtitles_dir}")
    df = pd.concat([parse_subtitle_file(p) for p in files], ignore_index=True)
    df["is_song"] = flag_song_lines(df, min_episodes, window_s)
    df["arc"] = df["episode"].map(arc_for_episode)
    log.info(
        f"Parsed {len(files)} files, {df['episode'].nunique()} episodes, {len(df):,} lines "
        f"({df['is_song'].mean():.1%} flagged as OP/ED lyrics)"
    )
    if drop_songs:
        df = df[~df["is_song"]].reset_index(drop=True)
    return df.sort_values(["episode", "line_idx"]).reset_index(drop=True)


def episode_scripts(dialogue: pd.DataFrame) -> pd.DataFrame:
    """Collapse line-level dialogue into one script per episode."""
    return (
        dialogue.groupby(["episode", "season", "arc"], sort=True)["text"]
        .agg(" ".join)
        .rename("script")
        .reset_index()
    )
