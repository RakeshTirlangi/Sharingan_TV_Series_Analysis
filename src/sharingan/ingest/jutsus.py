"""Jutsu dataset ingestion.

Accepts both the rich schema produced by our Scrapy spider (``classification`` as a
list) and the legacy flat schema (``jutsu_type`` as a comma separated string).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..core import get_logger

log = get_logger(__name__)

# Order matters: a technique tagged "Ninjutsu, Genjutsu" is a genjutsu in practice,
# and anything chakra-moulded beyond pure body skill is ninjutsu.
LABEL_PRIORITY = ("Genjutsu", "Ninjutsu", "Taijutsu")
_WS_RE = re.compile(r"\s+")


def primary_label(classes: list[str]) -> str | None:
    joined = " ".join(classes)
    for label in LABEL_PRIORITY:
        if label.lower() in joined.lower():
            return label
    return None


def _classes(row: pd.Series) -> list[str]:
    value = row.get("classification")
    if isinstance(value, (list, tuple)) and len(value):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = row.get("jutsu_type") or ""
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def load_jutsus(primary: Path, fallback: Path | None = None) -> pd.DataFrame:
    """Load scraped jutsus -> ``name, classes, label, description, text``.

    Uses the freshly scraped file when present, the bundled reference snapshot otherwise.
    """
    path = primary if primary.exists() else fallback
    if path is None or not path.exists():
        raise FileNotFoundError(f"No jutsu data at {primary} (run `sharingan scrape`)")
    log.info(f"Loading jutsus from {path}")
    df = pd.read_json(path, lines=True)
    df = df.rename(columns={"jutsu_name": "name", "jutsu_description": "description"})
    df["classes"] = df.apply(_classes, axis=1)
    df["label"] = df["classes"].map(primary_label)
    df["description"] = df["description"].fillna("").map(lambda s: _WS_RE.sub(" ", s).strip())
    df["text"] = df["name"].str.strip() + ". " + df["description"]
    df = df[df["label"].notna() & (df["description"].str.len() > 20)]
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    return df[["name", "classes", "label", "description", "text"]]
