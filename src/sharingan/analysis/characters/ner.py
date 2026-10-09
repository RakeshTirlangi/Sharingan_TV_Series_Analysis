"""Character-mention extraction: spaCy NER + cast gazetteer + alias resolution."""

from __future__ import annotations

import re

import pandas as pd
import spacy
from rich.progress import track
from wordfreq import zipf_frequency

from ...core import get_logger
from .cast import HONORIFIC_RE, POSSESSIVE_RE, STOPLIST, TITLE_PREFIX_RE, alias_table, ruler_patterns

log = get_logger(__name__)

# Model-discovered names (not in the cast list) must look like proper nouns, not
# common English words: "Gennai" (Zipf 1.0) passes, "Transform" (4.0) does not.
MAX_ZIPF = 3.0
_NAME_SHAPE_RE = re.compile(r"^[A-Z][a-zA-Zōūāēī'-]+( [A-Z][a-zA-Zōūāēī'-]+){0,2}$")


def normalise_name(surface: str, aliases: dict[str, str]) -> str | None:
    """Map a raw PERSON span to a canonical character name (or ``None`` to discard)."""
    raw = surface.strip(" .,!?…\"'")
    truncated = raw.endswith(("-", "–", "—"))         # cut-off speech: "Sakur--"
    name = POSSESSIVE_RE.sub("", raw.rstrip("-–— "))
    if truncated:
        hits = {canon for alias, canon in aliases.items() if len(name) >= 3 and alias.startswith(name.lower())}
        return hits.pop() if len(hits) == 1 else None
    name = HONORIFIC_RE.sub("", name).strip()
    if name.lower() in aliases:
        return aliases[name.lower()]
    name = TITLE_PREFIX_RE.sub("", name).strip()
    if name.lower() in aliases:
        return aliases[name.lower()]
    first = name.split(" ")[0] if name else ""
    if first.lower() in aliases:                       # "Naruto Uzumaki-kun" -> Naruto
        return aliases[first.lower()]
    if name.isupper():
        name = name.title()
    if not name or name.lower() in STOPLIST or len(name) < 4 or not _NAME_SHAPE_RE.match(name):
        return None
    if any(zipf_frequency(tok, "en") >= MAX_ZIPF for tok in name.lower().split()):
        return None
    return name


def clean_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    """Drop stoplisted names from cached mentions, so stoplist edits apply without re-running NER."""
    return mentions[~mentions["character"].str.lower().isin(STOPLIST)].reset_index(drop=True)


class MentionExtractor:
    def __init__(self, model: str = "en_core_web_trf", batch_size: int = 256):
        keep = {"transformer", "tok2vec", "ner"}
        self.nlp = spacy.load(model)
        self.nlp.select_pipes(enable=[p for p in self.nlp.pipe_names if p in keep])
        ruler = self.nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        ruler.add_patterns(ruler_patterns())
        self.aliases = alias_table()
        self.batch_size = batch_size
        log.info(f"Mention extractor ready ({model}; pipes={self.nlp.pipe_names})")

    def extract(self, dialogue: pd.DataFrame) -> pd.DataFrame:
        """One row per (line, character) mention; a character counts once per line."""
        texts = dialogue["text"].tolist()
        docs = self.nlp.pipe(texts, batch_size=self.batch_size)
        rows = []
        meta = dialogue[["episode", "arc", "line_idx", "start"]].itertuples(index=False)
        for (episode, arc, line_idx, start), doc in track(zip(meta, docs), total=len(texts),
                                                          description="Extracting characters"):
            seen = set()
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                canonical = ent.ent_id_ or normalise_name(ent.text, self.aliases)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    rows.append((episode, arc, line_idx, start, canonical, ent.text,
                                 "gazetteer" if ent.ent_id_ else "model"))
        return pd.DataFrame(rows, columns=["episode", "arc", "line_idx", "start",
                                           "character", "surface", "source"])
