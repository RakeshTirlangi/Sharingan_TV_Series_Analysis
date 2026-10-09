"""Typed, validated configuration loaded from ``config.yaml`` + environment overrides.

Environment variables of the form ``SHARINGAN__<SECTION>__<KEY>=value`` override the
YAML file, e.g. ``SHARINGAN__THEMES__BATCH_SIZE=64``. Values are parsed as YAML so
lists and numbers work (``SHARINGAN__THEMES__LABELS='[hope, love]'``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
ENV_PREFIX = "SHARINGAN__"


class Paths(BaseModel):
    subtitles_dir: Path
    transcript_csv: Path
    jutsus_jsonl: Path
    jutsus_fallback_jsonl: Path
    processed_dir: Path
    outputs_dir: Path

    def resolved(self) -> "Paths":
        return Paths(**{k: (v if v.is_absolute() else PROJECT_ROOT / v) for k, v in self})


class SubtitleCfg(BaseModel):
    song_min_episodes: int = 5
    song_window_seconds: float = 150


class ThemeCfg(BaseModel):
    model_name: str
    hypothesis_template: str = "This dialogue is about {}."
    labels: list[str]
    chunk_max_tokens: int = 320
    batch_size: int = 32


class CharacterCfg(BaseModel):
    spacy_model: str = "en_core_web_trf"
    window_lines: int = 10
    decay_tau: float = 4.0
    min_mentions: int = 8
    top_edges: int = 250


class JutsuCfg(BaseModel):
    model_name: str
    labels: list[str]
    max_length: int = 256
    epochs: int = 4
    learning_rate: float = 5e-5
    batch_size: int = 16
    seed: int = 42
    hub_model_id: str | None = None


class PersonaCfg(BaseModel):
    base_model: str
    embedding_model: str
    adapter_dir: Path
    retrieved_examples: int = 4
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9


class Settings(BaseModel):
    paths: Paths
    subtitles: SubtitleCfg = Field(default_factory=SubtitleCfg)
    themes: ThemeCfg
    characters: CharacterCfg = Field(default_factory=CharacterCfg)
    jutsu: JutsuCfg
    persona: PersonaCfg

    # Convenience locations under outputs/
    def out(self, *parts: str) -> Path:
        p = self.paths.outputs_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _apply_env_overrides(raw: dict) -> dict:
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        section, _, field = key[len(ENV_PREFIX):].lower().partition("__")
        if section in raw and field:
            raw[section][field] = yaml.safe_load(value)
    return raw


@lru_cache(maxsize=4)
def get_settings(config_path: str | os.PathLike | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)  # HF_TOKEN, SHARINGAN__* overrides
    path = Path(config_path or os.environ.get("SHARINGAN_CONFIG", DEFAULT_CONFIG))
    raw = _apply_env_overrides(yaml.safe_load(path.read_text()))
    settings = Settings(**raw)
    settings.paths = settings.paths.resolved()
    if not settings.persona.adapter_dir.is_absolute():
        settings.persona.adapter_dir = PROJECT_ROOT / settings.persona.adapter_dir
    return settings
