"""Stage orchestration shared by the CLI and the web app.

Each ``run_*`` function is idempotent: it reuses persisted outputs unless
``refresh=True``, so the web UI can open instantly on precomputed results.
"""

from __future__ import annotations

import json

import pandas as pd

from .core import Settings, cached_frame, get_logger, get_settings
from .ingest import load_dialogue

log = get_logger(__name__)


def dialogue(settings: Settings | None = None, *, refresh: bool = False) -> pd.DataFrame:
    s = settings or get_settings()
    return cached_frame(
        s.paths.processed_dir / "dialogue.parquet",
        lambda: load_dialogue(s.paths.subtitles_dir, min_episodes=s.subtitles.song_min_episodes,
                              window_s=s.subtitles.song_window_seconds),
        refresh=refresh,
    )


# ----------------------------------------------------------------------- themes
def run_themes(settings: Settings | None = None, *, refresh: bool = False,
               labels: list[str] | None = None, episodes: list[int] | None = None):
    from .analysis.themes import ThemeAnalyzer, ThemeResult

    s = settings or get_settings()
    labels = labels or s.themes.labels
    ep_path, chunk_path = s.out("themes", "episodes.parquet"), s.out("themes", "chunks.parquet")
    custom = labels != s.themes.labels or episodes is not None
    if not custom and not refresh and ep_path.exists() and chunk_path.exists():
        return ThemeResult(pd.read_parquet(ep_path), pd.read_parquet(chunk_path))

    df = dialogue(s)
    if episodes is not None:
        df = df[df["episode"].isin(episodes)]
    t = s.themes
    analyzer = ThemeAnalyzer(t.model_name, labels, hypothesis_template=t.hypothesis_template,
                             chunk_max_tokens=t.chunk_max_tokens, batch_size=t.batch_size)
    checkpoint = None if custom else s.out("themes", "_checkpoint.parquet")
    if refresh and checkpoint and checkpoint.exists():
        checkpoint.unlink()
    result = analyzer.analyze(df, checkpoint=checkpoint)
    if not custom:
        result.episodes.to_parquet(ep_path, index=False)
        result.chunks.to_parquet(chunk_path, index=False)
        checkpoint.unlink(missing_ok=True)
        log.info(f"Theme scores written to {ep_path.parent}")
    return result


# ------------------------------------------------------------------- characters
def run_characters(settings: Settings | None = None, *, refresh: bool = False):
    from .analysis.characters import CharacterNetwork, MentionExtractor, clean_mentions

    s = settings or get_settings()
    c = s.characters
    mentions = cached_frame(
        s.out("characters", "mentions.parquet"),
        lambda: MentionExtractor(c.spacy_model).extract(dialogue(s)),
        refresh=refresh,
    )
    network = CharacterNetwork.from_mentions(
        clean_mentions(mentions), window=c.window_lines, tau=c.decay_tau, min_mentions=c.min_mentions
    )
    network.edges.to_parquet(s.out("characters", "edges.parquet"), index=False)
    network.metrics.to_parquet(s.out("characters", "metrics.parquet"), index=False)
    html_path = s.out("characters", "network.html")
    network.to_html(html_path, top_edges=c.top_edges)
    log.info(f"Character network: {network.graph.number_of_nodes()} characters, "
             f"{network.graph.number_of_edges()} relationships -> {html_path}")
    return network


# ------------------------------------------------------------------------ jutsu
def run_jutsu_training(settings: Settings | None = None, *, model_name: str | None = None,
                       epochs: int | None = None, push: bool = False) -> dict:
    from .ingest import load_jutsus
    from .models.jutsu import JutsuTrainer, make_splits, run_baselines

    s = settings or get_settings()
    j = s.jutsu
    data = load_jutsus(s.paths.jutsus_jsonl, s.paths.jutsus_fallback_jsonl)
    splits = make_splits(data, seed=j.seed)
    report_dir = s.out("models", "jutsu", "reports", "x").parent

    baselines = run_baselines(splits, embedding_model=s.persona.embedding_model)
    trainer = JutsuTrainer(
        model_name=model_name or j.model_name, labels=j.labels, output_dir=s.out("models", "jutsu", "x").parent,
        max_length=j.max_length, epochs=epochs or j.epochs, learning_rate=j.learning_rate,
        batch_size=j.batch_size, seed=j.seed,
    )
    transformer = trainer.fit(splits)
    if push and j.hub_model_id:
        trainer.push(j.hub_model_id)

    report = {"data": {k: int(len(v)) for k, v in splits.items()},
              "label_counts": data["label"].value_counts().to_dict(),
              "baselines": baselines, "transformer": transformer}
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    log.info(f"Jutsu classifier report -> {report_dir / 'metrics.json'}")
    return report


def run_jutsu_crossval(settings: Settings | None = None, *, folds: int = 5, transformer: bool = True,
                       epochs: int | None = None) -> dict:
    from .ingest import load_jutsus
    from .models.jutsu import cross_validate, save_report

    s = settings or get_settings()
    j = s.jutsu
    data = load_jutsus(s.paths.jutsus_jsonl, s.paths.jutsus_fallback_jsonl)
    result = cross_validate(
        data, j.labels, folds=folds, seed=j.seed, embedding_model=s.persona.embedding_model,
        transformer=dict(model_name=j.model_name, max_length=j.max_length, epochs=epochs or j.epochs,
                         learning_rate=j.learning_rate, batch_size=j.batch_size) if transformer else None,
        work_dir=s.out("models", "jutsu", "cv", "x").parent,
    )
    path = save_report(result, s.out("models", "jutsu", "reports", "crossval.json"))
    log.info(f"Cross-validation report -> {path}")
    return result


# ---------------------------------------------------------------------- persona
def persona_memory(settings: Settings | None = None, *, refresh: bool = False):
    from .ingest import load_transcript
    from .models.persona import PersonaMemory

    s = settings or get_settings()
    return PersonaMemory.load_or_build(
        s.out("persona", "memory", "x").parent, s.persona.embedding_model,
        dialogue_fn=lambda: dialogue(s),
        transcript_fn=lambda: load_transcript(s.paths.transcript_csv) if s.paths.transcript_csv.exists() else None,
        refresh=refresh,
    )


_ENGINE = None


def persona_engine(settings: Settings | None = None):
    """Process-wide singleton: loading an LLM is the expensive part of chatting."""
    global _ENGINE
    if _ENGINE is None:
        from .models.persona import PersonaEngine

        s = settings or get_settings()
        p = s.persona
        _ENGINE = PersonaEngine(p.base_model, persona_memory(s), adapter_root=p.adapter_dir,
                                retrieved_examples=p.retrieved_examples, max_new_tokens=p.max_new_tokens,
                                temperature=p.temperature, top_p=p.top_p)
    return _ENGINE


def run_persona_training(character: str = "Naruto", settings: Settings | None = None, *,
                         base_model: str | None = None, epochs: int = 3) -> dict:
    from .models.persona.sft import build_sft_dataset, train_persona_adapter

    s = settings or get_settings()
    memory = persona_memory(s)
    dataset = build_sft_dataset(memory.voice, character)
    log.info(f"Fine-tuning {character} persona on {len(dataset)} exchanges")
    return train_persona_adapter(base_model or s.persona.base_model, dataset,
                                 s.persona.adapter_dir / character, epochs=epochs)
