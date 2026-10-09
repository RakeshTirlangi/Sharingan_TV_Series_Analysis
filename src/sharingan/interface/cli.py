"""``sharingan`` command-line interface."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..core import get_settings

app = typer.Typer(help="Sharingan: read a TV series with NLP & LLMs.", no_args_is_help=True,
                  pretty_exceptions_show_locals=False)
jutsu_app = typer.Typer(help="Jutsu-type text classifier.", no_args_is_help=True)
persona_app = typer.Typer(help="Character persona chatbot.", no_args_is_help=True)
app.add_typer(jutsu_app, name="jutsu")
app.add_typer(persona_app, name="persona")
console = Console()


@app.command()
def scrape(limit: Optional[int] = typer.Option(None, help="Only fetch the first N jutsu articles.")):
    """Crawl jutsu articles from the Naruto Fandom wiki (MediaWiki API) into JSON Lines."""
    from ..scraping.runner import crawl_jutsus

    out = crawl_jutsus(get_settings().paths.jutsus_jsonl, limit=limit)
    console.print(f"[green]Saved[/] {sum(1 for _ in out.open())} jutsus -> {out}")


@app.command()
def themes(refresh: bool = typer.Option(False, help="Recompute even if cached results exist."),
           labels: Optional[str] = typer.Option(None, help="Comma-separated custom theme labels.")):
    """Zero-shot theme scores per episode, per arc and for the whole series."""
    from ..analysis.themes import series_profile
    from ..pipeline import run_themes

    result = run_themes(refresh=refresh, labels=[t.strip() for t in labels.split(",")] if labels else None)
    table = Table("Theme", "Mean P(entail)", "Relative", title="Series theme profile")
    for r in series_profile(result.episodes, result.labels).itertuples():
        table.add_row(r.theme, f"{r.score:.3f}", "█" * int(round(r.relative * 20)))
    console.print(table)


@app.command()
def characters(refresh: bool = typer.Option(False, help="Re-run NER even if mentions are cached.")):
    """NER character mentions -> weighted co-occurrence network (HTML + metrics)."""
    from ..pipeline import run_characters

    network = run_characters(refresh=refresh)
    table = Table("Character", "Mentions", "PageRank", "Betweenness", "Community", title="Most central characters")
    for r in network.metrics.head(15).itertuples():
        table.add_row(r.character, str(r.mentions), f"{r.pagerank:.3f}", f"{r.betweenness:.3f}",
                      str(r.community + 1))
    console.print(table)


@jutsu_app.command("train")
def jutsu_train(model: Optional[str] = typer.Option(None, help="HF encoder to fine-tune."),
                epochs: Optional[int] = typer.Option(None),
                push: bool = typer.Option(False, help="Push to jutsu.hub_model_id on the HF Hub.")):
    """Train baselines + fine-tune a transformer, evaluated on a held-out test split."""
    from ..pipeline import run_jutsu_training

    report = run_jutsu_training(model_name=model, epochs=epochs, push=push)
    table = Table("Model", "Test macro-F1", "Test accuracy", "Train time (s)", title="Jutsu classifier")
    for name, r in {**report["baselines"], "transformer": report["transformer"]}.items():
        table.add_row(name, f"{r['macro_f1']:.3f}", f"{r['accuracy']:.3f}", str(r["train_seconds"]))
    console.print(table)


@jutsu_app.command("cv")
def jutsu_cv(folds: int = 5, transformer: bool = typer.Option(True, help="Include the fine-tuned model (slow on CPU)."),
             epochs: Optional[int] = typer.Option(None)):
    """Stratified k-fold comparison of all models (mean ± std, paired per fold)."""
    from ..pipeline import run_jutsu_crossval

    r = run_jutsu_crossval(folds=folds, transformer=transformer, epochs=epochs)
    table = Table("Model", "Macro-F1", "Accuracy", *[f"F1 {c}" for c in get_settings().jutsu.labels],
                  title=f"{folds}-fold cross-validation (mean ± std)")
    fmt = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"  # noqa: E731
    for name, m in r["models"].items():
        table.add_row(name, fmt(m["macro_f1"]), fmt(m["accuracy"]), *[fmt(v) for v in m["per_class_f1"].values()])
    console.print(table)
    if "transformer_minus_tfidf" in r:
        d = r["transformer_minus_tfidf"]
        console.print(f"Transformer − TF-IDF macro-F1: {d['mean']:+.3f} ± {d['std']:.3f} "
                      f"(won {d['folds_won']}/{folds} folds)")


@jutsu_app.command("predict")
def jutsu_predict(text: str, model_path: Optional[str] = typer.Option(None)):
    """Classify a jutsu description."""
    from ..models.jutsu import load_classifier

    path = model_path or str(get_settings().out("models", "jutsu", "model", "x").parent)
    probs = load_classifier(path).predict_proba(text)[0]
    console.print_json(json.dumps({k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}))


@persona_app.command("memory")
def persona_memory(refresh: bool = typer.Option(False)):
    """Build the voice + scene retrieval indices."""
    from ..pipeline import persona_memory as build

    memory = build(refresh=refresh)
    console.print(memory.voice.groupby(["character", "source"]).size().to_string())
    console.print(f"{len(memory.scenes):,} scene windows indexed")


@persona_app.command("train")
def persona_train(character: str = "Naruto", base_model: Optional[str] = None, epochs: int = 3):
    """LoRA fine-tune the chat model on one character's lines (QLoRA on CUDA)."""
    from ..pipeline import run_persona_training

    console.print_json(json.dumps(run_persona_training(character, base_model=base_model, epochs=epochs)))


@persona_app.command("eval")
def persona_eval(character: str = "Naruto"):
    """A/B the character's LoRA adapter against the base model on held-out prompts."""
    from ..models.persona.evaluate import evaluate_adapter
    from ..pipeline import persona_engine

    s = get_settings()
    report = evaluate_adapter(persona_engine(), character)
    out = s.persona.adapter_dir / character / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    table = Table("Metric", "Base", "LoRA", title=f"{character}: base vs. LoRA ({report['prompts']} held-out prompts)")
    for metric in report["metrics"]["base"]:
        table.add_row(metric, str(report["metrics"]["base"][metric]), str(report["metrics"]["lora"][metric]))
    console.print(table)
    for sample in report["samples"]:
        console.print(f"\n[bold]{sample['prompt']}[/]\n  [dim]base:[/] {sample['base']}\n  [red]lora:[/] {sample['lora']}")
    v = report["verdict"]
    failed = [k for k, ok in v["checks"].items() if not ok]
    console.print(f"\nVerdict: {'[green]use the adapter[/]' if v['recommend_adapter'] else '[yellow]keep the base model[/] (failed: ' + ', '.join(failed) + ')'}")
    console.print(f"Report -> {out}")


@persona_app.command("chat")
def persona_chat(character: str = "Naruto"):
    """Chat with a character in the terminal (Ctrl-D to quit)."""
    from ..models.persona import CARDS
    from ..pipeline import persona_engine

    engine = persona_engine()
    history: list[dict] = []
    console.print(f"[bold]{character}:[/] {CARDS[character].greeting}")
    while True:
        try:
            message = console.input("[bold cyan]you:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        reply = engine.reply(character, message, history)
        console.print(f"[bold]{character}:[/] {reply}")
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]


@app.command()
def all(skip_jutsu: bool = typer.Option(False, help="Skip classifier training (slowest on CPU).")):
    """Run every offline stage: themes, characters, jutsu classifier, persona memory."""
    from .. import pipeline

    pipeline.run_themes()
    pipeline.run_characters()
    if not skip_jutsu:
        pipeline.run_jutsu_training()
    pipeline.persona_memory()
    console.print("[green]All stages complete.[/] Launch the UI with [bold]sharingan app[/].")


@app.command("app")
def launch(port: int = typer.Option(8000), host: str = typer.Option("127.0.0.1"),
           reload: bool = typer.Option(False, help="Auto-reload on code changes (development).")):
    """Serve the API and the built React site (web/dist) on one port."""
    import uvicorn

    from .api import WEB_DIST

    if not WEB_DIST.exists():
        console.print("[yellow]web/dist not found[/]: API only. Build the site with "
                      "[bold]cd web && npm install && npm run build[/], or run [bold]npm run dev[/] for hot reload.")
    console.print(f"Sharingan running at [bold]http://{host}:{port}[/]  (API docs: /docs)")
    uvicorn.run("sharingan.interface.api:app", host=host, port=port, reload=reload, log_level="warning")


if __name__ == "__main__":
    app()
