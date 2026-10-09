"""FastAPI backend for the React frontend (``web/``).

Read endpoints serve precomputed outputs (``sharingan all``); the only live model calls
are custom theme scoring, jutsu classification and character chat. In production the
built frontend (``web/dist``) is served from the same origin.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..core import PROJECT_ROOT, get_logger, get_settings
from ..ingest.arcs import ARC_ORDER, ARCS

log = get_logger(__name__)
WEB_DIST = PROJECT_ROOT / "web" / "dist"
ALL_ARCS = "all"

app = FastAPI(title="Sharingan API", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- cached loads
# Caches are keyed on output-file mtimes, so results produced while the server is
# running (e.g. `sharingan characters` in another shell) are picked up without a restart.
def _mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


@lru_cache(maxsize=2)
def _themes_at(_stamp: float):
    from ..pipeline import run_themes

    return run_themes()


def _themes():
    stamp = _mtime(get_settings().out("themes", "episodes.parquet"))
    return _themes_at(stamp) if stamp else None


@lru_cache(maxsize=32)
def _network_at(arc: str, _stamp: float):
    from ..analysis.characters import CharacterNetwork, clean_mentions

    mentions = clean_mentions(pd.read_parquet(get_settings().out("characters", "mentions.parquet")))
    c = get_settings().characters
    whole = arc == ALL_ARCS
    return CharacterNetwork.from_mentions(mentions, window=c.window_lines, tau=c.decay_tau,
                                          min_mentions=c.min_mentions if whole else max(3, c.min_mentions // 3),
                                          arc=None if whole else arc)


def _network(arc: str):
    stamp = _mtime(get_settings().out("characters", "mentions.parquet"))
    return _network_at(arc, stamp) if stamp else None


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _require(value, what: str, command: str):
    if value is None:
        raise HTTPException(404, f"{what} not computed yet. Run `sharingan {command}`.")
    return value


# ---------------------------------------------------------------------- meta
@app.get("/api/overview")
def overview():
    s = get_settings()
    dialogue_path = s.paths.processed_dir / "dialogue.parquet"
    d = pd.read_parquet(dialogue_path, columns=["episode"]) if dialogue_path.exists() else None
    metrics_path = s.out("models", "jutsu", "reports", "metrics.json")
    report = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    net = _network(ALL_ARCS)
    from ..models.persona import CARDS

    return {
        "version": __version__,
        "episodes": int(d["episode"].nunique()) if d is not None else 0,
        "lines": int(len(d)) if d is not None else 0,
        "jutsus": int(sum(report["data"].values())) if report else 0,
        "characters": int(net.graph.number_of_nodes()) if net else 0,
        "relationships": int(net.graph.number_of_edges()) if net else 0,
        "personas": len(CARDS),
        "models": {"themes": s.themes.model_name, "ner": s.characters.spacy_model,
                   "jutsu": report["transformer"]["model"] if report else s.jutsu.model_name,
                   "persona": s.persona.base_model, "embeddings": s.persona.embedding_model},
        "ready": {"themes": _themes() is not None, "characters": net is not None,
                  "jutsu": s.out("models", "jutsu", "model", "config.json").exists(),
                  "persona": (s.out("persona", "memory", "embeddings.npz")).exists()},
    }


# -------------------------------------------------------------------- themes
@app.get("/api/themes")
def themes():
    from ..analysis.themes import arc_profile, series_profile

    result = _require(_themes(), "Theme scores", "themes")
    labels = result.labels
    lift = arc_profile(result.episodes, labels, ARC_ORDER)
    return {
        "labels": labels,
        "model": get_settings().themes.model_name,
        "profile": _records(series_profile(result.episodes, labels)),
        "episodes": _records(result.episodes[["episode", "arc", *labels]].round(4)),
        "arcs": [{"arc": arc, "lift": {k: round(float(v), 3) for k, v in row.items()}} for arc, row in lift.iterrows()],
        "arcBounds": [{"arc": name, "first": a, "last": b} for a, b, name in ARCS],
    }


@app.get("/api/themes/excerpts")
def theme_excerpts(theme: str, k: int = Query(3, ge=1, le=10)):
    from ..analysis.themes import top_excerpts

    result = _require(_themes(), "Theme scores", "themes")
    if theme not in result.labels:
        raise HTTPException(400, f"Unknown theme {theme!r}")
    rows = top_excerpts(result.chunks, theme, k).rename(columns={theme: "score"})
    return _records(rows.round({"score": 4}))


class ThemeQuery(BaseModel):
    labels: list[str] = Field(min_length=1, max_length=8)
    first: int = Field(1, ge=1, le=220)
    last: int = Field(5, ge=1, le=220)


@app.post("/api/themes/score")
def score_custom_themes(q: ThemeQuery):
    from ..analysis.themes import series_profile
    from ..pipeline import run_themes

    if q.last < q.first or q.last - q.first > 19:
        raise HTTPException(400, "Choose a range of at most 20 episodes (scoring runs live).")
    labels = [label.strip() for label in q.labels if label.strip()]
    result = run_themes(labels=labels, episodes=list(range(q.first, q.last + 1)))
    return {"labels": labels, "profile": _records(series_profile(result.episodes, labels)),
            "episodes": _records(result.episodes[["episode", "arc", *labels]].round(4))}


# ---------------------------------------------------------------- characters
@app.get("/api/characters")
def characters(arc: str = ALL_ARCS, top_edges: int = Query(250, ge=10, le=1000)):
    if arc != ALL_ARCS and arc not in ARC_ORDER:
        raise HTTPException(400, f"Unknown arc {arc!r}")
    import networkx as nx

    net = _require(_network(arc), "Character mentions", "characters")
    edges = net.edges.head(top_edges)
    # Draw only the largest connected component: stray pairs would push the layout apart.
    drawn = nx.from_pandas_edgelist(edges, "source", "target")
    keep = max(nx.connected_components(drawn), key=len) if drawn.number_of_nodes() else set()
    edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
    nodes = net.metrics[net.metrics["character"].isin(keep)].rename(columns={"character": "id"})
    return {
        "arc": arc, "arcs": ARC_ORDER,
        "nodes": _records(nodes.round(5)),
        "links": _records(edges.round(3)),
        "ranking": _records(net.metrics.head(30).round(5)),
        "communities": int(net.metrics["community"].nunique()),
    }


@app.get("/api/characters/ties")
def character_ties(name: str, arc: str = ALL_ARCS, k: int = Query(8, ge=1, le=30)):
    net = _require(_network(arc), "Character mentions", "characters")
    if name not in net.graph:
        raise HTTPException(404, f"{name!r} is not in this network")
    return _records(net.strongest_ties(name, k))


# --------------------------------------------------------------------- jutsu
def _jutsu_path() -> str:
    s = get_settings()
    local = s.out("models", "jutsu", "model", "config.json")
    if local.exists():
        return str(local.parent)
    if s.jutsu.hub_model_id:
        return s.jutsu.hub_model_id
    raise HTTPException(404, "No trained classifier. Run `sharingan jutsu train`.")


class JutsuQuery(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


@app.post("/api/jutsu/classify")
def classify(q: JutsuQuery):
    from ..models.jutsu import load_classifier

    probs = load_classifier(_jutsu_path()).predict_proba(q.text)[0]
    ranked = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
    return {"label": next(iter(ranked)), "probabilities": {k: round(v, 4) for k, v in ranked.items()}}


@app.get("/api/jutsu/report")
def jutsu_report():
    reports = get_settings().out("models", "jutsu", "reports", "x").parent
    path = reports / "metrics.json"
    report = json.loads(_require(path if path.exists() else None, "Jutsu report", "jutsu train").read_text())
    cv = reports / "crossval.json"
    report["crossval"] = json.loads(cv.read_text()) if cv.exists() else None
    return report


# ------------------------------------------------------------------- persona
@app.get("/api/personas")
def personas():
    from ..models.persona import CARDS

    adapters = get_settings().persona.adapter_dir
    return [{"name": c.name, "fullName": c.full_name, "identity": c.identity, "greeting": c.greeting,
             "speech": c.speech, "hasAdapter": (adapters / c.name / "adapter_config.json").exists()}
            for c in CARDS.values()]


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatQuery(BaseModel):
    character: str
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)


@app.post("/api/chat")
def chat(q: ChatQuery):
    """Server-sent events: one ``context`` event (retrieved memories), ``delta`` events, then ``done``."""
    from ..models.persona import CARDS
    from ..pipeline import persona_engine

    if q.character not in CARDS:
        raise HTTPException(400, f"Unknown character {q.character!r}")
    engine = persona_engine()
    history = [t.model_dump() for t in q.history]
    messages, retrieved = engine.prepare(q.character, q.message, history)

    def events():
        yield f"event: context\ndata: {json.dumps(retrieved)}\n\n"
        sent = ""
        try:
            for text in engine.stream(q.character, q.message, messages=messages):
                if len(text) > len(sent):
                    yield f"event: delta\ndata: {json.dumps(text[len(sent):])}\n\n"
                    sent = text
        except Exception as exc:  # surface generation errors to the client
            log.exception("chat generation failed")
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -------------------------------------------------------------- static site
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        file = WEB_DIST / path
        if path and file.is_file():
            return FileResponse(file)
        # index.html must never be cached: it points at the content-hashed bundles of the latest build.
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
