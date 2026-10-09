"""Stratified k-fold comparison of the jutsu classifiers.

A single test split holds only ~11 Genjutsu examples, so one prediction swings Genjutsu F1 by
~0.06. Cross-validation evaluates every model on the same k folds (paired), so differences are
reported with their spread instead of as a single noisy number.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ...core import get_logger
from .baselines import embedding_logreg, tfidf_logreg

log = get_logger(__name__)


def _summary(fold_reports: list[dict], classes: list[str]) -> dict:
    def stat(values):
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values, ddof=1)), 4)}

    return {
        "macro_f1": stat([r["macro_f1"] for r in fold_reports]),
        "accuracy": stat([r["accuracy"] for r in fold_reports]),
        "per_class_f1": {c: stat([r["per_class"][c]["f1-score"] for r in fold_reports]) for c in classes},
        "folds": [r["macro_f1"] for r in fold_reports],
    }


def cross_validate(data: pd.DataFrame, classes: list[str], *, folds: int = 5, seed: int = 42,
                   embedding_model: str, transformer: dict | None = None, work_dir: Path) -> dict:
    """``transformer`` holds JutsuTrainer kwargs (model_name, epochs, ...) or ``None`` to skip it."""
    from .trainer import JutsuTrainer

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    reports: dict[str, list[dict]] = {"tfidf_logreg": [], "embedding_logreg": []}
    if transformer:
        reports["transformer"] = []

    for k, (train_idx, test_idx) in enumerate(skf.split(data, data["label"]), 1):
        t0 = time.perf_counter()
        train_full, test = data.iloc[train_idx], data.iloc[test_idx]
        train, val = train_test_split(train_full, test_size=0.15, stratify=train_full["label"], random_state=seed)
        splits = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
                  "test": test.reset_index(drop=True)}
        # Baselines see train+val (they have no model selection), the transformer uses val for it.
        base_splits = {"train": train_full.reset_index(drop=True), "test": splits["test"]}
        reports["tfidf_logreg"].append(tfidf_logreg(base_splits, classes))
        reports["embedding_logreg"].append(embedding_logreg(base_splits, classes, embedding_model))
        if transformer:
            fold_dir = work_dir / f"fold_{k}"
            trainer = JutsuTrainer(labels=classes, output_dir=fold_dir, seed=seed, **transformer)
            reports["transformer"].append(trainer.fit(splits, save=False))
            shutil.rmtree(fold_dir, ignore_errors=True)
        log.info(f"Fold {k}/{folds} done in {time.perf_counter() - t0:.0f}s: "
                 + ", ".join(f"{name} {r[-1]['macro_f1']:.3f}" for name, r in reports.items()))

    result = {"folds": folds, "seed": seed, "n": int(len(data)),
              "models": {name: _summary(r, classes) for name, r in reports.items()}}
    if transformer:
        diffs = [t["macro_f1"] - b["macro_f1"] for t, b in zip(reports["transformer"], reports["tfidf_logreg"])]
        result["transformer_minus_tfidf"] = {
            "mean": round(float(np.mean(diffs)), 4), "std": round(float(np.std(diffs, ddof=1)), 4),
            "folds_won": int(sum(d > 0 for d in diffs)), "per_fold": [round(d, 4) for d in diffs],
        }
        result["models"]["transformer"]["model"] = transformer["model_name"]
    return result


def save_report(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path
