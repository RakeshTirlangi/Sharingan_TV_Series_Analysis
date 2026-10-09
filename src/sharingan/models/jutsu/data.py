"""Train / validation / test splitting and evaluation helpers for the jutsu classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def make_splits(df: pd.DataFrame, *, seed: int = 42, val: float = 0.15, test: float = 0.15) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 split. The test split is never seen during model selection."""
    train_val, test_df = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(
        train_val, test_size=val / (1 - test), stratify=train_val["label"], random_state=seed
    )
    return {k: v.reset_index(drop=True) for k, v in {"train": train_df, "val": val_df, "test": test_df}.items()}


def class_weights(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """Inverse-frequency ("balanced") weights so rare genjutsu still shape the loss."""
    return compute_class_weight("balanced", classes=np.array(classes), y=labels.to_numpy())


def evaluate(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=classes)), 4),
        "per_class": {
            k: {m: round(float(v), 4) for m, v in d.items()}
            for k, d in classification_report(y_true, y_pred, labels=classes, output_dict=True,
                                              zero_division=0).items()
            if k in classes
        },
        "confusion_matrix": {"labels": classes,
                             "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist()},
    }
