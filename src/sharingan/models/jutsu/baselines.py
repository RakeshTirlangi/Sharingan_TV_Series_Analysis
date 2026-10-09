"""Classical baselines, so the fine-tuned transformer's gain is measured, not assumed."""

from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from ...core import get_logger
from .data import evaluate

log = get_logger(__name__)


def tfidf_logreg(splits, classes):
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    )
    model = make_pipeline(features, LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"))
    model.fit(splits["train"]["text"], splits["train"]["label"])
    return evaluate(splits["test"]["label"].tolist(), model.predict(splits["test"]["text"]).tolist(), classes)


def embedding_logreg(splits, classes, embedding_model: str):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(embedding_model)
    enc = lambda s: encoder.encode(s["text"].tolist(), batch_size=64, normalize_embeddings=True)  # noqa: E731
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    clf.fit(enc(splits["train"]), splits["train"]["label"])
    return evaluate(splits["test"]["label"].tolist(), clf.predict(enc(splits["test"])).tolist(), classes)


def run_baselines(splits, *, embedding_model: str, classes: list[str] | None = None) -> dict:
    classes = classes or sorted(splits["train"]["label"].unique())
    results = {}
    for name, fn in [("tfidf_logreg", lambda: tfidf_logreg(splits, classes)),
                     ("embedding_logreg", lambda: embedding_logreg(splits, classes, embedding_model))]:
        t0 = time.perf_counter()
        results[name] = fn()
        results[name]["train_seconds"] = round(time.perf_counter() - t0, 1)
        log.info(f"Baseline {name}: macro-F1 {results[name]['macro_f1']:.3f} "
                 f"accuracy {results[name]['accuracy']:.3f}")
    return results
