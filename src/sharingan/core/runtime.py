"""Runtime helpers: logging, device selection, reproducibility, artefact caching."""

from __future__ import annotations

import logging
import os
import random
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "sharingan") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("SHARINGAN_LOG_LEVEL", "INFO"),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", message=".*NVML.*")
        _CONFIGURED = True
    return logging.getLogger(name)


def pick_device() -> str:
    """Best available accelerator: CUDA > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def cached_frame(path: Path, build: Callable[[], pd.DataFrame], *, refresh: bool = False) -> pd.DataFrame:
    """Return the parquet at ``path`` if present, otherwise build, persist and return it."""
    log = get_logger()
    if path.exists() and not refresh:
        log.info(f"Loading cached [bold]{path.name}[/]", extra={"markup": True})
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info(f"Saved [bold]{path}[/] ({len(df):,} rows)", extra={"markup": True})
    return df
