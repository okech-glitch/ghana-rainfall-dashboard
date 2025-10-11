"""Utility helpers for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from .config import DATA_PATH, LOG_PATH, SUBMISSION_PATH


def configure_logging(name: str = "ghana_intel") -> logging.Logger:
    """Configure and return a module-level logger.

    Logs are emitted to both stdout and a rotating file within `logs/`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs(LOG_PATH, exist_ok=True)
    log_file = Path(LOG_PATH) / f"{name}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger configured with handlers: %s", logger.handlers)
    return logger


@contextlib.contextmanager
def timer(description: str) -> Iterator[None]:
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logging.getLogger("ghana_intel").info("%s took %.2fs", description, elapsed)


def ensure_directory(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with basic validation."""
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"CSV file not found: {full_path}")
    df = pd.read_csv(full_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {full_path}")
    return df


def detect_target_column(columns: Iterable[str]) -> Optional[str]:
    """Infer the most likely target column name from provided options."""
    candidate_names = {
        "target",
        "Target",
        "rain_type",
        "rainfall_type",
        "12hr_rain_type",
        "24hr_rain_type",
        "RainType",
    }

    for column in columns:
        if column in candidate_names:
            return column
    return None


def validate_submission_format(submission: pd.DataFrame, sample_path: str | Path) -> None:
    """Ensure submission matches the sample submission format exactly."""
    sample = load_csv(sample_path)
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match SampleSubmission.")
    if len(submission) != len(sample):
        raise ValueError("Submission row count mismatch.")
    if not submission[submission.columns[0]].isin(sample[sample.columns[0]]).all():
        raise ValueError("Submission IDs do not align with SampleSubmission IDs.")


def save_json(data: dict, path: str | Path) -> None:
    """Persist dictionary data as JSON."""
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def check_class_balance(target: pd.Series) -> dict[str, float]:
    """Return normalized class distribution."""
    counts = target.value_counts(normalize=True)
    return counts.to_dict()


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return missing value counts per column."""
    return df.isnull().sum().to_dict()


def assert_no_nan_inf(df: pd.DataFrame) -> None:
    """Raise if DataFrame contains NaN or infinite values."""
    if np.any(np.isnan(df.to_numpy())):
        raise ValueError("DataFrame contains NaN values.")
    if np.any(np.isinf(df.to_numpy())):
        raise ValueError("DataFrame contains infinite values.")
