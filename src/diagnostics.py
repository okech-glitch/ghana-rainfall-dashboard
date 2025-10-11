"""Diagnostic utilities to analyze data quality and model performance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .config import DATA_PATH, MODEL_PATH
from .utils import configure_logging, ensure_directory

LOGGER = configure_logging()


def _load_train_dataframe(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Training dataframe is empty.")
    return df


def _find_target_column(columns: Iterable[str]) -> str:
    for col in columns:
        lower = col.lower()
        if "target" in lower or "rain" in lower:
            return col
    raise ValueError("Unable to infer target column name from training data.")


def data_quality_report(train_df: pd.DataFrame, target_column: str) -> None:
    LOGGER.info("===== DATA QUALITY CHECK =====")

    # 1. Class distribution
    counts = train_df[target_column].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    LOGGER.info("Target distribution:\n%s", counts)
    LOGGER.info("Imbalance ratio: %.2f:1", imbalance_ratio)

    # 2. Missing values
    missing = train_df.isna().sum()
    missing_pct = (missing / len(train_df) * 100).round(2)
    missing_df = (
        pd.DataFrame({"column": missing.index, "count": missing.values, "pct": missing_pct.values})
        .query("count > 0")
        .sort_values("count", ascending=False)
    )
    if missing_df.empty:
        LOGGER.info("No missing values detected.")
    else:
        LOGGER.warning("Columns with missing values:\n%s", missing_df.to_string(index=False))

    # 3. Feature types
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    LOGGER.info("Numeric columns: %d | Categorical columns: %d | Total: %d", len(numeric_cols), len(categorical_cols), len(train_df.columns))

    # 4. Cardinality for categorical
    for col in categorical_cols:
        if col == target_column:
            continue
        n_unique = train_df[col].nunique(dropna=False)
        if n_unique > 0:
            LOGGER.debug("Cardinality %s: %d", col, n_unique)
            if n_unique > 50:
                LOGGER.warning("High-cardinality categorical feature detected: %s (%d unique)", col, n_unique)

    # 5. Low variance numeric features
    low_variance = []
    for col in numeric_cols:
        if col == target_column:
            continue
        variance = train_df[col].var()
        if variance < 0.01:
            low_variance.append((col, variance))
    if low_variance:
        for col, variance in low_variance:
            LOGGER.warning("Low-variance numeric feature: %s (var=%.6f)", col, variance)


def _load_predictions_and_labels() -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    oof_path = Path(MODEL_PATH) / "final_oof_predictions.npy"
    y_path = Path(MODEL_PATH) / "y_train.npy"
    le_path = Path(MODEL_PATH) / "label_encoder.npy"

    if not oof_path.exists() or not y_path.exists():
        LOGGER.warning("OOF predictions or labels not found. Expected %s and %s", oof_path, y_path)
        return None, None, None

    try:
        oof = np.load(oof_path)
        y_true = np.load(y_path)
        class_names = np.load(le_path, allow_pickle=True)
        return oof, y_true, class_names
    except Exception as e:
        LOGGER.error("Error loading prediction files: %s", str(e))
        return None, None, None


def per_class_performance_report(oof: np.ndarray, y_true: np.ndarray, class_names: Iterable[str]) -> None:
    LOGGER.info("===== PER-CLASS PERFORMANCE =====")

    y_pred = np.argmax(oof, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(oof.shape[1]))

    header = f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}"
    LOGGER.info(header)
    for idx, class_name in enumerate(class_names):
        LOGGER.info(
            f"{str(class_name):<15}{precision[idx]:>12.4f}{recall[idx]:>12.4f}{f1[idx]:>12.4f}{int(support[idx]):>12d}"
        )
        if f1[idx] < 0.85:
            if recall[idx] < precision[idx]:
                LOGGER.warning("%s: low recall; investigate FN cases.", class_name)
            elif precision[idx] < recall[idx]:
                LOGGER.warning("%s: low precision; investigate FP cases.", class_name)
            else:
                LOGGER.warning("%s: low overall F1; requires attention.", class_name)

    cm = confusion_matrix(y_true, y_pred, labels=range(oof.shape[1]))
    LOGGER.info("Confusion matrix:\n%s", cm)

    misclassification_summary = []
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        if total == 0:
            continue
        for j in range(cm.shape[1]):
            if i == j:
                continue
            if cm[i, j] > 0:
                misclassification_summary.append(
                    {
                        "true": class_names[i],
                        "pred": class_names[j],
                        "count": int(cm[i, j]),
                        "pct": float(cm[i, j] / total * 100),
                    }
                )
    if misclassification_summary:
        misclassification_summary.sort(key=lambda x: x["count"], reverse=True)
        LOGGER.info("Top misclassifications:")
        for item in misclassification_summary[:10]:
            LOGGER.info(
                "%s -> %s: %d (%.2f%%)",
                item["true"],
                item["pred"],
                item["count"],
                item["pct"],
            )

    probs = pd.DataFrame(oof, columns=[str(name) for name in class_names])
    descriptive = probs.describe().loc[["mean", "std", "min", "max"]]
    LOGGER.info("Probability summary by class:\n%s", descriptive)


def metrics_report() -> None:
    metrics_path = Path(MODEL_PATH) / "metrics.json"
    if not metrics_path.exists():
        LOGGER.warning("Metrics file %s not found.", metrics_path)
        return
    metrics = json.loads(metrics_path.read_text())
    LOGGER.info("===== MODEL METRICS =====")
    for model_name, score in metrics.items():
        LOGGER.info("%s macro-F1: %.6f", model_name, score)


def run_diagnostics() -> None:
    ensure_directory(MODEL_PATH)
    train_path = Path(DATA_PATH) / "train.csv"

    train_df = _load_train_dataframe(train_path)
    target_col = _find_target_column(train_df.columns)

    data_quality_report(train_df, target_col)
    metrics_report()

    oof, y_true, class_names = _load_predictions_and_labels()
    if oof is not None and y_true is not None and class_names is not None:
        per_class_performance_report(oof, y_true, class_names)
    else:
        LOGGER.warning("Skipping per-class diagnostics; missing OOF predictions or labels.")


if __name__ == "__main__":
    run_diagnostics()
