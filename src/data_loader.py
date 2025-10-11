"""Data loading and sanity checks for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .config import DATA_PATH
from .utils import (
    check_class_balance,
    check_missing_values,
    configure_logging,
    detect_target_column,
    ensure_directory,
    load_csv,
    validate_submission_format,
)

LOGGER = configure_logging()


@dataclass
class DatasetBundle:
    """Structured container for dataset objects and metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    target_column: str
    id_column: str
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    submission_columns: List[str]


def _infer_id_column(df: pd.DataFrame) -> str:
    """Infer identifier column name from common patterns."""
    candidate_cols = [col for col in df.columns if col.lower() in {"id", "row_id", "forecast_id"}]
    if candidate_cols:
        return candidate_cols[0]

    for col in df.columns:
        if col.lower().endswith("_id"):
            return col

    raise ValueError("Unable to infer ID column. Ensure dataset contains an identifier column.")


def load_data(data_path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission datasets."""
    ensure_directory(data_path)
    train_path = f"{data_path}train.csv"
    test_path = f"{data_path}test.csv"
    sample_path = f"{data_path}SampleSubmission.csv"

    LOGGER.info("Loading datasets from %s", data_path)
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)
    sample_submission = load_csv(sample_path)

    LOGGER.info("Loaded train shape=%s, test shape=%s", train_df.shape, test_df.shape)
    return train_df, test_df, sample_submission


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission: Optional[pd.DataFrame] = None,
) -> DatasetBundle:
    """Run dataset validations and infer key metadata."""
    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test data must not be empty.")

    target_col = detect_target_column(train_df.columns)
    if target_col is None:
        raise ValueError(
            "Could not detect target column. Please ensure target column matches expected naming conventions."
        )

    id_col = _infer_id_column(train_df.drop(columns=[target_col], errors="ignore"))

    if id_col not in test_df.columns:
        raise ValueError("Test dataset is missing the inferred ID column '%s'." % id_col)

    if sample_submission is not None:
        validate_submission_format(sample_submission.copy(), f"{DATA_PATH}SampleSubmission.csv")
        sample_ids = sample_submission[sample_submission.columns[0]].tolist()
        test_ids = test_df[id_col].tolist()
        if sorted(sample_ids) != sorted(test_ids):
            raise ValueError("Sample submission IDs do not align with test dataset IDs.")

    feature_cols = [col for col in train_df.columns if col not in {target_col, id_col}]

    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])]
    categorical_features = [col for col in feature_cols if col not in numeric_features]

    missing_train = check_missing_values(train_df)
    missing_test = check_missing_values(test_df)
    LOGGER.info("Train missing values: %s", {k: v for k, v in missing_train.items() if v > 0})
    LOGGER.info("Test missing values: %s", {k: v for k, v in missing_test.items() if v > 0})

    class_balance = check_class_balance(train_df[target_col])
    LOGGER.info("Class distribution: %s", class_balance)

    n_classes = train_df[target_col].nunique(dropna=False)
    if n_classes not in {3, 4}:
        raise ValueError(f"Unexpected number of classes: {n_classes}. Expected 3 or 4.")

    submission_columns = (
        list(sample_submission.columns)
        if sample_submission is not None and not sample_submission.empty
        else [id_col, target_col]
    )

    return DatasetBundle(
        train=train_df,
        test=test_df,
        sample_submission=sample_submission if sample_submission is not None else pd.DataFrame(),
        target_column=target_col,
        id_column=id_col,
        feature_columns=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        submission_columns=submission_columns,
    )
