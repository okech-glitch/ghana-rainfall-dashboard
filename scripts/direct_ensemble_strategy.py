"""Blend baseline CatBoost probabilities with tree ensembles using recovered validation split."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
CONFIG_PATH = BASE_DIR / "config" / "training.yaml"

BASELINE_VAL_PROBS = ARTIFACTS_DIR / "real_val_predictions.npy"
BASELINE_TEST_PROBS = ARTIFACTS_DIR / "test_probabilities.npy"
VAL_LABELS_PATH = ARTIFACTS_DIR / "real_val_labels.npy"
VAL_INDICES_PATH = ARTIFACTS_DIR / "recovered_val_indices.npy"
LABEL_MAP_PATH = ARTIFACTS_DIR / "real_val_label_mapping.json"

# Ensure src/ is importable for engineer_features
sys.path.append(str(BASE_DIR / "src"))


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config at {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def engineer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    from preprocessing import engineer_features

    return engineer_features(df.copy())


def derive_label_mapping() -> Dict[str, int]:
    if LABEL_MAP_PATH.exists():
        mapping = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        return {v: int(k) for k, v in mapping.items()}
    default = ["HEAVYRAIN", "MEDIUMRAIN", "NORAIN", "SMALLRAIN"]
    return {label: idx for idx, label in enumerate(default)}

# Helper used inside FunctionTransformer must be pickleable (defined at module level).
def flatten_text(column) -> np.ndarray:
    arr = np.asarray(column)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(arr.shape[0])
    series = pd.Series(arr)
    return series.astype(str).fillna("").to_numpy()


def build_preprocessor(
    num_features: Sequence[str],
    cat_features: Sequence[str],
    text_features: Sequence[str],
    tfidf_max: int = 800,
) -> ColumnTransformer:
    transformers: List[Tuple[str, object, Sequence[str]]] = []

    if num_features:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, list(num_features)))

    if cat_features:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, list(cat_features)))

    for col in text_features:
        transformers.append(
            (
                f"text_{col}",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(flatten_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=tfidf_max, ngram_range=(1, 2)),
                        ),
                    ]
                ),
                [col],
            )
        )

    return ColumnTransformer(transformers=transformers)


def ensure_columns(frame: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    work = frame.copy()
    for col in required:
        if col not in work.columns:
            work[col] = np.nan
    return work[required]


def load_baseline_arrays() -> Tuple[np.ndarray, np.ndarray]:
    if not BASELINE_VAL_PROBS.exists() or not BASELINE_TEST_PROBS.exists():
        raise FileNotFoundError("Baseline probability artifacts missing. Run use_existing_baseline.py first.")
    return np.load(BASELINE_VAL_PROBS), np.load(BASELINE_TEST_PROBS)


def train_with_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, float, float, object]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(np.unique(y))), dtype=np.float32)
    metrics: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model_clone = clone(model)
        model_clone.fit(X[train_idx], y[train_idx])
        preds = model_clone.predict_proba(X[val_idx])
        oof[val_idx] = preds
        score = f1_score(y[val_idx], preds.argmax(axis=1), average="macro")
        metrics.append(score)
        print(f"  Fold {fold}: macro F1 = {score:.4f}")

    mean_score = float(np.mean(metrics))
    std_score = float(np.std(metrics))

    final_model = clone(model)
    final_model.fit(X, y)
    return oof, mean_score, std_score, final_model


def main() -> None:
    config = load_config()
    cat_features = config.get("cat_features", [])
    text_features = config.get("text_features", [])
    num_features = config.get("num_features", [])
    feature_cols = cat_features + text_features + num_features

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")

    engineered_train = engineer_dataset(train_raw)
    engineered_test = engineer_dataset(test_raw)

    missing_cols = [c for c in feature_cols if c not in engineered_train.columns]
    if missing_cols:
        raise KeyError(f"engineer_features missing required columns: {missing_cols}")

    engineered_test = ensure_columns(engineered_test, feature_cols)

    label_mapping = derive_label_mapping()
    y_series = (train_raw["Target"] if "Target" in train_raw.columns else train_raw["target"]).astype(str)
    y_encoded = y_series.map(label_mapping).to_numpy(dtype=int)

    val_indices = np.load(VAL_INDICES_PATH)
    if len(val_indices) != len(np.load(VAL_LABELS_PATH)):
        raise ValueError("Recovered validation indices size mismatch.")

    train_mask = np.ones(len(train_raw), dtype=bool)
    train_mask[val_indices] = False

    X_train_df = ensure_columns(engineered_train.iloc[train_mask], feature_cols)
    X_val_df = ensure_columns(engineered_train.iloc[val_indices], feature_cols)
    X_test_df = ensure_columns(engineered_test, feature_cols)

    preprocessor = build_preprocessor(num_features, cat_features, text_features)

    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    y_train = y_encoded[train_mask]
    y_val = y_encoded[val_indices]

    print("Training RandomForest with 5-fold CV...")
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_oof, rf_mean, rf_std, rf_final = train_with_cv(rf_model, X_train, y_train)
    print(f"RandomForest CV macro F1: {rf_mean:.4f} ± {rf_std:.4f}")

    print("Training ExtraTrees with 5-fold CV...")
    et_model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=43,
        n_jobs=-1,
    )
    et_oof, et_mean, et_std, et_final = train_with_cv(et_model, X_train, y_train)
    print(f"ExtraTrees CV macro F1: {et_mean:.4f} ± {et_std:.4f}")

    baseline_val_probs, baseline_test_probs = load_baseline_arrays()
    val_labels = np.load(VAL_LABELS_PATH)

    rf_val_probs = rf_final.predict_proba(X_val)
    et_val_probs = et_final.predict_proba(X_val)
    rf_test_probs = rf_final.predict_proba(X_test)
    et_test_probs = et_final.predict_proba(X_test)

    weight_grid = [
        (1.0, 0.0, 0.0),
        (0.8, 0.1, 0.1),
        (0.7, 0.15, 0.15),
        (0.6, 0.2, 0.2),
        (0.5, 0.25, 0.25),
        (0.4, 0.3, 0.3),
        (0.34, 0.33, 0.33),
    ]

    baseline_preds = baseline_val_probs.argmax(axis=1)
    baseline_f1 = f1_score(val_labels, baseline_preds, average="macro")
    print(f"Baseline validation macro F1: {baseline_f1:.6f}")

    best = {
        "weights": (1.0, 0.0, 0.0),
        "macro_f1": baseline_f1,
    }

    for w in weight_grid:
        blended_val = w[0] * baseline_val_probs + w[1] * rf_val_probs + w[2] * et_val_probs
        score = f1_score(val_labels, blended_val.argmax(axis=1), average="macro")
        print(f"Weights {w}: macro F1 = {score:.6f}")
        if score > best["macro_f1"]:
            best = {"weights": w, "macro_f1": score}

    weights = best["weights"]
    print(f"Best weights: {weights} | macro F1 = {best['macro_f1']:.6f}")

    blended_test = weights[0] * baseline_test_probs + weights[1] * rf_test_probs + weights[2] * et_test_probs
    class_order = [label for label, idx in sorted(derive_label_mapping().items(), key=lambda x: x[1])]
    submission_labels = [class_order[idx] for idx in blended_test.argmax(axis=1)]

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    submission_path = SUBMISSIONS_DIR / "ensemble_rf_et_seed42.csv"
    pd.DataFrame({"ID": test_raw["ID"], "target": submission_labels}).to_csv(submission_path, index=False)

    report = {
        "baseline_macro_f1": float(baseline_f1),
        "ensemble_macro_f1": float(best["macro_f1"]),
        "weights": {
            "baseline": weights[0],
            "random_forest": weights[1],
            "extra_trees": weights[2],
        },
        "rf_cv_macro_f1": float(rf_mean),
        "rf_cv_std": float(rf_std),
        "et_cv_macro_f1": float(et_mean),
        "et_cv_std": float(et_std),
        "val_size": int(len(val_labels)),
        "train_size": int(len(y_train)),
        "submission": str(submission_path),
    }

    report_path = REPORTS_DIR / "ensemble_rf_et_seed42.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "rf_model": rf_final,
            "et_model": et_final,
            "feature_columns": feature_cols,
        },
        ARTIFACTS_DIR / "ensemble_rf_et_models.joblib",
    )

    print(f"Saved submission -> {submission_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
