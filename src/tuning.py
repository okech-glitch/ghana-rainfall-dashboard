"""Hyperparameter tuning CLI for Ghana Indigenous Intel models using Optuna."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from .config import MODEL_PATH, N_JOBS, RANDOM_SEED, set_seed
from .ensemble import LOGGER
from .feature_engineering import select_target_encoding_columns
from .models import cross_validate_model, train_lgbm, train_xgb
from .pipeline import build_feature_matrices, prepare_datasets
from .utils import ensure_directory


def _load_training_data() -> tuple[pd.DataFrame, pd.Series, int, list[str]]:
    """Prepare model-ready training features and labels."""
    set_seed(RANDOM_SEED)
    bundle = prepare_datasets()
    train_features, y_encoded, _, _, _ = build_feature_matrices(bundle)
    y_series = pd.Series(y_encoded, name="target")
    encoder_cols = select_target_encoding_columns(train_features)
    n_classes = int(len(np.unique(y_series)))
    return train_features, y_series, n_classes, encoder_cols


def _normalize_params(params: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            normalized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _objective_lightgbm(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    encoder_cols: list[str],
    n_classes: int,
    n_folds: int,
) -> float:
    params: Dict[str, object] = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 32, 512),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_bin": trial.suggest_int("max_bin", 128, 512),
        "n_estimators": trial.suggest_int("n_estimators", 800, 3200),
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": N_JOBS,
    }

    trial.set_user_attr("full_params", _normalize_params(params))

    def _train_fold(
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_params: Dict[str, object] = params,
    ):
        return train_lgbm(X_tr, y_tr, X_val, y_val, dict(fold_params))

    result = cross_validate_model(
        X,
        y,
        model_func=_train_fold,
        params=params,
        test_df=None,
        encoder_cols=encoder_cols,
        n_classes=n_classes,
        n_folds=n_folds,
    )

    oof_preds = result.oof_predictions
    score = f1_score(y, np.argmax(oof_preds, axis=1), average="macro")
    trial.set_user_attr("cv_scores", result.cv_scores)
    LOGGER.info("Trial %s LightGBM macro-F1: %.6f", trial.number, score)
    return score


def _objective_xgboost(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    encoder_cols: list[str],
    n_classes: int,
    n_folds: int,
) -> float:
    params: Dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 2.0),
        "tree_method": "hist",
        "seed": RANDOM_SEED,
    }

    trial.set_user_attr("full_params", _normalize_params(params))

    def _train_fold(
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold_params: Dict[str, object] = params,
    ):
        return train_xgb(X_tr, y_tr, X_val, y_val, dict(fold_params))

    result = cross_validate_model(
        X,
        y,
        model_func=_train_fold,
        params=params,
        test_df=None,
        encoder_cols=encoder_cols,
        n_classes=n_classes,
        n_folds=n_folds,
    )

    oof_preds = result.oof_predictions
    score = f1_score(y, np.argmax(oof_preds, axis=1), average="macro")
    trial.set_user_attr("cv_scores", result.cv_scores)
    LOGGER.info("Trial %s XGBoost macro-F1: %.6f", trial.number, score)
    return score


def _save_results(model_name: str, study: optuna.Study) -> None:
    if study.best_trial is None:
        LOGGER.warning("No successful trials to save for %s", model_name)
        return

    best_trial = study.best_trial
    trial_params = best_trial.user_attrs.get("full_params", study.best_params)
    metadata = {
        "model": model_name,
        "best_score": best_trial.value,
        "best_params": _normalize_params(trial_params),
        "trial_number": best_trial.number,
        "cv_scores": best_trial.user_attrs.get("cv_scores"),
        "datetime": datetime.utcnow().isoformat() + "Z",
        "n_trials": len(study.trials),
    }

    ensure_directory(MODEL_PATH)
    output_path = Path(MODEL_PATH) / "tuning_results.json"
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as fp:
                history = json.load(fp)
        except json.JSONDecodeError:
            history = {}
    else:
        history = {}

    history[model_name] = metadata
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2, ensure_ascii=False)

    LOGGER.info("Saved tuning results for %s to %s", model_name, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for top models.")
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], required=True, help="Model to tune.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional time budget in seconds.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds during tuning.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name (useful when providing a storage backend).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL for persistent studies.",
    )
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna worker processes (requires persistent storage).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    LOGGER.info(
        "Starting hyperparameter tuning for %s with %d trials (n_folds=%d)...",
        args.model,
        args.trials,
        args.n_folds,
    )

    X, y, n_classes, encoder_cols = _load_training_data()

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", study_name=args.study_name)

    if args.model == "lightgbm":
        objective = lambda trial: _objective_lightgbm(trial, X, y, encoder_cols, n_classes, args.n_folds)
    else:
        objective = lambda trial: _objective_xgboost(trial, X, y, encoder_cols, n_classes, args.n_folds)

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, n_jobs=args.optuna_jobs)

    LOGGER.info(
        "Best macro-F1 for %s: %.6f (trial %d)",
        args.model,
        study.best_value,
        study.best_trial.number if study.best_trial else -1,
    )

    _save_results(args.model, study)


if __name__ == "__main__":
    main()
