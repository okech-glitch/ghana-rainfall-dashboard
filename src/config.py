"""Configuration module for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

# Reproducibility -----------------------------------------------------------
RANDOM_SEED: int = 42
N_FOLDS: int = 10
N_JOBS: int = -1

# Model parameters ---------------------------------------------------------
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.006347540825687927,
    "num_leaves": 399,
    "max_depth": 12,
    "min_child_samples": 10,
    "min_child_weight": 0.04120914404625767,
    "subsample": 0.6023850769081028,
    "subsample_freq": 1,
    "colsample_bytree": 0.6079845841277962,
    "reg_alpha": 0.0019389177677252227,
    "reg_lambda": 0.02490139835385627,
    "min_split_gain": 0.052042289422128525,
    "max_bin": 442,
    "n_estimators": 1176,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": N_JOBS,
}

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 4,
    "eval_metric": "mlogloss",
    "learning_rate": 0.04101356102614908,
    "max_depth": 12,
    "min_child_weight": 2.1742626564803027,
    "subsample": 0.9396207146659871,
    "colsample_bytree": 0.6034147699629462,
    "colsample_bylevel": 0.7054208833434691,
    "gamma": 0.003147478100735665,
    "reg_alpha": 0.00014532496178032488,
    "reg_lambda": 0.0035592696909190983,
    "max_delta_step": 0.5135981842040488,
    "tree_method": "hist",
    "seed": RANDOM_SEED,
}

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.01,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "random_seed": RANDOM_SEED,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "auto_class_weights": "Balanced",
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
}

TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.3,
    "n_independent": 2,
    "n_shared": 2,
    "seed": RANDOM_SEED,
    "verbose": 1,
}

# Paths --------------------------------------------------------------------
DATA_PATH: str = "data/"
MODEL_PATH: str = "models/"
SUBMISSION_PATH: str = "submissions/"
LOG_PATH: str = "logs/"
EXPLAIN_PATH: str = "models/explainability/"
EXPORT_PATH: str = "models/exports/"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducibility across supported libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_path(path: str) -> str:
    """Return an absolute path for the provided relative path."""
    return os.path.join(os.getcwd(), path)
