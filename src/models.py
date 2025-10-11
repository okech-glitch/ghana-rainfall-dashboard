"""Model training utilities for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    CATBOOST_PARAMS,
    LGBM_PARAMS,
    N_FOLDS,
    RANDOM_SEED,
    RF_PARAMS,
    TABNET_PARAMS,
    XGB_PARAMS,
)
from .feature_engineering import apply_target_encoding, select_target_encoding_columns
from .utils import configure_logging

LOGGER = configure_logging()


@dataclass
class ModelResult:
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    cv_scores: List[float]
    models: List[object] = field(default_factory=list)


def _compute_sample_weights(y: pd.Series) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute sample weights for class imbalance mitigation."""
    classes, counts = np.unique(y, return_counts=True)
    # Use inverse sqrt frequency for more aggressive minority weighting
    total_samples = len(y)
    class_weights = total_samples / (len(classes) * counts ** 0.5)
    weight_map = {cls: weight for cls, weight in zip(classes, class_weights)}
    sample_weights = y.map(weight_map).to_numpy(dtype=float)
    return sample_weights, weight_map


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
) -> Tuple[object, Dict[int, float]]:
    import lightgbm as lgb

    LOGGER.info("Training LightGBM with params: %s", params)

    train_weights, weight_map = _compute_sample_weights(y_train)
    val_weights = y_val.map(weight_map).to_numpy(dtype=float)

    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, weight=val_weights)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model, weight_map


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
    weight_map: Optional[Dict[int, float]] = None,
) -> object:
    import xgboost as xgb

    LOGGER.info("Training XGBoost with params: %s", params)

    if weight_map is None:
        train_weights, weight_map = _compute_sample_weights(y_train)
    else:
        train_weights = y_train.map(weight_map).to_numpy(dtype=float)
    val_weights = y_val.map(weight_map).to_numpy(dtype=float)

    # Calculate scale_pos_weight for each class
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    scale_pos_weight = total_samples / (len(classes) * counts)
    params['scale_pos_weight'] = scale_pos_weight.mean()  # Use average for simplicity

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val, weight=val_weights)

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dval, "valid")],
        evals_result=evals_result,
        early_stopping_rounds=200,
        verbose_eval=100,
    )
    model.evals_result_ = evals_result
    return model


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> object:
    from catboost import CatBoostClassifier

    LOGGER.info("Training CatBoost with params: %s", params)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=params.get("verbose", 100),
    )
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> object:
    LOGGER.info("Training RandomForest with params: %s", params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_tabnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
    weight_map: Optional[Dict[int, float]] = None,
) -> object:
    from pytorch_tabnet.metrics import Metric
    from pytorch_tabnet.tab_model import TabNetClassifier

    class F1Macro(Metric):
        def __init__(self):
            self._name = "f1_macro"
            self._maximize = True

        def __call__(self, y_true, y_pred):
            preds = np.argmax(y_pred, axis=1)
            return f1_score(y_true, preds, average="macro")

    LOGGER.info("Training TabNet with params: %s", params)

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    model = TabNetClassifier(**params)
    if weight_map is None:
        train_weights, weight_map = _compute_sample_weights(y_train)
    else:
        train_weights = y_train.map(weight_map).to_numpy(dtype=float)
    val_weights = y_val.map(weight_map).to_numpy(dtype=float)

    model.fit(
        X_train.to_numpy(),
        y_train_np,
        eval_set=[(X_val.to_numpy(), y_val_np)],
        eval_name=["valid"],
        weights=train_weights,
        max_epochs=300,
        patience=50,
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=0,
    )
    return model


def predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    module_name = model.__class__.__module__
    class_name = model.__class__.__name__

    if hasattr(model, "predict_proba"):
        if module_name.startswith("pytorch_tabnet"):
            if isinstance(X, pd.DataFrame):
                data = X.to_numpy(dtype=np.float32, copy=False)
            else:
                data = np.asarray(X, dtype=np.float32)
            return model.predict_proba(data)
        return model.predict_proba(X)

    if module_name.startswith("lightgbm"):
        if isinstance(X, pd.DataFrame):
            data = X.to_numpy(dtype=np.float32, copy=False)
        elif isinstance(X, np.ndarray):
            data = X.astype(np.float32, copy=False)
        else:  # fallback for other array-like structures
            data = np.asarray(X, dtype=np.float32)
        preds = model.predict(data)
        if isinstance(preds, pd.DataFrame):
            preds = preds.to_numpy()
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    if class_name == "Booster" and module_name.startswith("xgboost"):  # XGBoost Booster
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)
    if hasattr(model, "predict"):
        if isinstance(X, pd.DataFrame):
            data = X.to_numpy(dtype=np.float32, copy=False)
        else:
            data = np.asarray(X, dtype=np.float32)
        preds = model.predict(data)
        if preds.ndim == 1:
            preds = pd.get_dummies(preds).to_numpy()
        return preds
    raise AttributeError("Model does not support probability predictions.")


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_func: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict], object],
    params: Dict,
    test_df: Optional[pd.DataFrame] = None,
    encoder_cols: Optional[Sequence[str]] = None,
    n_classes: int = 4,
    n_folds: int = N_FOLDS,
) -> ModelResult:
    """Cross-validate model and return OOF/test predictions."""
    encoder_cols = encoder_cols or []

    # Use GroupKFold based on district if available
    if "district" in X.columns:
        groups = X["district"]
        skf = GroupKFold(n_splits=min(n_folds, groups.nunique()))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    oof_predictions = np.zeros((len(X), n_classes), dtype=float)
    test_predictions = np.zeros((len(test_df), n_classes), dtype=float) if test_df is not None else None
    cv_scores: List[float] = []
    models: List[object] = []
    weight_map: Optional[Dict[int, float]] = None

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y, groups=groups if "district" in X.columns else None)):
        LOGGER.info("=== Fold %d/%d ===", fold + 1, n_folds)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        if encoder_cols:
            X_train, X_valid, encoder = apply_target_encoding(
                X_train,
                y_train,
                X_valid,
                encoder_cols,
                n_classes,
            )
        else:
            encoder = None

        model_output = model_func(X_train, y_train, X_valid, y_valid, params)
        if isinstance(model_output, tuple):
            model, weight_map = model_output
        else:
            model = model_output
        val_pred = predict_proba(model, X_valid)
        oof_predictions[valid_idx] = val_pred

        val_label = np.argmax(val_pred, axis=1)
        fold_f1 = f1_score(y_valid, val_label, average="macro")
        cv_scores.append(fold_f1)
        LOGGER.info("Fold %d F1: %.6f", fold + 1, fold_f1)

        models.append(model)

        if test_df is not None:
            X_test_fold = test_df.copy()
            if encoder is not None:
                X_test_fold = encoder.transform(X_test_fold)
            test_pred = predict_proba(model, X_test_fold)
            test_predictions += test_pred / n_folds

    mean_f1 = float(np.mean(cv_scores))
    std_f1 = float(np.std(cv_scores))
    LOGGER.info("=== CV F1: %.6f (+/- %.6f) ===", mean_f1, std_f1)

    return ModelResult(models=models, oof_predictions=oof_predictions, test_predictions=test_predictions, cv_scores=cv_scores)


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_df: pd.DataFrame,
    target_classes: Sequence[str],
) -> Dict[str, ModelResult]:
    """Train all required models and return their results."""
    encoder_cols = select_target_encoding_columns(X)
    LOGGER.info("Target encoding columns: %s", encoder_cols)

    results: Dict[str, ModelResult] = {}

    results["lightgbm"] = cross_validate_model(
        X,
        y,
        model_func=lambda X_tr, y_tr, X_val, y_val, params=LGBM_PARAMS: train_lgbm(
            X_tr, y_tr, X_val, y_val, params
        ),
        params=LGBM_PARAMS,
        test_df=test_df,
        encoder_cols=encoder_cols,
    )

    def xgb_with_weights(X_tr, y_tr, X_val, y_val, params=XGB_PARAMS, weight_map=None):
        return train_xgb(X_tr, y_tr, X_val, y_val, params, weight_map)

    results["xgboost"] = cross_validate_model(
        X,
        y,
        model_func=xgb_with_weights,
        params=XGB_PARAMS,
        test_df=test_df,
        encoder_cols=encoder_cols,
    )

    results["catboost"] = cross_validate_model(
        X,
        y,
        model_func=lambda X_tr, y_tr, X_val, y_val, params=CATBOOST_PARAMS: train_catboost(X_tr, y_tr, X_val, y_val, params),
        params=CATBOOST_PARAMS,
        test_df=test_df,
        encoder_cols=encoder_cols,
    )

    results["random_forest"] = cross_validate_model(
        X,
        y,
        model_func=lambda X_tr, y_tr, X_val, y_val, params=RF_PARAMS: train_random_forest(X_tr, y_tr, X_val, y_val, params),
        params=RF_PARAMS,
        test_df=test_df,
        encoder_cols=encoder_cols,
    )

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401

        def tabnet_with_weights(X_tr, y_tr, X_val, y_val, params=TABNET_PARAMS, weight_map=None):
            return train_tabnet(X_tr, y_tr, X_val, y_val, params, weight_map)

        results["tabnet"] = cross_validate_model(
            X,
            y,
            model_func=tabnet_with_weights,
            params=TABNET_PARAMS,
            test_df=test_df,
            encoder_cols=encoder_cols,
        )
    except ImportError:
        LOGGER.warning("TabNet not installed; skipping TabNet training.")

    return results
