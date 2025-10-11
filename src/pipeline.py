"""Unified ML pipeline for Ghana Indigenous Intel rainfall prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from .config import DATA_PATH, MODEL_PATH, RANDOM_SEED, SUBMISSION_PATH, set_seed
from .data_loader import DatasetBundle, load_data, preprocess_data
from .ensemble import (
    apply_thresholds,
    create_precision_weighted_final_ensemble,
    optimize_ensemble_weights,
    optimize_thresholds,
    train_meta_learner,
    weighted_average,
)
from .feature_engineering import FeatureEngineer, create_features
from .explainability import generate_shap_values
from .export_utils import export_to_onnx
from .models import ModelResult, train_all_models
from .utils import (
    assert_no_nan_inf,
    configure_logging,
    ensure_directory,
    save_json,
    validate_submission_format,
)

LOGGER = configure_logging()


def prepare_datasets() -> DatasetBundle:
    train_df, test_df, sample_submission = load_data()
    bundle = preprocess_data(train_df, test_df, sample_submission)
    return bundle


def build_feature_matrices(bundle: DatasetBundle) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, FeatureEngineer, LabelEncoder]:
    engineer = FeatureEngineer(bundle=bundle)
    train_features, train_target, engineer = create_features(
        bundle.train,
        bundle,
        engineer=engineer,
        is_train=True,
    )
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(train_target)

    test_features, _, _ = create_features(bundle.test, bundle, engineer=engineer, is_train=False)
    return train_features, y_encoded, test_features, engineer, label_encoder


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> Dict[str, ModelResult]:
    classes = label_encoder.classes_
    results = train_all_models(X_train, pd.Series(y_train), X_test, target_classes=classes)
    return results


def ensemble_predictions(
    model_results: Dict[str, ModelResult],
    y_true: np.ndarray,
    selected_models: List[str],
) -> Dict[str, np.ndarray]:
    if not selected_models:
        raise ValueError("selected_models must contain at least one model name.")

    oof_list = [model_results[name].oof_predictions for name in selected_models]
    weights = optimize_ensemble_weights(oof_list, y_true)
    test_predictions = [model_results[name].test_predictions for name in selected_models]
    ensemble_test = weighted_average(test_predictions, weights)
    oof_ensemble = weighted_average(oof_list, weights)

    meta_output = train_meta_learner(oof_list, y_true, test_predictions)
    return {
        "weights": weights,
        "predictions": ensemble_test,
        "models": selected_models,
        "oof_ensemble": oof_ensemble,
        "stacked_oof": meta_output["train_predictions"],
        "stacked_predictions": meta_output["test_predictions"],
        "meta_model": meta_output["model"],
        "meta_scaler": meta_output["scaler"],
    }


def evaluate_oof(model_results: Dict[str, ModelResult], y_true: np.ndarray) -> Dict[str, float]:
    scores = {}
    for name, result in model_results.items():
        max_pred = np.argmax(result.oof_predictions, axis=1)
        score = f1_score(y_true, max_pred, average="macro")
        scores[name] = score
        LOGGER.info("Model %s OOF F1: %.6f", name, score)
    return scores


def save_models(model_results: Dict[str, ModelResult]) -> None:
    ensure_directory(MODEL_PATH)
    for name, result in model_results.items():
        for idx, model in enumerate(result.models):
            model_path = Path(MODEL_PATH) / f"{name}_fold{idx}.pkl"
            try:
                import joblib

                joblib.dump(model, model_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to save model %s fold %d: %s", name, idx, exc)


def create_submission_file(
    predictions: np.ndarray,
    bundle: DatasetBundle,
    label_encoder: LabelEncoder,
    filename: str = "final_submission.csv",
) -> pd.DataFrame:
    ensure_directory(SUBMISSION_PATH)
    pred_labels = np.argmax(predictions, axis=1)
    pred_classes = label_encoder.inverse_transform(pred_labels)
    submission = pd.DataFrame({
        bundle.id_column: bundle.test[bundle.id_column],
        bundle.target_column: pred_classes,
    })
    submission = submission[bundle.submission_columns]

    sample_path = Path(DATA_PATH) / "SampleSubmission.csv"
    validate_submission_format(submission.copy(), sample_path)
    submission_path = Path(SUBMISSION_PATH) / filename
    submission.to_csv(submission_path, index=False)
    LOGGER.info("Submission saved to %s", submission_path)
    return submission


def log_metadata(
    metrics: Dict[str, float],
    weights: np.ndarray,
    engineer: FeatureEngineer,
    selected_models: List[str],
    thresholds: np.ndarray,
    stacking_info: Optional[Dict[str, object]] = None,
) -> None:
    metadata = {
        "metrics": metrics,
        "weights": weights.tolist(),
        "selected_models": selected_models,
        "thresholds": thresholds.tolist(),
        "indicator_columns": engineer.indicator_columns,
        "region_column": engineer.region_column,
        "farmer_column": engineer.farmer_column,
    }
    if stacking_info is not None:
        metadata["stacking"] = stacking_info
    ensure_directory(MODEL_PATH)
    save_json(metadata, Path(MODEL_PATH) / "metadata.json")


def save_meta_learner(meta_model: object, scaler: object, name: str = "stacked_meta") -> None:
    if meta_model is None or scaler is None:
        return
    ensure_directory(MODEL_PATH)
    try:
        import joblib

        joblib.dump(meta_model, Path(MODEL_PATH) / f"{name}_model.pkl")
        joblib.dump(scaler, Path(MODEL_PATH) / f"{name}_scaler.pkl")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to save meta-learner artifacts: %s", exc)


def run_pipeline() -> Dict[str, float]:
    set_seed(RANDOM_SEED)
    bundle = prepare_datasets()
    train_features, y_encoded, test_features, engineer, label_encoder = build_feature_matrices(bundle)

    assert_no_nan_inf(train_features)
    assert_no_nan_inf(test_features)

    y_train = y_encoded

    model_results = train_models(train_features, y_train, test_features, label_encoder)
    metrics = evaluate_oof(model_results, y_train)

    # Select top-performing models for ensemble (threshold 0.80)
    selected_models = [name for name, score in metrics.items() if score >= 0.80]
    if len(selected_models) < 2:
        # Fallback to top two models by score
        selected_models = sorted(metrics.keys(), key=metrics.get, reverse=True)[:2]

    # For diversity, also include CatBoost and RandomForest if they meet a lower threshold
    diversity_models = [name for name in ["catboost", "random_forest"] if metrics.get(name, 0) >= 0.65]
    selected_models.extend(diversity_models)

    # Remove duplicates and ensure at least 2 models
    selected_models = list(dict.fromkeys(selected_models))
    if len(selected_models) < 2:
        selected_models = sorted(metrics.keys(), key=metrics.get, reverse=True)[:2]

    LOGGER.info("Selected models for ensemble: %s", selected_models)

    # Apply precision-boosting ensemble strategies
    test_predictions_list = [model_results[name].test_predictions for name in selected_models]

    # Use precision-weighted final ensemble instead of basic ensemble
    precision_ensemble_output = create_precision_weighted_final_ensemble(
        model_results, y_train, selected_models, test_predictions_list
    )

    # Set ensemble_output for backward compatibility
    ensemble_output = precision_ensemble_output

    final_oof = precision_ensemble_output.get("oof_predictions")
    final_test_probs = precision_ensemble_output.get("test_predictions")

    # Fallback to basic ensemble if precision ensemble fails
    if final_oof is None or final_test_probs is None:
        LOGGER.warning("Precision ensemble failed, falling back to basic ensemble")
        # Create basic ensemble from individual model predictions
        basic_oof_list = []
        basic_test_list = []

        for model_name in selected_models:
            if model_name in model_results:
                model_result = model_results[model_name]
                if hasattr(model_result, 'oof_predictions') and hasattr(model_result, 'test_predictions'):
                    basic_oof_list.append(model_result.oof_predictions)
                    basic_test_list.append(model_result.test_predictions)

        if basic_oof_list:
            # Simple average for fallback
            final_oof = sum(basic_oof_list) / len(basic_oof_list)
            final_test_probs = sum(basic_test_list) / len(basic_test_list)
            LOGGER.info("Using basic ensemble fallback")
        else:
            raise ValueError("No valid predictions found for ensemble")

    # Use calibrated predictions if available
    if "calibrated_oof" in precision_ensemble_output and "calibrated_test" in precision_ensemble_output:
        final_oof = precision_ensemble_output["calibrated_oof"]
        final_test_probs = precision_ensemble_output["calibrated_test"]
        LOGGER.info("Using calibrated precision ensemble predictions")

    thresholds = optimize_thresholds(final_oof, y_train, n_classes=len(label_encoder.classes_))
    test_predictions = apply_thresholds(final_test_probs, thresholds)

    submission = create_submission_file_from_labels(test_predictions, bundle, label_encoder)
    save_models(model_results)
    save_meta_learner(ensemble_output.get("meta_model"), ensemble_output.get("meta_scaler"))

    ensure_directory(MODEL_PATH)
    if final_oof is not None:
        oof_path = Path(MODEL_PATH) / "final_oof_predictions.npy"
        np.save(oof_path, final_oof)
        LOGGER.info("Saved final OOF probabilities to %s", oof_path)
    y_path = Path(MODEL_PATH) / "y_train.npy"
    np.save(y_path, y_train)
    LOGGER.info("Saved training labels to %s", y_path)

    stacking_info: Optional[Dict[str, object]] = None
    meta_model = ensemble_output.get("meta_model")
    meta_scaler = ensemble_output.get("meta_scaler")
    if meta_model is not None and meta_scaler is not None:
        stacking_info = {
            "meta_model": meta_model.__class__.__name__,
            "solver": getattr(meta_model, "solver", None),
            "class_weight": getattr(meta_model, "class_weight", None),
            "coef": meta_model.coef_.tolist() if hasattr(meta_model, "coef_") else None,
            "intercept": meta_model.intercept_.tolist() if hasattr(meta_model, "intercept_") else None,
            "scaler_mean": meta_scaler.mean_.tolist() if hasattr(meta_scaler, "mean_") else None,
            "scaler_scale": meta_scaler.scale_.tolist() if hasattr(meta_scaler, "scale_") else None,
        }

    log_metadata(
        metrics,
        precision_ensemble_output.get("ensemble_weights", np.array([])).tolist() if hasattr(precision_ensemble_output.get("ensemble_weights", np.array([])), 'tolist') else precision_ensemble_output.get("ensemble_weights", []),
        engineer,
        precision_ensemble_output.get("selected_models", selected_models),
        thresholds,
        stacking_info,
    )

    # Persist label encoder
    ensure_directory(MODEL_PATH)
    label_path = Path(MODEL_PATH) / "label_encoder.npy"
    np.save(label_path, label_encoder.classes_)
    LOGGER.info("Saved label encoder classes to %s", label_path)

    # Export primary model to ONNX (e.g., LightGBM first model)
    lightgbm_result = model_results.get("lightgbm")
    if lightgbm_result and lightgbm_result.models:
        try:
            export_to_onnx(lightgbm_result.models[0], train_features.iloc[:100], "lightgbm_fold0")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ONNX export failed: %s", exc)

        # Generate SHAP explainability for LightGBM model
        try:
            generate_shap_values(
                lightgbm_result.models[0],
                train_features,
                feature_names=list(train_features.columns),
                model_name="lightgbm_fold0",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SHAP explainability failed: %s", exc)

    # Add final ensemble macro F1 to metrics
    final_predictions = np.argmax(final_test_probs, axis=1)
    final_macro_f1 = f1_score(y_train, final_predictions, average="macro")
    metrics["macro_f1"] = final_macro_f1
    metrics["precision_ensemble_f1"] = precision_ensemble_output.get("macro_f1", final_macro_f1)

    LOGGER.info("Final ensemble macro F1: %.6f", final_macro_f1)

    # Save predictions for optimization (Step 1.1)
    LOGGER.info("Saving predictions for optimization...")
    try:
        # Save best model OOF and test predictions
        if 'lightgbm' in model_results:
            np.save('models/oof_lightgbm.npy', model_results['lightgbm'].oof_predictions)
        if 'xgboost' in model_results:
            np.save('models/oof_xgboost.npy', model_results['xgboost'].oof_predictions)

        # Save test predictions for selected models
        if len(test_predictions_list) > 0 and 'lightgbm' in selected_models:
            lightgbm_idx = selected_models.index('lightgbm')
            if lightgbm_idx < len(test_predictions_list):
                np.save('models/test_lightgbm.npy', test_predictions_list[lightgbm_idx])

        if len(test_predictions_list) > 1 and 'xgboost' in selected_models:
            xgboost_idx = selected_models.index('xgboost')
            if xgboost_idx < len(test_predictions_list):
                np.save('models/test_xgboost.npy', test_predictions_list[xgboost_idx])

        np.save('models/y_encoded.npy', y_train)
        LOGGER.info("✓ Predictions saved for optimization")
    except Exception as e:
        LOGGER.warning("Failed to save predictions for optimization: %s", e)

    return metrics


def create_submission_file_from_labels(
    predicted_labels: np.ndarray,
    bundle: DatasetBundle,
    label_encoder: LabelEncoder,
    filename: str = "final_submission.csv",
) -> pd.DataFrame:
    ensure_directory(SUBMISSION_PATH)
    pred_classes = label_encoder.inverse_transform(predicted_labels)
    submission = pd.DataFrame({
        bundle.id_column: bundle.test[bundle.id_column],
        bundle.target_column: pred_classes,
    })
    submission = submission[bundle.submission_columns]

    sample_path = Path(DATA_PATH) / "SampleSubmission.csv"
    validate_submission_format(submission.copy(), sample_path)
    submission_path = Path(SUBMISSION_PATH) / filename
    submission.to_csv(submission_path, index=False)
    LOGGER.info("Submission saved to %s", submission_path)
    return submission
