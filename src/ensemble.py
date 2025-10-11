# src/ensemble.py - FIXED TWO-STAGE FUNCTIONS

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from typing import Sequence
import logging

logger = logging.getLogger(__name__)


def apply_thresholds(probabilities, thresholds):
    """Convert probability matrix to class predictions using thresholds."""
    if probabilities.shape[1] != len(thresholds):
        raise ValueError("Number of thresholds must match number of classes.")

    adjusted = probabilities.copy()
    for class_idx, threshold in enumerate(thresholds):
        adjusted[:, class_idx] = adjusted[:, class_idx] - threshold

    predictions = np.argmax(adjusted, axis=1)
    return predictions


def optimize_thresholds(probabilities, y_true, n_classes):
    """Optimize per-class thresholds for converting probabilities to labels with class-specific constraints."""

    # Initialize thresholds based on class frequency
    class_counts = np.bincount(y_true, minlength=n_classes)
    initial_thresholds = class_counts / class_counts.sum()  # Proportional to class frequency

    def objective(thresholds):
        preds = apply_thresholds(probabilities, thresholds)
        return -f1_score(y_true, preds, average="macro")

    bounds = [(0.1, 0.9)] * n_classes  # Constrain thresholds to avoid extreme values
    result = minimize(objective, initial_thresholds, bounds=bounds, method="Nelder-Mead")

    if not result.success:
        logger.warning("Threshold optimization failed: %s", result.message)
        return initial_thresholds

    thresholds = np.clip(result.x, 0.1, 0.9)  # Ensure thresholds are reasonable
    logger.info("Optimized thresholds: %s", thresholds)
    return thresholds


def weighted_average(predictions_list, weights):
    if len(predictions_list) != len(weights):
        raise ValueError("Length of predictions_list and weights must match.")
    weights = _normalize_weights(weights)
    ensemble = np.zeros_like(predictions_list[0])
    for weight, preds in zip(weights, predictions_list):
        ensemble += weight * preds
    return ensemble


def _normalize_weights(weights):
    weights = np.maximum(weights, 0.0)
    total = np.sum(weights)
    if total == 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def optimize_ensemble_weights(oof_predictions_list, y_true):
    """Optimize ensemble weights by maximizing macro F1 on OOF predictions."""
    if not oof_predictions_list:
        raise ValueError("oof_predictions_list must not be empty.")

    n_models = len(oof_predictions_list)
    initial_weights = np.ones(n_models) / n_models

    def objective(weights):
        normalized = _normalize_weights(weights)
        ensemble_pred = np.zeros_like(oof_predictions_list[0])
        for w, preds in zip(normalized, oof_predictions_list):
            ensemble_pred += w * preds
        pred_labels = np.argmax(ensemble_pred, axis=1)
        return -f1_score(y_true, pred_labels, average="macro")

    bounds = [(0.0, 1.0) for _ in range(n_models)]
    result = minimize(objective, initial_weights, bounds=bounds, method="Nelder-Mead")

    if not result.success:
        logger.warning("Weight optimization failed: %s", result.message)
        return initial_weights

    optimal_weights = _normalize_weights(result.x)
    logger.info("Optimal ensemble weights: %s", optimal_weights)
    return optimal_weights


def train_meta_learner(oof_predictions_list, y_true, test_predictions_list, class_weight="balanced", calibrate=True, calibration_method="isotonic"):
    """Train a stacking meta-learner on model probability outputs with optional calibration."""

    if len(oof_predictions_list) == 1:
        train_preds = oof_predictions_list[0]
        test_preds = test_predictions_list[0]
        result = {
            "model": None,
            "scaler": None,
            "train_predictions": train_preds,
            "test_predictions": test_preds,
        }
    else:
        X_train = _stack_predictions(oof_predictions_list)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test = _stack_predictions(test_predictions_list)
        X_test_scaled = scaler.transform(X_test)

        meta_model = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            solver="lbfgs",
            n_jobs=-1,
        )
        meta_model.fit(X_train_scaled, y_true)

        train_predictions = meta_model.predict_proba(X_train_scaled)
        test_predictions = meta_model.predict_proba(X_test_scaled)

        result = {
            "model": meta_model,
            "scaler": scaler,
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
        }

    # Apply calibration if requested
    if calibrate:
        calibration_result = calibrate_probabilities(
            result["train_predictions"], y_true, result["test_predictions"],
            calibration_method
        )
        result.update(calibration_result)

    return result


def calibrate_probabilities(oof_predictions, y_true, test_predictions, method="isotonic"):
    """Calibrate ensemble probabilities to reduce overconfidence."""
    if method == "isotonic":
        calibrators = []
        calibrated_oof = np.zeros_like(oof_predictions)
        calibrated_test = np.zeros_like(test_predictions)

        for class_idx in range(oof_predictions.shape[1]):
            # Fit isotonic regression for each class
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_predictions[:, class_idx], y_true == class_idx)
            calibrators.append(iso)

            calibrated_oof[:, class_idx] = iso.predict(oof_predictions[:, class_idx])
            calibrated_test[:, class_idx] = iso.predict(test_predictions[:, class_idx])

        # Renormalize probabilities
        calibrated_oof = calibrated_oof / calibrated_oof.sum(axis=1, keepdims=True)
        calibrated_test = calibrated_test / calibrated_test.sum(axis=1, keepdims=True)

        return {
            "calibrators": calibrators,
            "calibrated_oof": calibrated_oof,
            "calibrated_test": calibrated_test,
        }
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def _stack_predictions(predictions_list):
    if not predictions_list:
        raise ValueError("predictions_list must not be empty.")
    return np.concatenate(predictions_list, axis=1)


def train_two_stage_classifier(oof_predictions_list, y_true, test_predictions_list):
    """
    Two-stage classification:
    Stage 1: Binary (NORAIN vs ANY_RAIN)
    Stage 2: Multi-class among rain types (HEAVY, MEDIUM, SMALL)

    FIXED VERSION
    """
    logger.info("="*80)
    logger.info("TWO-STAGE CLASSIFICATION TRAINING")
    logger.info("="*80)

    # Stack OOF predictions as features
    X_train = np.hstack([pred for pred in oof_predictions_list])
    X_test = np.hstack([pred for pred in test_predictions_list])

    # Determine NORAIN class (usually class 2 based on alphabetical order)
    # HEAVYRAIN=0, MEDIUMRAIN=1, NORAIN=2, SMALLRAIN=3
    NORAIN_CLASS = 2

    # Stage 1: Binary classification (NORAIN=0 vs RAIN=1)
    logger.info("Stage 1: Training NORAIN vs RAIN classifier...")
    y_binary = (y_true != NORAIN_CLASS).astype(int)

    stage1_model = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    stage1_model.fit(X_train, y_binary)

    # Stage 1 predictions
    stage1_oof_probs = stage1_model.predict_proba(X_train)[:, 1]  # Prob of RAIN
    stage1_test_probs = stage1_model.predict_proba(X_test)[:, 1]  # Prob of RAIN

    stage1_f1 = f1_score(y_binary, stage1_model.predict(X_train), average='binary')
    logger.info(f"Stage 1 Binary F1: {stage1_f1:.6f}")

    # Stage 2: Among RAIN instances, classify type
    logger.info("Stage 2: Training RAIN type classifier...")
    rain_mask = (y_true != NORAIN_CLASS)

    if rain_mask.sum() > 0:
        X_train_rain = X_train[rain_mask]
        y_train_rain = y_true[rain_mask]

        # Remap labels for rain classes only (0, 1, 3 -> 0, 1, 2)
        y_train_rain_mapped = y_train_rain.copy()
        label_map = {}
        unique_rain_classes = sorted([c for c in np.unique(y_true) if c != NORAIN_CLASS])
        for new_idx, old_idx in enumerate(unique_rain_classes):
            label_map[old_idx] = new_idx
            y_train_rain_mapped[y_train_rain == old_idx] = new_idx

        stage2_model = LogisticRegression(
            multi_class='multinomial',
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        stage2_model.fit(X_train_rain, y_train_rain_mapped)

        # Evaluate stage 2
        stage2_preds = stage2_model.predict(X_train_rain)
        stage2_f1 = f1_score(y_train_rain_mapped, stage2_preds, average='macro')
        logger.info(f"Stage 2 Rain-Type F1: {stage2_f1:.6f}")
    else:
        stage2_model = None
        label_map = {}
        logger.warning("No rain instances found for Stage 2!")

    logger.info("Two-stage classifier training complete.")
    logger.info("="*80)

    return {
        'stage1_model': stage1_model,
        'stage2_model': stage2_model,
        'label_map': label_map,
        'inverse_label_map': {v: k for k, v in label_map.items()},
        'norain_class': NORAIN_CLASS,
        'stage1_oof_probs': stage1_oof_probs,
        'stage1_test_probs': stage1_test_probs
    }


def predict_two_stage(stage1_model, stage2_model, X_stacked, classifier_info):
    """
    Make predictions using two-stage classifier

    FIXED VERSION - handles array dimensions correctly

    Args:
        stage1_model: Binary classifier (NORAIN vs RAIN)
        stage2_model: Multi-class classifier (HEAVY, MEDIUM, SMALL)
        X_stacked: Stacked features (test predictions)
        classifier_info: Dict with label mappings and metadata

    Returns:
        final_probabilities: (N_samples, 4) array with probabilities for all classes
    """
    n_samples = X_stacked.shape[0]
    n_classes = 4  # HEAVY, MEDIUM, NORAIN, SMALL
    NORAIN_CLASS = classifier_info['norain_class']

    # Initialize final predictions
    final_probabilities = np.zeros((n_samples, n_classes))

    # Stage 1: Get RAIN probabilities
    stage1_probs = stage1_model.predict_proba(X_stacked)
    rain_prob = stage1_probs[:, 1]  # Probability of RAIN (not NORAIN)
    norain_prob = stage1_probs[:, 0]  # Probability of NORAIN

    # Assign NORAIN probabilities
    final_probabilities[:, NORAIN_CLASS] = norain_prob

    # Stage 2: Among predicted RAIN, classify type
    if stage2_model is not None:
        # Get rain type probabilities for ALL samples
        stage2_probs = stage2_model.predict_proba(X_stacked)  # Shape: (n_samples, 3)

        # Map stage2 predictions back to original class indices
        inverse_map = classifier_info['inverse_label_map']

        for new_idx in range(stage2_probs.shape[1]):
            old_idx = inverse_map[new_idx]
            # Multiply by rain probability (chain rule)
            final_probabilities[:, old_idx] = stage2_probs[:, new_idx] * rain_prob
    else:
        # If no stage2 model, distribute rain probability equally
        rain_classes = [c for c in range(n_classes) if c != NORAIN_CLASS]
        for cls in rain_classes:
            final_probabilities[:, cls] = rain_prob / len(rain_classes)

    # Normalize probabilities (should sum to 1)
    row_sums = final_probabilities.sum(axis=1, keepdims=True)
    final_probabilities = final_probabilities / (row_sums + 1e-10)

    return final_probabilities


def ensemble_predictions_two_stage(model_results, y_true, selected_models, test_predictions_list):
    """
    Create two-stage ensemble predictions

    FIXED VERSION
    """
    logger.info("="*80)
    logger.info("TWO-STAGE ENSEMBLE PREDICTIONS")
    logger.info("="*80)

    # Collect OOF and test predictions
    oof_list = []
    test_list = []

    for model_name in selected_models:
        if model_name in model_results:
            oof_list.append(model_results[model_name].oof_predictions)

    test_list = test_predictions_list

    # Train two-stage classifier
    two_stage_classifier = train_two_stage_classifier(oof_list, y_true, test_list)

    # Get OOF predictions
    X_train_stacked = np.hstack([pred for pred in oof_list])
    oof_probabilities = predict_two_stage(
        two_stage_classifier["stage1_model"],
        two_stage_classifier["stage2_model"],
        X_train_stacked,
        two_stage_classifier
    )

    # Get test predictions
    X_test_stacked = np.hstack([pred for pred in test_list])
    test_probabilities = predict_two_stage(
        two_stage_classifier["stage1_model"],
        two_stage_classifier["stage2_model"],
        X_test_stacked,
        two_stage_classifier
    )

    # Evaluate
    oof_f1 = f1_score(y_true, np.argmax(oof_probabilities, axis=1), average='macro')
    logger.info(f"Two-stage OOF F1: {oof_f1:.6f}")
    logger.info("="*80)

    return {
        'oof_predictions': oof_probabilities,
        'test_predictions': test_probabilities,
        'two_stage_f1': oof_f1,
        'classifier': two_stage_classifier
    }


def create_precision_weighted_final_ensemble(model_results, y_true, selected_models, test_predictions_list):
    """
    Create final ensemble with multiple strategies

    FIXED VERSION
    """
    logger.info("\n" + "="*80)
    logger.info("PRECISION-WEIGHTED FINAL ENSEMBLE")
    logger.info("="*80)

    # Strategy 1: Traditional weighted ensemble
    logger.info("Creating traditional ensemble...")
    oof_list = []
    for model_name in selected_models:
        if model_name in model_results:
            oof_list.append(model_results[model_name].oof_predictions)

    def objective(weights):
        weights = np.array(weights) / np.sum(weights)
        ensemble = sum([w * pred for w, pred in zip(weights, oof_list)])
        preds = np.argmax(ensemble, axis=1)
        return -f1_score(y_true, preds, average='macro')

    result = minimize(
        objective,
        x0=[1.0] * len(oof_list),
        bounds=[(0.0, 1.0)] * len(oof_list),
        method='Nelder-Mead'
    )

    optimal_weights = result.x / np.sum(result.x)
    logger.info(f"Optimal ensemble weights: {optimal_weights}")

    weighted_oof = sum([w * pred for w, pred in zip(optimal_weights, oof_list)])
    weighted_test = sum([w * pred for w, pred in zip(optimal_weights, test_predictions_list)])
    weighted_f1 = f1_score(y_true, np.argmax(weighted_oof, axis=1), average='macro')

    logger.info(f"Weighted ensemble F1: {weighted_f1:.6f}")

    # Strategy 2: Two-stage ensemble
    logger.info("Creating two-stage ensemble...")
    try:
        two_stage_result = ensemble_predictions_two_stage(
            model_results, y_true, selected_models, test_predictions_list
        )
        two_stage_f1 = two_stage_result['two_stage_f1']
    except Exception as e:
        logger.warning(f"Two-stage ensemble failed: {e}")
        logger.warning("Using weighted ensemble only")
        two_stage_result = None
        two_stage_f1 = 0

    # Choose best strategy
    if two_stage_result and two_stage_f1 > weighted_f1:
        logger.info(f"\n✓ Using two-stage ensemble (F1: {two_stage_f1:.6f})")
        best_oof = two_stage_result['oof_predictions']
        best_test = two_stage_result['test_predictions']
        best_f1 = two_stage_f1
        best_method = 'two_stage'
    else:
        logger.info(f"\n✓ Using weighted ensemble (F1: {weighted_f1:.6f})")
        best_oof = weighted_oof
        best_test = weighted_test
        best_f1 = weighted_f1
        best_method = 'weighted'

    logger.info("="*80)

    return {
        'oof_predictions': best_oof,
        'test_predictions': best_test,
        'ensemble_f1': best_f1,
        'method': best_method,
        'weighted_result': {
            'weights': optimal_weights,
            'f1': weighted_f1
        },
        'two_stage_result': two_stage_result
    }
