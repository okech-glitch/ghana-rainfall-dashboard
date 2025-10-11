"""
Breakthrough Pipeline for 0.97+ F1 Score
Ghana Indigenous Intel Rainfall Prediction

Strategy: Focus on LightGBM + XGBoost (already 0.89+ F1)
and optimize them to reach 0.97+ through:
1. Advanced feature engineering
2. Optimized hyperparameters
3. Precision-focused ensemble
4. Threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize, differential_evolution
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ==================== STEP 1: ADVANCED FEATURE ENGINEERING ====================

def create_advanced_features(df, target_col=None, is_train=True):
    """
    Advanced feature engineering specifically for 0.97+ F1
    """
    df = df.copy()

    # Identify indicator columns
    indicator_keywords = ['sun', 'cloud', 'wind', 'moon', 'heat', 'tree',
                          'bird', 'animal', 'star', 'indicator', 'rain', 'humidity']

    indicator_cols = [col for col in df.columns
                      if any(kw in col.lower() for kw in indicator_keywords)]

    logger.info(f"Found {len(indicator_cols)} indicator columns")

    # 1. CRITICAL: All pairwise interactions (top combinations)
    numeric_indicators = [col for col in indicator_cols
                         if pd.api.types.is_numeric_dtype(df[col])]

    if len(numeric_indicators) >= 2:
        # Only create interactions for top correlated features
        for i in range(min(5, len(numeric_indicators))):
            for j in range(i+1, min(5, len(numeric_indicators))):
                col1, col2 = numeric_indicators[i], numeric_indicators[j]

                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

    # 2. Aggregation features
    if numeric_indicators:
        df['indicator_mean'] = df[numeric_indicators].mean(axis=1)
        df['indicator_std'] = df[numeric_indicators].std(axis=1)
        df['indicator_max'] = df[numeric_indicators].max(axis=1)
        df['indicator_min'] = df[numeric_indicators].min(axis=1)
        df['indicator_range'] = df['indicator_max'] - df['indicator_min']
        df['indicator_cv'] = df['indicator_std'] / (df['indicator_mean'] + 1e-5)

    # 3. Count-based features
    df['num_indicators'] = df[indicator_cols].notna().sum(axis=1)
    df['num_strong_indicators'] = (df[numeric_indicators] >
                                    df[numeric_indicators].median()).sum(axis=1) if numeric_indicators else 0

    # 4. Temporal features (if exist)
    time_cols = [col for col in df.columns if any(x in col.lower()
                 for x in ['hour', 'time', 'day', 'month'])]

    for col in time_cols:
        if 'hour' in col.lower() and df[col].dtype in [np.int64, np.float64]:
            # Cyclical encoding
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / 24)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / 24)

    # 5. Regional patterns (if region exists)
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)

    # 6. Missing value indicators (crucial for tree models)
    for col in indicator_cols:
        if df[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)

    return df


# ==================== STEP 2: OPTIMIZED LIGHTGBM ====================

def train_optimized_lgbm(X_train, y_train, X_val, y_val):
    """
    LightGBM with parameters optimized for 0.97+ F1
    """
    # Calculate class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

    # Sample weights
    sample_weights = np.array([weight_dict[y] for y in y_train])

    # Optimized parameters for minority class precision
    params = {
        'objective': 'multiclass',
        'num_class': len(classes),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,  # Lower for better generalization
        'num_leaves': 255,       # Higher capacity
        'max_depth': 10,
        'min_child_samples': 3,  # Allow smaller leaves for minority classes
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_split_gain': 0.01,
        'min_child_weight': 0.001,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'force_col_wise': True,
        'extra_trees': False,  # Turn off for more precision
    }

    lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # Train with early stopping
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(250),  # More patience
            lgb.log_evaluation(200)
        ]
    )

    return model


# ==================== STEP 3: OPTIMIZED XGBOOST ====================

def train_optimized_xgb(X_train, y_train, X_val, y_val):
    """
    XGBoost with parameters optimized for 0.97+ F1
    """
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    sample_weights = np.array([weight_dict[y] for y in y_train])

    params = {
        'objective': 'multi:softprob',
        'num_class': len(classes),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.005,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'tree_method': 'hist',
        'seed': 42,
        'n_jobs': -1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=250,
        verbose_eval=200
    )

    return model


# ==================== STEP 4: PRECISION-FOCUSED ENSEMBLE ====================

def create_precision_focused_ensemble(oof_lgbm, oof_xgb, test_lgbm, test_xgb, y_true):
    """
    Only use top 2 models (LightGBM + XGBoost) with precision optimization
    """
    def objective(weights):
        w_lgbm, w_xgb = weights
        w_lgbm = abs(w_lgbm) / (abs(w_lgbm) + abs(w_xgb))
        w_xgb = 1 - w_lgbm

        ensemble = w_lgbm * oof_lgbm + w_xgb * oof_xgb
        preds = np.argmax(ensemble, axis=1)

        return -f1_score(y_true, preds, average='macro')

    # Optimize weights
    result = minimize(objective, x0=[0.5, 0.5], method='Nelder-Mead')

    w_lgbm = abs(result.x[0]) / (abs(result.x[0]) + abs(result.x[1]))
    w_xgb = 1 - w_lgbm

    logger.info(f"Optimized weights: LightGBM={w_lgbm:.4f}, XGBoost={w_xgb:.4f}")

    # Create ensemble
    ensemble_oof = w_lgbm * oof_lgbm + w_xgb * oof_xgb
    ensemble_test = w_lgbm * test_lgbm + w_xgb * test_xgb

    ensemble_f1 = f1_score(y_train, np.argmax(ensemble_oof, axis=1), average='macro')
    logger.info(f"Ensemble F1: {ensemble_f1:.6f}")

    return ensemble_oof, ensemble_test, (w_lgbm, w_xgb)


# ==================== STEP 5: THRESHOLD OPTIMIZATION ====================

def optimize_decision_thresholds(oof_probs, y_true):
    """
    Optimize decision thresholds for maximum F1
    """
    def objective(adjustments):
        adjusted = oof_probs.copy()
        for i in range(len(adjustments)):
            adjusted[:, i] += adjustments[i]

        preds = np.argmax(adjusted, axis=1)
        return -f1_score(y_true, preds, average='macro')

    # Optimize
    bounds = [(-0.3, 0.3)] * oof_probs.shape[1]
    result = differential_evolution(objective, bounds, seed=42, maxiter=100)

    optimal_adjustments = result.x

    # Apply to OOF
    adjusted_oof = oof_probs.copy()
    for i in range(len(optimal_adjustments)):
        adjusted_oof[:, i] += optimal_adjustments[i]

    adjusted_f1 = f1_score(y_true, np.argmax(adjusted_oof, axis=1), average='macro')
    logger.info(f"After threshold optimization: {adjusted_f1:.6f}")
    logger.info(f"Optimal adjustments: {optimal_adjustments}")

    return optimal_adjustments, adjusted_oof


# ==================== MAIN BREAKTHROUGH PIPELINE ====================

def run_breakthrough_pipeline(X_train, y_train, X_test, test_ids, n_folds=10):
    """
    Complete breakthrough pipeline for 0.97+ F1
    """
    logger.info("="*80)
    logger.info("BREAKTHROUGH PIPELINE: TARGET 0.97+ F1")
    logger.info("="*80)

    # Feature engineering
    logger.info("\n[1/5] Advanced feature engineering...")
    X_train_fe = create_advanced_features(X_train, is_train=True)
    X_test_fe = create_advanced_features(X_test, is_train=False)

    # Align columns
    common_cols = list(set(X_train_fe.columns) & set(X_test_fe.columns))
    X_train_fe = X_train_fe[common_cols]
    X_test_fe = X_test_fe[common_cols]

    logger.info(f"Feature count: {len(common_cols)}")

    # Cross-validation
    logger.info("\n[2/5] Training optimized models...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_lgbm = np.zeros((len(X_train_fe), len(np.unique(y_train))))
    oof_xgb = np.zeros((len(X_train_fe), len(np.unique(y_train))))
    test_lgbm = np.zeros((len(X_test_fe), len(np.unique(y_train))))
    test_xgb = np.zeros((len(X_test_fe), len(np.unique(y_train))))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train)):
        logger.info(f"\nFold {fold+1}/{n_folds}")

        X_tr, X_val = X_train_fe.iloc[train_idx], X_train_fe.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # LightGBM
        lgbm_model = train_optimized_lgbm(X_tr, y_tr, X_val, y_val)
        oof_lgbm[val_idx] = lgbm_model.predict(X_val)
        test_lgbm += lgbm_model.predict(X_test_fe) / n_folds

        # XGBoost
        xgb_model = train_optimized_xgb(X_tr, y_tr, X_val, y_val)
        oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
        test_xgb += xgb_model.predict(xgb.DMatrix(X_test_fe)) / n_folds

        # Fold scores
        fold_lgbm_f1 = f1_score(y_val, np.argmax(oof_lgbm[val_idx], axis=1), average='macro')
        fold_xgb_f1 = f1_score(y_val, np.argmax(oof_xgb[val_idx], axis=1), average='macro')
        logger.info(f"  LightGBM: {fold_lgbm_f1:.6f}, XGBoost: {fold_xgb_f1:.6f}")

    # Overall CV scores
    lgbm_cv_f1 = f1_score(y_train, np.argmax(oof_lgbm, axis=1), average='macro')
    xgb_cv_f1 = f1_score(y_train, np.argmax(oof_xgb, axis=1), average='macro')

    logger.info("\n[3/5] Model CV scores:")
    logger.info(f"  LightGBM: {lgbm_cv_f1:.6f}")
    logger.info(f"  XGBoost:  {xgb_cv_f1:.6f}")

    # Ensemble
    logger.info("\n[4/5] Creating precision-focused ensemble...")
    ensemble_oof, ensemble_test, weights = create_precision_focused_ensemble(
        oof_lgbm, oof_xgb, test_lgbm, test_xgb, y_train
    )

    ensemble_f1 = f1_score(y_train, np.argmax(ensemble_oof, axis=1), average='macro')
    logger.info(f"Ensemble F1: {ensemble_f1:.6f}")

    # Threshold optimization
    logger.info("\n[5/5] Optimizing decision thresholds...")
    optimal_adjustments, final_oof = optimize_decision_thresholds(ensemble_oof, y_train)

    # Apply to test
    final_test = ensemble_test.copy()
    for i in range(len(optimal_adjustments)):
        final_test[:, i] += optimal_adjustments[i]

    final_f1 = f1_score(y_train, np.argmax(final_oof, axis=1), average='macro')

    # Results
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Best single model: {max(lgbm_cv_f1, xgb_cv_f1):.6f}")
    logger.info(f"Ensemble:          {ensemble_f1:.6f}")
    logger.info(f"After optimization: {final_f1:.6f}")

    if final_f1 >= 0.97:
        logger.info("\n🎉 TARGET ACHIEVED! F1 >= 0.97 🎉")
    else:
        gap = 0.97 - final_f1
        logger.info(f"\n📊 Gap to target: {gap:.6f}")

    logger.info("\nPer-class performance:")
    final_preds = np.argmax(final_oof, axis=1)
    logger.info("\n" + classification_report(y_train, final_preds, digits=4))

    return final_oof, final_test, {
        'lgbm_f1': lgbm_cv_f1,
        'xgb_f1': xgb_cv_f1,
        'ensemble_f1': ensemble_f1,
        'final_f1': final_f1,
        'weights': weights,
        'adjustments': optimal_adjustments
    }


if __name__ == "__main__":
    # Load your data
    from src.data_loader import load_data, preprocess_data

    logger.info("Loading data...")
    train_df, test_df, sample_submission = load_data()
    bundle = preprocess_data(train_df, test_df, sample_submission)

    # Identify columns
    target_col = bundle.target_column
    id_col = bundle.id_column

    # Prepare data
    X_train = bundle.train.drop([id_col, target_col], axis=1)
    y_train = bundle.train[target_col]
    X_test = bundle.test.drop([id_col], axis=1)
    test_ids = bundle.test[id_col]

    # Encode target
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    logger.info(f"Training classes: {le.classes_}")
    logger.info(f"Class distribution: {np.bincount(y_train_encoded)}")

    # Run breakthrough pipeline
    final_oof, final_test, metrics = run_breakthrough_pipeline(
        X_train, y_train_encoded, X_test, test_ids
    )

    # Create submission
    final_preds = np.argmax(final_test, axis=1)
    final_classes = le.inverse_transform(final_preds)

    submission = pd.DataFrame({
        'ID': test_ids,
        'Target': final_classes
    })

    submission.to_csv('submissions/breakthrough_v1.csv', index=False)
    logger.info(f"\n✓ Submission saved: breakthrough_v1.csv")
    logger.info(f"✓ Final CV F1: {metrics['final_f1']:.6f}")

    # Save detailed results
    results_summary = {
        'final_f1': metrics['final_f1'],
        'lgbm_f1': metrics['lgbm_f1'],
        'xgb_f1': metrics['xgb_f1'],
        'ensemble_f1': metrics['ensemble_f1'],
        'weights': metrics['weights'],
        'adjustments': metrics['adjustments'],
        'target_classes': le.classes_.tolist(),
        'feature_count': len(X_train.columns)
    }

    import json
    with open('models/breakthrough_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info("✓ Results saved: breakthrough_results.json"
        l o g g e r . i n f o ( ' R e s u l t s   s a v e d :   b r e a k t h r o u g h _ r e s u l t s . j s o n ' ) 
    logger.info(" Results saved: breakthrough_results.json\)\n