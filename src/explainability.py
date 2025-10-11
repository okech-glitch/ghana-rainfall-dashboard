"""Explainability utilities for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:  # noqa: BLE001
    shap = None  # type: ignore[assignment]

from .config import EXPLAIN_PATH
from .utils import configure_logging, ensure_directory

LOGGER = configure_logging()


def generate_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
    feature_names: Optional[List[str]] = None,
    save_summary_plot: bool = True,
    model_name: str = "model",
) -> Dict[str, np.ndarray]:
    """Generate SHAP values for a tree-based model and optionally save summary plot."""
    if shap is None:
        raise ImportError("shap is required for explainability. Please install shap.")

    ensure_directory(EXPLAIN_PATH)

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X.copy()

    if feature_names is None:
        feature_names = list(X_sample.columns)

    if hasattr(model, "predict_proba"):
        tree_model = model
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_sample)
    elif model.__class__.__name__ == "Booster":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        raise TypeError(f"Unsupported model type for SHAP: {type(model)}")

    if save_summary_plot:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plot_path = Path(EXPLAIN_PATH) / f"{model_name}_shap_summary.png"
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        LOGGER.info("Saved SHAP summary plot to %s", plot_path)

    data = {"feature_names": np.array(feature_names), "shap_values": np.array(shap_values)}
    shap_path = Path(EXPLAIN_PATH) / f"{model_name}_shap_values.npy"
    np.save(shap_path, data, allow_pickle=True)
    LOGGER.info("Saved SHAP values to %s", shap_path)
    return data
