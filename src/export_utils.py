"""Model export utilities for Ghana Indigenous Intel pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import EXPORT_PATH
from .utils import configure_logging, ensure_directory

LOGGER = configure_logging()


def export_to_onnx(model: object, X_sample: pd.DataFrame, output_name: str) -> Path:
    """Export supported models to ONNX format."""
    ensure_directory(EXPORT_PATH)
    onnx_path = Path(EXPORT_PATH) / f"{output_name}.onnx"

    module_name = model.__class__.__module__
    if module_name.startswith("catboost"):
        _export_catboost(model, onnx_path)
        return onnx_path

    if module_name.startswith("sklearn"):
        _export_sklearn(model, X_sample, onnx_path)
        return onnx_path

    raise TypeError(f"Model type '{model.__class__.__name__}' not supported for ONNX export.")


def _export_catboost(model, onnx_path: Path) -> None:
    try:
        model.save_model(onnx_path, format="onnx")
        LOGGER.info("CatBoost model exported to %s", onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to export CatBoost model: {exc}") from exc


def _export_sklearn(model, X_sample: pd.DataFrame, onnx_path: Path) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:  # noqa: BLE001
        raise ImportError("skl2onnx is required to export sklearn models to ONNX.") from exc

    if not isinstance(X_sample, pd.DataFrame):
        raise TypeError("X_sample must be a pandas DataFrame for sklearn export.")

    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    model_float = model
    sample_array = X_sample.astype(np.float32)

    try:
        onnx_model = convert_sklearn(model_float, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        LOGGER.info("Sklearn model exported to %s", onnx_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to export sklearn model: {exc}") from exc
