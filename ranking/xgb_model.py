"""
xgb_model.py — XGBoost model load + inference.

Loads a pre-trained XGBRegressor from disk and provides inference.
Gracefully falls back if the model is missing or fails to load.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ranking.xgb_features import FEATURE_NAMES, NUM_FEATURES, build_feature_vector

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model: Optional[Any] = None
_model_loaded: bool = False


def _get_default_model_path() -> Path:
    """Return the default model artifact path."""
    return Path(__file__).parent.parent / "models" / "xgb_ranker.json"


def load_model(model_path: str | Path | None = None) -> bool:
    """
    Load the XGBoost model from disk.

    Args:
        model_path: Path to the model file. Uses default if None.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _model, _model_loaded

    if model_path is None:
        model_path = _get_default_model_path()
    model_path = Path(model_path)

    if not model_path.exists():
        logger.info(f"XGBoost model not found at {model_path} — will use rule-based scoring only.")
        _model = None
        _model_loaded = False
        return False

    try:
        from xgboost import XGBRegressor
        _model = XGBRegressor()
        _model.load_model(str(model_path))
        _model_loaded = True
        logger.info(f"XGBoost model loaded from {model_path}")
        return True
    except ImportError:
        logger.warning("xgboost package not installed — falling back to rule-based scoring.")
        _model = None
        _model_loaded = False
        return False
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model from {model_path}: {e}")
        _model = None
        _model_loaded = False
        return False


def is_available() -> bool:
    """Check if the XGBoost model is loaded and ready."""
    return _model_loaded and _model is not None


def predict_score(features: dict[str, Any]) -> float | None:
    """
    Run XGBoost inference on a single candidate's feature dict.

    Args:
        features: output of extract_all_features() + compute_final_score()

    Returns:
        Predicted score (0.0-1.0), or None if model is unavailable.
    """
    if not is_available():
        return None

    try:
        vector = build_feature_vector(features)
        X = np.array([vector], dtype=np.float32)
        pred = _model.predict(X)[0]
        # Clamp to 0-1
        return float(max(0.0, min(1.0, pred)))
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")
        return None


def predict_batch(feature_list: list[dict[str, Any]]) -> list[float | None]:
    """
    Run XGBoost inference on a batch of candidates.

    Args:
        feature_list: list of feature dicts

    Returns:
        List of predicted scores (or None for failures)
    """
    if not is_available():
        return [None] * len(feature_list)

    try:
        vectors = []
        valid_indices = []
        for i, features in enumerate(feature_list):
            try:
                vectors.append(build_feature_vector(features))
                valid_indices.append(i)
            except Exception:
                pass

        if not vectors:
            return [None] * len(feature_list)

        X = np.array(vectors, dtype=np.float32)
        preds = _model.predict(X)

        results: list[float | None] = [None] * len(feature_list)
        for idx, pred in zip(valid_indices, preds):
            results[idx] = float(max(0.0, min(1.0, pred)))
        return results
    except Exception as e:
        logger.warning(f"XGBoost batch prediction failed: {e}")
        return [None] * len(feature_list)
