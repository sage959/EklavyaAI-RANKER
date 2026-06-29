"""
xgb_train.py — Offline XGBoost training script.

Builds a training dataset from all candidates using the existing feature
extraction pipeline, then trains an XGBRegressor using pseudo-labels
derived from the current rule-based scoring system.

Usage:
    python -m ranking.xgb_train --data data/candidates.jsonl
    python -m ranking.xgb_train --data data/candidates.jsonl --output models/xgb_ranker.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking import feature_engineering, config
from ranking.scorer import compute_final_score
from ranking.xgb_features import FEATURE_NAMES, NUM_FEATURES, build_feature_vector
from ranking.proof_of_work import score_proof_of_work
from ranking.candidate_model import build_candidate_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_candidates(data_path: str) -> list[dict]:
    """Load raw candidate data from JSON or JSONL file."""
    p = Path(data_path)
    candidates = []

    if p.suffix == ".jsonl":
        # JSONL: one JSON object per line
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        # JSON: array or {"candidates": [...]}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else data.get("candidates", [])

    logger.info(f"Loaded {len(candidates)} candidates from {data_path}")
    return candidates


def _compute_sharpened_target(features: dict, pow_result: dict) -> float:
    """
    Compute a sharpened pseudo-label from rule-based signals.

    This goes beyond the simple final_score by amplifying strong signals
    and penalising weak ones, giving XGBoost a richer target to learn from.
    """
    rel = features.get("relevance", {})
    prod = features.get("production", {})
    tech = features.get("technical", {})
    beh = features.get("behavioral", {})
    neg = features.get("negative", {})
    chr_ = features.get("career_history_relevance", {})

    # Base: current rule-based score
    base = features.get("final_score", 0.0)

    # Sharpen with strong signals
    must_count = rel.get("must_count", 0)
    must_bonus = min(0.10, must_count * 0.015) if must_count >= 5 else 0.0

    ret_count = tech.get("retrieval_count", 0)
    tech_bonus = min(0.08, ret_count * 0.012) if ret_count >= 3 else 0.0

    prod_signals = prod.get("prod_signal_count", 0)
    prod_bonus = min(0.06, prod_signals * 0.01) if prod_signals >= 4 else 0.0

    pow_bonus = pow_result.get("pow_score", 0) * 0.05

    # Penalties from neg flags
    neg_flags = neg.get("flags", [])
    neg_penalty = 0.0
    if "no_production_signals" in neg_flags:
        neg_penalty += 0.03
    if "irrelevant_current_role" in neg_flags:
        neg_penalty += 0.04
    if "api_wrapper_only" in neg_flags:
        neg_penalty += 0.02
    if "under_experienced" in neg_flags:
        neg_penalty += 0.03

    # Career depth bonus
    chr_count = chr_.get("career_history_relevant_count", 0)
    chr_bonus = min(0.04, chr_count * 0.015)

    target = base + must_bonus + tech_bonus + prod_bonus + pow_bonus + chr_bonus - neg_penalty
    return max(0.0, min(1.0, target))


def build_training_data(candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and sharpened target vector y from candidates.

    Returns:
        (X, y) where X is (N, NUM_FEATURES) and y is (N,)
    """
    X_list: list[list[float]] = []
    y_list: list[float] = []
    skipped = 0

    for raw in candidates:
        try:
            features = feature_engineering.extract_all_features(raw)
            features = compute_final_score(features)

            profile = build_candidate_profile(raw)
            pow_result = score_proof_of_work(profile)

            vector = build_feature_vector(features)
            target = _compute_sharpened_target(features, pow_result)

            X_list.append(vector)
            y_list.append(target)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning(f"Skipped candidate: {e}")

    if skipped > 0:
        logger.info(f"Skipped {skipped} candidates due to errors.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info(f"Built training data: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, output_path: str) -> None:
    """Train an XGBRegressor and save to disk."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.error("xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)

    logger.info("Training XGBRegressor...")

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y, verbose=True)

    # Save model
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    logger.info(f"Model saved to {out}")

    # Log feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top 10 feature importances:")
    for i in sorted_idx[:10]:
        logger.info(f"  {FEATURE_NAMES[i]:30s} = {importances[i]:.4f}")

    # Log basic stats
    preds = model.predict(X)
    residuals = y - preds
    logger.info(f"Training MAE: {np.mean(np.abs(residuals)):.4f}")
    logger.info(f"Training R²:  {1 - np.var(residuals) / np.var(y):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost ranking model")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to candidate data file (.json or .jsonl)",
    )
    parser.add_argument(
        "--output",
        default="models/xgb_ranker.json",
        help="Path to save the trained model",
    )
    args = parser.parse_args()

    # Load candidates — auto-discover data file if not specified
    if args.data:
        data_path = Path(args.data)
    else:
        # Search common locations
        project_root = Path(__file__).parent.parent
        search_paths = [
            project_root / "data" / "candidates.jsonl",
            project_root / "data" / "candidate_data.json",
            project_root / "data" / "candidate_data.jsonl",
        ]
        data_path = None
        for sp in search_paths:
            if sp.exists():
                data_path = sp
                break
        if data_path is None:
            logger.error("No data file found. Specify --data <path_to_candidates.jsonl>")
            sys.exit(1)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    candidates = _load_candidates(str(data_path))

    if len(candidates) < 10:
        logger.error(f"Too few candidates ({len(candidates)}) for training.")
        sys.exit(1)

    # Build training data
    X, y = build_training_data(candidates)

    # Train and save
    train_model(X, y, args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
