"""
xgb_features.py — Build a stable numeric feature vector from existing
extracted features for XGBoost input.

Converts the dict-of-dicts output from feature_engineering.extract_all_features()
into a flat, ordered list of floats with explicit feature names.
"""
from __future__ import annotations

from typing import Any

# Ordered list of feature names — this defines the XGBoost input schema.
# NEVER reorder or remove features without retraining the model.
FEATURE_NAMES: list[str] = [
    # Relevance (7)
    "rel_score",
    "must_count",
    "nice_count",
    "must_count_norm",       # must_count / 14 (max possible)
    "nice_count_norm",       # nice_count / 20 (max possible)
    "title_relevant",        # 1/0
    "title_irrelevant",      # 1/0

    # Production (7)
    "prod_score",
    "prod_signal_count",
    "prod_signal_norm",      # prod_signal_count / 10
    "exp_years",
    "exp_years_norm",        # exp_years / 15
    "tier1",                 # 1/0
    "tier2",                 # 1/0

    # Technical (6)
    "tech_score",
    "retrieval_count",
    "retrieval_count_norm",  # retrieval_count / 10
    "eval_count",
    "python_count",
    "python_count_norm",     # python_count / 8

    # Behavioral (7)
    "beh_score",
    "open_to_work",          # 1/0
    "recency_score",
    "response_rate",         # 0-100
    "response_time_h",
    "notice_days",
    "github_score",

    # Career history (3)
    "chr_score",
    "chr_relevant_count",
    "is_career_pivot",       # 1/0

    # Negative signals (5)
    "neg_multiplier",
    "flag_no_production",    # 1/0
    "flag_irrelevant_role",  # 1/0
    "flag_wrapper_only",     # 1/0
    "flag_under_exp",        # 1/0

    # Meta (1)
    "rule_final_score",      # current rule-based final score
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert any value to float."""
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def build_feature_vector(features: dict[str, Any]) -> list[float]:
    """
    Convert extracted features dict into a flat numeric vector.

    Args:
        features: output of extract_all_features() + compute_final_score()

    Returns:
        List of floats in FEATURE_NAMES order
    """
    rel = features.get("relevance", {})
    prod = features.get("production", {})
    tech = features.get("technical", {})
    beh = features.get("behavioral", {})
    chr_ = features.get("career_history_relevance", {})
    neg = features.get("negative", {})
    neg_flags = neg.get("flags", [])

    must_count = _safe_float(rel.get("must_count", 0))
    nice_count = _safe_float(rel.get("nice_count", 0))
    prod_signal_count = _safe_float(prod.get("prod_signal_count", 0))
    exp_years = _safe_float(prod.get("exp_years", 0))
    retrieval_count = _safe_float(tech.get("retrieval_count", 0))
    python_count = _safe_float(tech.get("python_count", 0))

    vector = [
        # Relevance
        _safe_float(rel.get("score", 0)),
        must_count,
        nice_count,
        _clamp(must_count / 14.0),
        _clamp(nice_count / 20.0),
        1.0 if rel.get("title_relevant") else 0.0,
        1.0 if any(t in (features.get("negative", {}).get("flags", []))
                    for t in ["irrelevant_current_role", "irrelevant_current_role_mild"]) else 0.0,

        # Production
        _safe_float(prod.get("score", 0)),
        prod_signal_count,
        _clamp(prod_signal_count / 10.0),
        exp_years,
        _clamp(exp_years / 15.0),
        1.0 if prod.get("tier1") else 0.0,
        1.0 if prod.get("tier2") else 0.0,

        # Technical
        _safe_float(tech.get("score", 0)),
        retrieval_count,
        _clamp(retrieval_count / 10.0),
        _safe_float(tech.get("eval_count", 0)),
        python_count,
        _clamp(python_count / 8.0),

        # Behavioral
        _safe_float(beh.get("score", 0)),
        1.0 if beh.get("open_to_work") else 0.0,
        _safe_float(beh.get("recency_score", 0.3)),
        _safe_float(beh.get("response_rate", 50)),
        _safe_float(beh.get("response_time_h", 12)),
        _safe_float(beh.get("notice_days", 60)),
        _safe_float(beh.get("github", 40)),

        # Career history
        _safe_float(chr_.get("score", 0)),
        _safe_float(chr_.get("career_history_relevant_count", 0)),
        1.0 if chr_.get("is_career_pivot") else 0.0,

        # Negative signals
        _safe_float(neg.get("multiplier", 1.0)),
        1.0 if "no_production_signals" in neg_flags else 0.0,
        1.0 if "irrelevant_current_role" in neg_flags else 0.0,
        1.0 if "api_wrapper_only" in neg_flags else 0.0,
        1.0 if "under_experienced" in neg_flags else 0.0,

        # Meta
        _safe_float(features.get("final_score", 0.0)),
    ]

    assert len(vector) == NUM_FEATURES, (
        f"Feature vector length mismatch: got {len(vector)}, expected {NUM_FEATURES}"
    )
    return vector
