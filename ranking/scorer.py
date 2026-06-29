"""
scorer.py — Weighted combination of sub-scores + negative multiplier.
Applies the fixed weights from config.py.
Sorting/Tie-breaking is now handled inside rank.py via a custom heap.
"""
from __future__ import annotations
from typing import Any, Dict
from ranking import config

def compute_final_score(features: Dict[str, Any]) -> Dict[str, Any]:
    rel = features["relevance"]["score"]
    prod = features["production"]["score"]
    tech = features["technical"]["score"]
    beh = features["behavioral"]["score"]

    neg = features["negative"]["multiplier"]

    # Career history relevance bonus (up to 0.05)
    # This rewards candidates with strong engineering career history,
    # even if their current title is non-engineering (career pivot).
    chr_score = features.get("career_history_relevance", {}).get("score", 0.0)
    chr_bonus = chr_score * 0.05

    raw = (
        config.WEIGHTS["relevance"] * rel
        + config.WEIGHTS["production"] * prod
        + config.WEIGHTS["technical"] * tech
        + config.WEIGHTS["behavioral"] * beh
        + chr_bonus
    )

    final = raw * neg

    # Leave as float in 0.0-1.0 range since output doesn't require * 100
    final = max(0.0, min(1.0, final))
    
    # We round to 4 decimal places for precision in scoring, 
    # reducing ties, and matching the validation script float structure.
    features["final_score"] = round(final, 4)
    return features
