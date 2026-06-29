"""
rerank.py -- Cross-encoder reranking stage.
Reranks top-N retrieved candidates using a lightweight cross-encoder model.
Falls back gracefully to pass-through if dependencies unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# -- Model paths --
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_DIR = _PROJECT_ROOT / "models" / "cross-encoder-ms-marco"
_HUB_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def _load_cross_encoder():
    """Lazy-load the cross-encoder model from local dir or hub."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        if _LOCAL_MODEL_DIR.exists():
            model_path = str(_LOCAL_MODEL_DIR)
            logger.info(f"Loading cross-encoder from local: {model_path}")
            _cross_encoder = CrossEncoder(model_path, device="cpu",
                                           local_files_only=True)
        else:
            logger.info(f"Local model not found, loading from hub: {_HUB_MODEL_NAME}")
            _cross_encoder = CrossEncoder(_HUB_MODEL_NAME, device="cpu")
        logger.info("Cross-encoder loaded successfully.")
        return _cross_encoder
    except ImportError:
        logger.warning("sentence-transformers not installed. Cross-encoder reranking disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cross-encoder: {e}. Reranking disabled.")
        return None


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    candidate_texts: dict[str, str],
    top_n: int = 100,
    blend_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Rerank candidates using a cross-encoder.

    Args:
        query: The JD query text.
        candidates: List of candidate dicts with at least 'candidate_id' and some score.
        candidate_texts: Map of candidate_id -> full text for cross-encoder input.
        top_n: Number of candidates to rerank (performance budget).
        blend_weight: Weight for cross-encoder score in final blend (0-1).

    Returns:
        Reranked candidate list with 'rerank_score' and blended 'final_retrieval_score'.
    """
    encoder = _load_cross_encoder()

    # Limit to top_n for cross-encoder (expensive)
    to_rerank = candidates[:top_n]
    passthrough = candidates[top_n:]

    if encoder is None:
        # Fallback: return as-is with passthrough rerank_score = hybrid_score
        for c in to_rerank:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
            c["final_retrieval_score"] = c.get("hybrid_score", 0.0)
        for c in passthrough:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
            c["final_retrieval_score"] = c.get("hybrid_score", 0.0)
        return to_rerank + passthrough

    # Build pairs
    pairs: list[tuple[str, str]] = []
    valid_indices: list[int] = []
    for i, c in enumerate(to_rerank):
        cid = c["candidate_id"]
        text = candidate_texts.get(cid, "")
        if text:
            # Truncate to avoid exceeding model context
            pairs.append((query[:512], text[:512]))
            valid_indices.append(i)

    if not pairs:
        for c in to_rerank:
            c["rerank_score"] = c.get("hybrid_score", 0.0)
            c["final_retrieval_score"] = c.get("hybrid_score", 0.0)
        return to_rerank + passthrough

    # Score with cross-encoder
    logger.info(f"Cross-encoder reranking {len(pairs)} candidates...")
    # Pin PyTorch determinism for reproducible cross-encoder scores
    try:
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(1)
    except ImportError:
        pass

    raw_scores = encoder.predict(pairs)

    # Normalize cross-encoder scores to [0, 1]
    ce_min = float(min(raw_scores))
    ce_max = float(max(raw_scores))
    ce_range = max(ce_max - ce_min, 1e-9)

    ce_scores: dict[int, float] = {}
    for pair_idx, cand_idx in enumerate(valid_indices):
        ce_scores[cand_idx] = (float(raw_scores[pair_idx]) - ce_min) / ce_range

    # Blend with hybrid score
    for i, c in enumerate(to_rerank):
        hybrid = c.get("hybrid_score", 0.0)
        ce = ce_scores.get(i, 0.0)
        c["rerank_score"] = round(ce, 4)
        c["final_retrieval_score"] = round(
            (1 - blend_weight) * hybrid + blend_weight * ce, 4
        )

    for c in passthrough:
        c["rerank_score"] = 0.0
        c["final_retrieval_score"] = c.get("hybrid_score", 0.0) * 0.8

    # Re-sort by blended score
    all_candidates = to_rerank + passthrough
    all_candidates.sort(key=lambda x: (-x.get("final_retrieval_score", 0), x["candidate_id"]))
    return all_candidates
