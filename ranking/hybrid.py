"""
hybrid.py — Hybrid retrieval fusion combining lexical (BM25) and semantic (FAISS) scores.
Uses Reciprocal Rank Fusion (RRF) or weighted score normalisation.
"""
from __future__ import annotations

from typing import Any


def fuse_retrieval_scores(
    lexical_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    lexical_weight: float = 0.45,
    semantic_weight: float = 0.55,
    method: str = "weighted_norm",
) -> list[dict[str, Any]]:
    """
    Fuse lexical and semantic candidate-level retrieval results.

    Args:
        lexical_results: from BM25Index.query_candidates()
        semantic_results: from SemanticIndex.query_candidates()
        lexical_weight: weight for lexical score (0-1)
        semantic_weight: weight for semantic score (0-1)
        method: "weighted_norm" or "rrf"

    Returns:
        Fused results sorted by combined score descending.
    """
    if method == "rrf":
        return _reciprocal_rank_fusion(lexical_results, semantic_results)
    return _weighted_norm_fusion(lexical_results, semantic_results,
                                 lexical_weight, semantic_weight)


def _weighted_norm_fusion(
    lexical: list[dict],
    semantic: list[dict],
    lex_w: float,
    sem_w: float,
) -> list[dict[str, Any]]:
    """Normalise scores to [0, 1] and combine with weights."""
    # Normalize lexical
    lex_by_cid: dict[str, dict] = {}
    lex_scores = [r.get("lexical_score", 0) for r in lexical]
    lex_max = max(lex_scores) if lex_scores else 1.0
    lex_min = min(lex_scores) if lex_scores else 0.0
    lex_range = max(lex_max - lex_min, 1e-9)

    for r in lexical:
        cid = r["candidate_id"]
        norm = (r.get("lexical_score", 0) - lex_min) / lex_range
        lex_by_cid[cid] = {"lexical_norm": round(norm, 4), **r}

    # Normalize semantic
    sem_by_cid: dict[str, dict] = {}
    sem_scores = [r.get("semantic_score", 0) for r in semantic]
    sem_max = max(sem_scores) if sem_scores else 1.0
    sem_min = min(sem_scores) if sem_scores else 0.0
    sem_range = max(sem_max - sem_min, 1e-9)

    for r in semantic:
        cid = r["candidate_id"]
        norm = (r.get("semantic_score", 0) - sem_min) / sem_range
        sem_by_cid[cid] = {"semantic_norm": round(norm, 4), **r}

    # Fuse
    all_cids = sorted(set(lex_by_cid.keys()) | set(sem_by_cid.keys()))
    fused: list[dict[str, Any]] = []

    for cid in all_cids:
        lex_norm = lex_by_cid.get(cid, {}).get("lexical_norm", 0.0)
        sem_norm = sem_by_cid.get(cid, {}).get("semantic_norm", 0.0)

        combined = lex_w * lex_norm + sem_w * sem_norm

        entry: dict[str, Any] = {
            "candidate_id": cid,
            "hybrid_score": round(combined, 4),
            "lexical_norm": lex_norm,
            "semantic_norm": sem_norm,
        }

        # Merge diagnostic chunks from whichever source is present
        if cid in lex_by_cid:
            entry["lexical_chunks"] = lex_by_cid[cid].get("top_chunks", [])[:3]
        if cid in sem_by_cid:
            entry["semantic_chunks"] = sem_by_cid[cid].get("top_chunks", [])[:3]

        fused.append(entry)

    fused.sort(key=lambda x: (-x["hybrid_score"], x["candidate_id"]))
    return fused


def _reciprocal_rank_fusion(
    lexical: list[dict],
    semantic: list[dict],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank_i)) across all lists.
    """
    rrf_scores: dict[str, float] = {}
    rrf_details: dict[str, dict] = {}

    for rank, r in enumerate(lexical, 1):
        cid = r["candidate_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        rrf_details[cid] = {"candidate_id": cid}

    for rank, r in enumerate(semantic, 1):
        cid = r["candidate_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in rrf_details:
            rrf_details[cid] = {"candidate_id": cid}

    fused = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0])):
        entry = rrf_details[cid]
        entry["hybrid_score"] = round(score, 6)
        fused.append(entry)

    return fused
