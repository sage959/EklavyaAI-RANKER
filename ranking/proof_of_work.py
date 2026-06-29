"""
proof_of_work.py — Evidence-based proof-of-work / system-building evidence scoring.
Scores candidates on concrete evidence of shipping, building, and operating real systems.
Returns both numeric scores and evidence hits for grounding.
"""
from __future__ import annotations

from typing import Any

from ranking.candidate_model import CandidateProfile

# ── Evidence categories ──────────────────────────────────────────────────────

SHIPPING_EVIDENCE = [
    "shipped", "launched", "deployed", "released", "went live", "production",
    "live system", "live service", "real-time", "user-facing",
]

SCALE_EVIDENCE = [
    "million", "billion", "at scale", "qps", "queries per second", "rps",
    "requests per second", "throughput", "latency", "p99", "p95", "sla",
    "daily active", "mau", "concurrent", "high-traffic",
]

INFRA_EVIDENCE = [
    "api endpoint", "microservice", "pipeline", "serving", "inference",
    "kubernetes", "docker", "ci/cd", "monitoring", "alerting", "observability",
]

RETRIEVAL_INFRA_EVIDENCE = [
    "faiss", "elasticsearch", "opensearch", "solr", "lucene", "vector database",
    "vector store", "indexing", "index build", "reranking", "ranking pipeline",
    "search infrastructure", "retrieval pipeline",
]

EVAL_EVIDENCE = [
    "a/b test", "ab testing", "offline eval", "online eval", "ndcg", "mrr",
    "precision@", "recall@", "metrics", "evaluation framework", "benchmark",
    "hit rate", "relevance metric",
]

OPEN_SOURCE_EVIDENCE = [
    "open source", "open-source", "github", "contribution", "maintainer",
    "published", "paper", "arxiv",
]


def score_proof_of_work(profile: CandidateProfile) -> dict[str, Any]:
    """
    Score a candidate's proof-of-work based on concrete evidence in their profile.

    Returns:
        Dict with:
        - pow_score: float 0.0-1.0
        - evidence_hits: dict of category -> list of matched terms
        - evidence_summary: str
    """
    # Collect all career descriptions
    career_text = " ".join(
        ev.description.lower() for ev in profile.career_events if ev.description
    )
    full_text = profile.full_text

    # Score each category
    categories: dict[str, tuple[list[str], float]] = {
        "shipping": (SHIPPING_EVIDENCE, 0.25),
        "scale": (SCALE_EVIDENCE, 0.20),
        "infrastructure": (INFRA_EVIDENCE, 0.15),
        "retrieval_infra": (RETRIEVAL_INFRA_EVIDENCE, 0.20),
        "evaluation": (EVAL_EVIDENCE, 0.10),
        "open_source": (OPEN_SOURCE_EVIDENCE, 0.10),
    }

    evidence_hits: dict[str, list[str]] = {}
    category_scores: dict[str, float] = {}
    total_score = 0.0

    for cat_name, (terms, weight) in categories.items():
        matched = [t for t in terms if t in career_text or t in full_text]
        evidence_hits[cat_name] = matched

        # Diminishing returns: first few matches count more
        if len(matched) == 0:
            cat_score = 0.0
        elif len(matched) <= 2:
            cat_score = 0.4 * len(matched) / 2
        elif len(matched) <= 5:
            cat_score = 0.4 + 0.4 * (len(matched) - 2) / 3
        else:
            cat_score = 0.8 + 0.2 * min(1.0, (len(matched) - 5) / 5)

        cat_score = min(1.0, cat_score)
        category_scores[cat_name] = round(cat_score, 3)
        total_score += cat_score * weight

    total_score = min(1.0, total_score)

    # Build evidence summary
    summary_parts: list[str] = []
    if evidence_hits.get("shipping"):
        summary_parts.append(f"shipped ({len(evidence_hits['shipping'])} signals)")
    if evidence_hits.get("scale"):
        summary_parts.append(f"scale ({len(evidence_hits['scale'])} signals)")
    if evidence_hits.get("retrieval_infra"):
        summary_parts.append(f"retrieval infra ({len(evidence_hits['retrieval_infra'])} signals)")
    if evidence_hits.get("evaluation"):
        summary_parts.append(f"eval rigor ({len(evidence_hits['evaluation'])} signals)")
    if not summary_parts:
        summary_parts.append("no concrete production evidence found")

    return {
        "pow_score": round(total_score, 4),
        "category_scores": category_scores,
        "evidence_hits": {k: v[:5] for k, v in evidence_hits.items()},
        "evidence_summary": " · ".join(summary_parts),
        "total_evidence_count": sum(len(v) for v in evidence_hits.values()),
    }
