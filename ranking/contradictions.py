"""
contradictions.py — Contradiction and consistency checks.
Detects mismatches between claims and evidence, reduces confidence, and adds warnings.
"""
from __future__ import annotations

from typing import Any

from ranking.candidate_model import CandidateProfile
from ranking import config


def check_contradictions(
    profile: CandidateProfile,
    feature_scores: dict[str, Any],
    pow_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Run contradiction and consistency checks on a scored candidate.

    Returns:
        Dict with:
        - penalty: float multiplier (0.0-1.0, where 1.0 = no penalty)
        - warnings: list of warning strings
        - details: dict of check results
    """
    warnings: list[str] = []
    penalty = 1.0

    # ── 1. Title vs. Evidence mismatch ────────────────────────────────────
    title_lower = profile.current_title.lower()
    title_relevant = any(t in title_lower for t in config.RELEVANT_TITLE_TERMS)
    title_irrelevant = any(t in title_lower for t in config.IRRELEVANT_TITLE_TERMS)

    if title_relevant and pow_result["pow_score"] < 0.15:
        warnings.append(
            f"Title '{profile.current_title}' suggests engineering role "
            f"but proof-of-work score is very low ({pow_result['pow_score']:.2f})."
        )
        penalty *= 0.95

    if not title_irrelevant and feature_scores.get("relevance", {}).get("score", 0) > 0.7:
        current_career = [ev for ev in profile.career_events if ev.is_current]
        if current_career:
            current_desc = current_career[0].description.lower()
            ml_in_desc = any(kw in current_desc for kw in ["ml", "machine learning", "ai", "model", "retrieval"])
            if not ml_in_desc and "engineer" not in title_lower:
                warnings.append(
                    "High skill relevance but current role description lacks ML/AI content."
                )
                penalty *= 0.97

    # ── 2. Must-have skills claimed but absent from experience ────────────
    skill_names = set(profile.skill_names_lower)
    career_text = " ".join(ev.description.lower() for ev in profile.career_events)

    claimed_must_haves = [
        kw for kw in config.JD_MUST_HAVE if kw.lower() in skill_names
    ]
    for skill in claimed_must_haves[:5]:
        if skill.lower() not in career_text and skill.lower() not in profile.summary.lower():
            warnings.append(
                f"Skill '{skill}' listed but not found in career descriptions or summary."
            )
            penalty *= 0.99

    # ── 3. Experience inconsistency ───────────────────────────────────────
    total_career_months = sum(ev.duration_months for ev in profile.career_events)
    claimed_years = profile.years_of_experience
    career_years = total_career_months / 12.0

    if claimed_years > 0 and career_years > 0:
        ratio = career_years / claimed_years
        if ratio < 0.5:
            warnings.append(
                f"Claimed {claimed_years:.1f} yrs but career history totals "
                f"~{career_years:.1f} yrs (significant gap)."
            )
            penalty *= 0.96
        elif ratio > 2.0:
            warnings.append(
                f"Career history ({career_years:.1f} yrs) significantly exceeds "
                f"claimed experience ({claimed_years:.1f} yrs)."
            )

    # ── 4. High relevance but missing hard constraints ────────────────────
    relevance_score = feature_scores.get("relevance", {}).get("score", 0)
    must_count = feature_scores.get("relevance", {}).get("must_count", 0)

    if relevance_score > 0.6 and must_count < 3:
        warnings.append(
            f"Relevance score is high ({relevance_score:.2f}) but only "
            f"{must_count} must-have skill matches found."
        )
        penalty *= 0.97

    # ── 5. Seniority vs. experience mismatch ──────────────────────────────
    if "senior" in title_lower and claimed_years < 3.0:
        warnings.append(
            f"Title says 'Senior' but only {claimed_years:.1f} years of experience."
        )
        penalty *= 0.95

    # Floor the penalty
    penalty = max(penalty, 0.80)

    return {
        "penalty": round(penalty, 4),
        "warnings": warnings,
        "warning_count": len(warnings),
        "details": {
            "title_relevant": title_relevant,
            "title_irrelevant": title_irrelevant,
            "claimed_must_haves_in_skills": len(claimed_must_haves),
            "career_vs_claimed_ratio": round(career_years / max(claimed_years, 0.1), 2),
        },
    }
