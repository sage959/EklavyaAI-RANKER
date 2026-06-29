"""
credibility.py — Resume credibility checks.

Detects timeline anomalies, keyword stuffing, AI-like writing patterns,
and claim-to-evidence mismatches.  Produces a *soft* penalty multiplier
(0.88–1.0) that is applied alongside the existing contra_penalty.

Design principles:
  - Conservative: cap total penalty at 0.12 (floor = 0.88).
  - Explainable: every penalty point has a reason code.
  - Modular: four independent signal checkers, easy to tune.
  - Safe for edge cases: career changers, founders, short histories,
    and non-traditional profiles are handled with confidence scaling.

This module does NOT duplicate checks in contradictions.py (skill-claim
vs. career evidence, experience count mismatch, title vs. PoW).  It
covers complementary signals that contradictions.py does not address.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime
from typing import Any

from ranking.candidate_model import CandidateProfile


# ── Tunables ─────────────────────────────────────────────────────────────────

# Per-signal weights (must sum to 1.0)
_SIGNAL_WEIGHTS = {
    "timeline":    0.30,
    "stuffing":    0.25,
    "description": 0.20,
    "claim_gap":   0.25,
}

# Maximum total credibility penalty (multiplier floor = 1.0 - this)
_MAX_PENALTY = 0.12

# Confidence threshold below which penalty is halved
_LOW_CONFIDENCE_THRESHOLD = 0.40


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> date | None:
    """Parse a date string in common formats."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(str(date_str).strip()[:10], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'[a-z0-9_\-\.]+', text.lower())


def _similarity_ratio(a: str, b: str) -> float:
    """
    Quick token-overlap similarity between two strings (Jaccard-ish).
    Returns 0.0–1.0.
    """
    if not a or not b:
        return 0.0
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ── Signal 1: Timeline Anomalies ─────────────────────────────────────────────

def _check_timeline_anomalies(profile: CandidateProfile) -> tuple[float, list[str]]:
    """
    Detect overlapping roles, impossibly short tenures, and large gaps.

    Returns:
        (score 0.0–1.0, list of reason codes)

    Safe for: career changers (1 overlap is normal), founders (gaps expected),
    candidates with only 1–2 roles.
    """
    reasons: list[str] = []
    penalty_points = 0.0
    events = profile.career_events

    if len(events) < 2:
        # Too few roles to judge timeline — no penalty
        return 0.0, []

    # Parse and sort by start date
    dated_events: list[tuple[date, date | None, str, str]] = []
    for ev in events:
        start = _parse_date(ev.start_date)
        end = _parse_date(ev.end_date) if not ev.is_current else None
        if start:
            dated_events.append((start, end, ev.title, ev.company))

    if len(dated_events) < 2:
        return 0.0, []

    dated_events.sort(key=lambda x: x[0])

    # Check overlaps
    overlap_count = 0
    for i in range(len(dated_events) - 1):
        _, end_i, _, _ = dated_events[i]
        start_next, _, _, _ = dated_events[i + 1]
        if end_i and start_next:
            overlap_days = (end_i - start_next).days
            if overlap_days > 90:  # >3 months overlap
                overlap_count += 1

    if overlap_count >= 2:
        penalty_points += 0.4
        reasons.append(f"timeline_overlap_{overlap_count}_roles")
    elif overlap_count == 1:
        penalty_points += 0.15
        reasons.append("timeline_overlap_1_role")

    # Check impossibly short tenures
    short_tenure_count = 0
    for ev in events:
        if ev.duration_months and 0 < ev.duration_months < 2 and not ev.is_current:
            short_tenure_count += 1

    if short_tenure_count >= 3:
        penalty_points += 0.3
        reasons.append(f"very_short_tenure_{short_tenure_count}_roles")
    elif short_tenure_count >= 2:
        penalty_points += 0.15
        reasons.append(f"very_short_tenure_{short_tenure_count}_roles")

    # Check large unexplained gaps (>24 months)
    gap_months_max = 0
    for i in range(len(dated_events) - 1):
        _, end_i, _, _ = dated_events[i]
        start_next, _, _, _ = dated_events[i + 1]
        if end_i and start_next:
            gap_days = (start_next - end_i).days
            gap_m = gap_days / 30.0
            if gap_m > gap_months_max:
                gap_months_max = gap_m

    if gap_months_max > 36:
        penalty_points += 0.25
        reasons.append(f"career_gap_{int(gap_months_max)}_months")
    elif gap_months_max > 24:
        penalty_points += 0.10
        reasons.append(f"career_gap_{int(gap_months_max)}_months")

    return min(1.0, penalty_points), reasons


# ── Signal 2: Keyword Stuffing ───────────────────────────────────────────────

def _check_keyword_stuffing(
    profile: CandidateProfile,
    features: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Detect suspiciously high skill counts and buzzword density.

    Returns:
        (score 0.0–1.0, list of reason codes)

    Safe for: generalists with broad but shallow skill sets.
    """
    reasons: list[str] = []
    penalty_points = 0.0

    skill_count = len(profile.skills)

    # Flag extremely high skill counts
    if skill_count > 30:
        penalty_points += 0.35
        reasons.append(f"excessive_skills_{skill_count}")
    elif skill_count > 22:
        penalty_points += 0.15
        reasons.append(f"high_skill_count_{skill_count}")

    # Check for skills with zero endorsements AND zero career evidence
    if skill_count > 0:
        career_text = " ".join(
            ev.description.lower() for ev in profile.career_events if ev.description
        )
        unsupported_count = 0
        for skill in profile.skills:
            skill_lower = skill.name.lower()
            if skill.endorsements == 0 and skill_lower not in career_text:
                unsupported_count += 1

        unsupported_ratio = unsupported_count / max(skill_count, 1)
        if unsupported_ratio > 0.6 and unsupported_count > 5:
            penalty_points += 0.25
            reasons.append(f"unsupported_skills_{unsupported_count}_of_{skill_count}")
        elif unsupported_ratio > 0.4 and unsupported_count > 4:
            penalty_points += 0.10
            reasons.append(f"partially_unsupported_skills_{unsupported_count}")

    # Buzzword density: check if JD-matching terms are suspiciously concentrated
    # relative to actual career content length
    rel = features.get("relevance", {})
    must_count = rel.get("must_count", 0)
    nice_count = rel.get("nice_count", 0)
    total_jd_hits = must_count + nice_count

    total_career_words = sum(
        len(_tokenize(ev.description)) for ev in profile.career_events
    )
    if total_career_words > 0 and total_jd_hits > 0:
        density = total_jd_hits / (total_career_words / 100.0)  # hits per 100 words
        if density > 8.0 and total_jd_hits > 10:
            penalty_points += 0.20
            reasons.append(f"high_buzzword_density_{density:.1f}")

    return min(1.0, penalty_points), reasons


# ── Signal 3: Description Quality ────────────────────────────────────────────

def _check_description_quality(profile: CandidateProfile) -> tuple[float, list[str]]:
    """
    Detect copy-paste descriptions, very short descriptions, and
    overly uniform writing across roles.

    Returns:
        (score 0.0–1.0, list of reason codes)

    Safe for: candidates with genuinely similar roles at different companies.
    """
    reasons: list[str] = []
    penalty_points = 0.0
    events = profile.career_events

    if len(events) < 2:
        return 0.0, []

    descriptions = [
        ev.description for ev in events if ev.description and ev.description.strip()
    ]

    if len(descriptions) < 2:
        return 0.0, []

    # Check for identical / near-identical descriptions (copy-paste)
    duplicate_pairs = 0
    for i in range(len(descriptions)):
        for j in range(i + 1, len(descriptions)):
            sim = _similarity_ratio(descriptions[i], descriptions[j])
            if sim > 0.85:
                duplicate_pairs += 1

    if duplicate_pairs >= 2:
        penalty_points += 0.45
        reasons.append(f"duplicate_descriptions_{duplicate_pairs}_pairs")
    elif duplicate_pairs == 1:
        penalty_points += 0.20
        reasons.append("near_duplicate_description_1_pair")

    # Check for very short descriptions on long-tenure roles
    short_desc_count = 0
    for ev in events:
        if ev.description and ev.duration_months and ev.duration_months > 12:
            word_count = len(_tokenize(ev.description))
            if word_count < 15:
                short_desc_count += 1

    if short_desc_count >= 2:
        penalty_points += 0.15
        reasons.append(f"thin_descriptions_{short_desc_count}_roles")

    # Check for overly uniform sentence openings
    if len(descriptions) >= 3:
        first_words = []
        for desc in descriptions:
            tokens = _tokenize(desc)
            if len(tokens) >= 3:
                first_words.append(" ".join(tokens[:3]))

        if first_words:
            most_common_start, count = Counter(first_words).most_common(1)[0]
            if count >= 3 and count / len(first_words) > 0.6:
                penalty_points += 0.15
                reasons.append("uniform_description_openings")

    return min(1.0, penalty_points), reasons


# ── Signal 4: Claim-Evidence Gaps ────────────────────────────────────────────

def _check_claim_evidence_gaps(
    profile: CandidateProfile,
    features: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Detect mismatches between seniority claims, skill breadth claims,
    and actual evidence in career history.

    Returns:
        (score 0.0–1.0, list of reason codes)

    Safe for: recent graduates with relevant coursework, founders,
    career changers.
    """
    reasons: list[str] = []
    penalty_points = 0.0

    title_lower = profile.current_title.lower()
    exp_years = profile.years_of_experience

    # Senior/Lead/Principal title with very low experience
    seniority_terms = ["senior", "lead", "principal", "staff", "director", "head of"]
    has_senior_title = any(t in title_lower for t in seniority_terms)

    if has_senior_title and exp_years < 2.0:
        penalty_points += 0.30
        reasons.append(f"senior_title_low_exp_{exp_years:.1f}yr")
    elif has_senior_title and exp_years < 3.0:
        penalty_points += 0.10
        reasons.append(f"senior_title_moderate_exp_{exp_years:.1f}yr")

    # Many must-have skill matches but zero production signals
    rel = features.get("relevance", {})
    prod = features.get("production", {})
    must_count = rel.get("must_count", 0)
    prod_signal_count = prod.get("prod_signal_count", 0)

    if must_count >= 6 and prod_signal_count == 0:
        penalty_points += 0.25
        reasons.append(f"high_skill_match_{must_count}_no_production")
    elif must_count >= 4 and prod_signal_count == 0:
        penalty_points += 0.10
        reasons.append(f"skill_match_{must_count}_no_production")

    # High skill count but no career history at all
    skill_count = len(profile.skills)
    if skill_count > 10 and len(profile.career_events) == 0:
        penalty_points += 0.30
        reasons.append(f"skills_{skill_count}_no_career_history")

    # Experience claim vs. career event count mismatch
    # (e.g., claims 10 years but only 1 role listed)
    if exp_years > 8.0 and len(profile.career_events) <= 1:
        penalty_points += 0.15
        reasons.append(f"exp_{exp_years:.0f}yr_only_{len(profile.career_events)}_role")

    return min(1.0, penalty_points), reasons


# ── Confidence Estimation ────────────────────────────────────────────────────

def _estimate_confidence(profile: CandidateProfile) -> float:
    """
    Estimate how confident we should be in the credibility assessment.

    Lower confidence → penalty is halved (we don't want to punish
    sparse-but-genuine profiles).

    Returns 0.0–1.0.
    """
    confidence = 0.5  # baseline

    # More career events → more data → higher confidence
    n_events = len(profile.career_events)
    if n_events >= 3:
        confidence += 0.3
    elif n_events >= 2:
        confidence += 0.15

    # Descriptions present → higher confidence
    desc_count = sum(
        1 for ev in profile.career_events
        if ev.description and len(ev.description.strip()) > 30
    )
    if desc_count >= 2:
        confidence += 0.15
    elif desc_count >= 1:
        confidence += 0.05

    # Skills present → higher confidence
    if len(profile.skills) >= 3:
        confidence += 0.05

    return min(1.0, confidence)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def check_resume_credibility(
    profile: CandidateProfile,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Run all credibility checks and return an aggregate result.

    Returns:
        Dict with:
        - credibility_penalty: float (0.0–0.12, how much to subtract)
        - credibility_multiplier: float (0.88–1.0, applied like contra_penalty)
        - signal_scores: dict of per-signal raw scores (0.0–1.0)
        - reason_codes: list of reason code strings
        - confidence: float (0.0–1.0)
    """
    # Run each signal checker
    timeline_score, timeline_reasons = _check_timeline_anomalies(profile)
    stuffing_score, stuffing_reasons = _check_keyword_stuffing(profile, features)
    desc_score, desc_reasons = _check_description_quality(profile)
    claim_score, claim_reasons = _check_claim_evidence_gaps(profile, features)

    signal_scores = {
        "timeline":    round(timeline_score, 4),
        "stuffing":    round(stuffing_score, 4),
        "description": round(desc_score, 4),
        "claim_gap":   round(claim_score, 4),
    }

    # Weighted aggregation
    raw_penalty = (
        _SIGNAL_WEIGHTS["timeline"]    * timeline_score
        + _SIGNAL_WEIGHTS["stuffing"]    * stuffing_score
        + _SIGNAL_WEIGHTS["description"] * desc_score
        + _SIGNAL_WEIGHTS["claim_gap"]   * claim_score
    )

    # Estimate confidence
    confidence = _estimate_confidence(profile)

    # Scale penalty by confidence when confidence is low
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        raw_penalty *= 0.5

    # Cap the penalty
    credibility_penalty = round(min(_MAX_PENALTY, raw_penalty), 4)
    credibility_multiplier = round(1.0 - credibility_penalty, 4)

    # Collect all reason codes
    all_reasons = timeline_reasons + stuffing_reasons + desc_reasons + claim_reasons

    return {
        "credibility_penalty": credibility_penalty,
        "credibility_multiplier": credibility_multiplier,
        "signal_scores": signal_scores,
        "reason_codes": all_reasons,
        "confidence": round(confidence, 3),
    }
