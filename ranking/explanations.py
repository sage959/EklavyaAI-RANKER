"""
explanations.py — Grounded explanation generation.
Produces both user-facing text and machine-readable explanation objects.
The reasoning text is composed from evidence primitives, not slot-filled templates.
"""
from __future__ import annotations

import re
from typing import Any


def _sanitize_text(text: str) -> str:
    """Remove encoding artifacts, bad unicode, and formatting junk."""
    text = text.replace("\u00c2", "").replace("\u00a0", " ")
    text = text.replace("\ufffd", "").replace("\u200b", "")
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s*[·•]\s*$", "", text)
    text = re.sub(r"\.\.", ".", text)
    text = re.sub(r",\s*\.", ".", text)
    return text.strip()


def generate_explanation(
    features: dict[str, Any],
    pow_result: dict[str, Any],
    contradiction_result: dict[str, Any],
    retrieval_info: dict[str, Any] | None = None,
    credibility_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a grounded explanation for a candidate's ranking.

    Returns:
        Dict with:
        - reasoning_text: str (user-facing, for CSV)
        - strengths: list of evidence-cited strength statements
        - gaps: list of evidence-cited gap statements
        - warnings: list of contradiction warnings
        - evidence_refs: list of source references
        - confidence: float (0-1)
    """
    strengths: list[dict[str, str]] = []
    gaps: list[dict[str, str]] = []
    evidence_refs: list[str] = []

    # ── Extract all evidence primitives ───────────────────────────────────
    rel = features.get("relevance", {})
    must_count = rel.get("must_count", 0)
    must_matched = rel.get("must_matched", [])

    prod = features.get("production", {})
    exp_years = prod.get("exp_years", 0)
    is_tier1 = prod.get("tier1", False)
    prod_signals = prod.get("prod_signal_count", 0)

    tech = features.get("technical", {})
    ret_matched = tech.get("retrieval_matched", [])

    beh = features.get("behavioral", {})
    neg = features.get("negative", {})
    neg_flags = neg.get("flags", [])
    relevant_prior_count = neg.get("relevant_prior_count", 0)
    relevant_prior_titles = neg.get("relevant_prior_titles", [])
    chr_info = features.get("career_history_relevance", {})

    pow_score = pow_result.get("pow_score", 0)
    pow_summary = pow_result.get("evidence_summary", "")

    # ── Build structured strengths (for UI panel) ────────────────────────

    if must_count >= 6:
        top_skills = ", ".join(must_matched[:3])
        strengths.append({
            "claim": f"Matches {must_count} must-have requirements ({top_skills})",
            "source": "skills + career text",
        })
        evidence_refs.append("skill_relevance")
    elif must_count >= 3:
        top_skills = ", ".join(must_matched[:2])
        strengths.append({
            "claim": f"Matches {must_count} must-have skills ({top_skills})",
            "source": "skills + career text",
        })
        evidence_refs.append("skill_relevance")

    if is_tier1:
        strengths.append({
            "claim": f"Tier-1 tech company background, {exp_years:.0f} yrs experience",
            "source": "career_history companies",
        })
        evidence_refs.append("career_history")
    elif prod_signals >= 3:
        strengths.append({
            "claim": f"Production deployment evidence across {exp_years:.0f} years",
            "source": "career descriptions",
        })
        evidence_refs.append("career_descriptions")

    if len(ret_matched) >= 3:
        top_ret = ", ".join(ret_matched[:3])
        strengths.append({
            "claim": f"Relevant technical stack: {top_ret}",
            "source": "skills + career text",
        })
        evidence_refs.append("technical_stack")

    if pow_score >= 0.4:
        strengths.append({
            "claim": f"Proof-of-work evidence ({pow_summary})",
            "source": "career descriptions",
        })
        evidence_refs.append("proof_of_work")

    if beh.get("open_to_work") and beh.get("notice_days", 90) <= 30:
        strengths.append({
            "claim": f"Available within {beh['notice_days']} days",
            "source": "redrob_signals",
        })
    elif beh.get("response_time_h", 24) <= 6:
        strengths.append({
            "claim": f"Responsive ({beh['response_time_h']:.0f}h avg reply)",
            "source": "redrob_signals",
        })
    elif beh.get("github", 0) >= 70:
        strengths.append({
            "claim": "Active GitHub contributor",
            "source": "redrob_signals",
        })

    if chr_info.get("is_career_pivot"):
        pass
    elif chr_info.get("career_history_relevant_count", 0) >= 3:
        count = chr_info["career_history_relevant_count"]
        strengths.append({
            "claim": f"{count} relevant prior engineering roles",
            "source": "career_history titles",
        })

    # ── Build structured gaps (for UI panel) ─────────────────────────────

    if "no_production_signals" in neg_flags:
        gaps.append({
            "claim": "No production deployment evidence found",
            "source": "career descriptions",
        })

    if "irrelevant_current_role" in neg_flags:
        gaps.append({
            "claim": "Current role lacks engineering relevance",
            "source": "profile.current_title",
        })
    elif "irrelevant_current_role_mild" in neg_flags:
        prior_title = relevant_prior_titles[0] if relevant_prior_titles else "an engineering role"
        gaps.append({
            "claim": f"Current role is non-technical (prior: {prior_title})",
            "source": "profile.current_title + career_history",
        })
    elif "career_pivot_noted" in neg_flags:
        prior_list = ", ".join(relevant_prior_titles[:3])
        strengths.append({
            "claim": f"Career pivot with {relevant_prior_count} prior relevant roles ({prior_list})",
            "source": "career_history titles",
        })
        evidence_refs.append("career_pivot")

    if "api_wrapper_only" in neg_flags:
        gaps.append({
            "claim": "Experience skews toward API integration over systems depth",
            "source": "career descriptions",
        })
    if "under_experienced" in neg_flags:
        gaps.append({
            "claim": f"Experience ({exp_years:.0f} yrs) below JD minimum",
            "source": "profile.years_of_experience",
        })

    if not gaps and must_count < 5:
        comps = [
            ("technical", tech.get("score", 0)),
            ("production", prod.get("score", 0)),
            ("relevance", rel.get("score", 0)),
        ]
        comps.sort(key=lambda x: x[1])
        lowest_name = comps[0][0]
        gap_map = {
            "technical": "Limited retrieval/ranking technology evidence",
            "production": "Thin production-scale deployment evidence",
            "relevance": "Several core JD requirements not clearly covered",
        }
        gaps.append({
            "claim": gap_map.get(lowest_name, "Some JD areas not well covered"),
            "source": f"{lowest_name} score analysis",
        })

    # ── Warnings ─────────────────────────────────────────────────────────
    warnings = contradiction_result.get("warnings", [])

    # ── Confidence ───────────────────────────────────────────────────────
    base_confidence = min(1.0, len(strengths) * 0.2 + 0.3)
    if warnings:
        base_confidence *= max(0.7, 1.0 - len(warnings) * 0.05)

    # ── Credibility concerns (if penalty > 0.02) ─────────────────────────
    if credibility_result:
        cred_penalty = credibility_result.get("credibility_penalty", 0.0)
        if cred_penalty > 0.02:
            reason_codes = credibility_result.get("reason_codes", [])
            signal_scores = credibility_result.get("signal_scores", {})

            # Map reason codes to human-readable concern statements
            concern_parts: list[str] = []
            if signal_scores.get("timeline", 0) > 0.15:
                concern_parts.append("timeline inconsistencies")
            if signal_scores.get("stuffing", 0) > 0.15:
                concern_parts.append("possible keyword inflation")
            if signal_scores.get("description", 0) > 0.15:
                concern_parts.append("low description quality")
            if signal_scores.get("claim_gap", 0) > 0.15:
                concern_parts.append("claim-evidence gaps")

            if concern_parts:
                gaps.append({
                    "claim": f"Credibility signals: {', '.join(concern_parts)}",
                    "source": "resume credibility checks",
                })
                warnings.extend([
                    f"Credibility: {code}" for code in reason_codes[:3]
                ])
                base_confidence *= max(0.85, 1.0 - cred_penalty)

    # ── Build natural reasoning text from evidence ───────────────────────
    evidence = _build_evidence_bag(
        must_count=must_count,
        must_matched=must_matched,
        exp_years=exp_years,
        is_tier1=is_tier1,
        prod_signals=prod_signals,
        ret_matched=ret_matched,
        pow_score=pow_score,
        pow_summary=pow_summary,
        neg_flags=neg_flags,
        relevant_prior_count=relevant_prior_count,
        relevant_prior_titles=relevant_prior_titles,
        chr_relevant_count=chr_info.get("career_history_relevant_count", 0),
        beh=beh,
        gaps=gaps,
        tech_score=tech.get("score", 0),
        prod_score=prod.get("score", 0),
        rel_score=rel.get("score", 0),
    )
    reasoning_text = _compose_natural_reasoning(evidence)

    return {
        "reasoning_text": reasoning_text,
        "strengths": strengths,
        "gaps": gaps,
        "warnings": warnings,
        "evidence_refs": evidence_refs,
        "confidence": round(base_confidence, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Evidence bag + compositional reasoning generator
# ═════════════════════════════════════════════════════════════════════════════

def _build_evidence_bag(**kwargs) -> dict[str, Any]:
    """Package all evidence into a single bag for the composer."""
    return kwargs


def _compose_natural_reasoning(ev: dict[str, Any]) -> str:
    """
    Build 1-2 natural sentences from raw evidence primitives.

    Strategy:
    1. Identify the candidate's "profile shape" (what's strongest).
    2. Pick a lead-in strategy based on that shape.
    3. Compose a natural sentence adding one supporting detail.
    4. Optionally append one honest concern.
    5. Clean and return.
    """
    must_count = ev["must_count"]
    must_matched = ev["must_matched"]
    exp_years = ev["exp_years"]
    is_tier1 = ev["is_tier1"]
    prod_signals = ev["prod_signals"]
    ret_matched = ev["ret_matched"]
    pow_score = ev["pow_score"]
    pow_summary = ev["pow_summary"]
    neg_flags = ev["neg_flags"]
    chr_relevant = ev["chr_relevant_count"]
    beh = ev["beh"]
    gaps = ev["gaps"]

    # ── Determine profile shape scores for strategy selection ──
    skill_strength = must_count  # 0-10+
    tech_depth = len(ret_matched)  # 0-5+
    career_depth = 1 if is_tier1 else (0.7 if prod_signals >= 3 else 0.3)
    exp_strength = min(1.0, exp_years / 10) if exp_years > 0 else 0

    # Pick the dominant signal for the lead sentence
    signals = [
        ("skills", skill_strength * 1.5),
        ("tech", tech_depth * 2.0),
        ("career", career_depth * 5),
        ("experience", exp_strength * 4),
    ]
    signals.sort(key=lambda x: -x[1])
    lead_signal = signals[0][0]

    # Use a seed for deterministic secondary variation
    seed = must_count * 11 + int(exp_years * 7) + tech_depth * 13

    # ── Compose lead sentence based on dominant signal ──
    lead = _build_lead(lead_signal, ev, seed)

    # ── Compose supporting detail ──
    support = _build_support(lead_signal, ev, seed)

    # ── Compose concern/limitation ──
    concern = _build_concern(ev, seed)

    # ── Assemble ──
    parts = [p for p in [lead, support, concern] if p]
    text = " ".join(parts)

    # ── Final cleanup ──
    text = _sanitize_text(text)
    if text and not text.endswith("."):
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    text = text.replace("..", ".").replace(". .", ".").replace(".,", ",")
    text = re.sub(r"\s+", " ", text).strip()

    return text or "Profile reviewed with limited matching signals."


def _build_lead(signal: str, ev: dict, seed: int) -> str:
    """Build the opening phrase based on the dominant evidence signal."""
    must_count = ev["must_count"]
    must_matched = ev["must_matched"]
    exp_years = ev["exp_years"]
    is_tier1 = ev["is_tier1"]
    prod_signals = ev["prod_signals"]
    ret_matched = ev["ret_matched"]

    if signal == "skills" and must_count >= 6:
        top = ", ".join(must_matched[:3])
        options = [
            f"Well-matched on core JD requirements with skills in {top}.",
            f"The profile covers key areas the JD calls for, including {top}.",
            f"Directly relevant skill set spanning {top} and {must_count - 3} other JD requirements.",
            f"Clear alignment on {must_count} JD requirements, notably {top}.",
        ]
        return options[seed % len(options)]

    if signal == "skills" and must_count >= 3:
        top = ", ".join(must_matched[:2])
        options = [
            f"Covers several JD requirements including {top}.",
            f"The candidate's background touches on {top} and {must_count - 2} other relevant areas.",
            f"Partial but meaningful skill overlap with the JD, particularly in {top}.",
        ]
        return options[seed % len(options)]

    if signal == "tech" and len(ret_matched) >= 3:
        top = ", ".join(ret_matched[:3])
        options = [
            f"The technical profile is a strong match, with hands-on {top} experience.",
            f"Brings direct experience in {top}, closely aligned with what the role requires.",
            f"Technically well-positioned for this role, with evidence of {top} usage.",
        ]
        return options[seed % len(options)]

    if signal == "career":
        if is_tier1:
            options = [
                f"Comes from a tier-1 tech background with {exp_years:.0f} years in the field.",
                f"Seasoned engineer with {exp_years:.0f} years of experience, including time at a major tech company.",
                f"The candidate's {exp_years:.0f}-year career includes high-caliber engineering environments.",
            ]
        elif prod_signals >= 3:
            options = [
                f"Has a track record of production engineering work across {exp_years:.0f} years.",
                f"The career history shows real production ownership over {exp_years:.0f} years.",
                f"Brings {exp_years:.0f} years of experience with clear production deployment signals.",
            ]
        else:
            options = [
                f"Brings {exp_years:.0f} years of relevant industry experience.",
                f"The candidate has {exp_years:.0f} years in the field.",
            ]
        return options[seed % len(options)]

    if signal == "experience" and exp_years >= 3:
        options = [
            f"With {exp_years:.0f} years in the industry, the candidate has substantial depth.",
            f"Brings {exp_years:.0f} years of engineering experience to the table.",
            f"An experienced practitioner with {exp_years:.0f} years of relevant work.",
        ]
        return options[seed % len(options)]

    # Fallback
    if must_count >= 1:
        top = must_matched[0] if must_matched else "relevant areas"
        return f"The profile shows some alignment with the JD, particularly around {top}."
    return "The profile was evaluated against the JD requirements."


def _build_support(lead_signal: str, ev: dict, seed: int) -> str:
    """Build a supporting detail that adds new information beyond the lead."""
    must_count = ev["must_count"]
    must_matched = ev["must_matched"]
    exp_years = ev["exp_years"]
    is_tier1 = ev["is_tier1"]
    prod_signals = ev["prod_signals"]
    ret_matched = ev["ret_matched"]
    pow_score = ev["pow_score"]
    pow_summary = ev["pow_summary"]
    chr_relevant = ev["chr_relevant_count"]
    beh = ev["beh"]

    # Avoid repeating the lead signal — pick a different supporting fact
    candidates = []

    # Production evidence (if lead wasn't career)
    if lead_signal != "career" and prod_signals >= 3:
        candidates.append(
            f"Career history includes clear production deployment evidence over {exp_years:.0f} years."
        )
    if lead_signal != "career" and is_tier1:
        candidates.append(
            f"Background includes experience at a major tech company."
        )

    # Skill count (if lead wasn't skills)
    if lead_signal != "skills" and must_count >= 5:
        top = ", ".join(must_matched[:2])
        candidates.append(
            f"Matches {must_count} core JD skills including {top}."
        )

    # Tech stack (if lead wasn't tech)
    if lead_signal != "tech" and len(ret_matched) >= 2:
        top = ", ".join(ret_matched[:2])
        candidates.append(
            f"The technical stack includes {top}, which aligns with JD needs."
        )

    # PoW evidence
    if pow_score >= 0.4 and pow_summary:
        clean_summary = pow_summary.strip().rstrip(".")
        candidates.append(
            f"Proof-of-work signals are present ({clean_summary})."
        )

    # Career trajectory
    if chr_relevant >= 3:
        candidates.append(
            f"The career path shows {chr_relevant} relevant engineering roles, indicating sustained focus."
        )

    # Behavioral signals
    if beh.get("open_to_work") and beh.get("notice_days", 90) <= 30:
        candidates.append(
            f"Currently open to opportunities with a {beh['notice_days']}-day notice period."
        )

    if not candidates:
        return ""

    return candidates[seed % len(candidates)]


def _build_concern(ev: dict, seed: int) -> str:
    """Build one honest concern or limitation, if warranted."""
    gaps = ev["gaps"]
    neg_flags = ev["neg_flags"]
    must_count = ev["must_count"]
    tech_score = ev.get("tech_score", 0)
    prod_score = ev.get("prod_score", 0)
    rel_score = ev.get("rel_score", 0)

    # Use the first gap if available
    if gaps:
        gap_claim = gaps[0]["claim"]
        # Rephrase as a natural qualifier instead of "Gap: X"
        qualifiers = [
            f"That said, {gap_claim[0].lower()}{gap_claim[1:]}.",
            f"One area to note: {gap_claim[0].lower()}{gap_claim[1:]}.",
            f"The main concern is that {gap_claim[0].lower()}{gap_claim[1:]}.",
            f"A limitation is that {gap_claim[0].lower()}{gap_claim[1:]}.",
            f"However, {gap_claim[0].lower()}{gap_claim[1:]}.",
        ]
        return qualifiers[seed % len(qualifiers)]

    # If no explicit gap but the candidate is mid/low range
    if must_count < 4:
        options = [
            "The overall fit with JD-specific requirements is partial.",
            "Several JD requirements are not directly evidenced in the profile.",
            "The match is present but less comprehensive than stronger candidates.",
        ]
        return options[seed % len(options)]

    return ""
