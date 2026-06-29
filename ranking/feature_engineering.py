"""
feature_engineering.py — Extract normalised sub-scores (0.0–1.0) from a candidate.
Updated to map exactly to the real candidate_schema.json fields.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from ranking import config
from ranking.data_loader import build_text_corpus


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _lerp_inv(value: float, best: float, worst: float) -> float:
    if best == worst:
        return 1.0 if value <= best else 0.0
    return _clamp((value - worst) / (best - worst))


def _count_matches(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    matched = [kw for kw in keywords if kw.lower() in text]
    return len(matched), matched


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def score_skill_relevance(candidate: Dict[str, Any], parsed_jd: Any = None) -> Dict[str, Any]:
    corpus = build_text_corpus(candidate)

    must_have = parsed_jd.must_have_skills if parsed_jd else config.JD_MUST_HAVE
    nice_have = parsed_jd.nice_to_have_skills if parsed_jd else config.JD_NICE_TO_HAVE

    must_count, must_matched = _count_matches(corpus, must_have)
    must_score = _clamp(must_count * 0.12)

    nice_count, nice_matched = _count_matches(corpus, nice_have)
    nice_bonus = _clamp(nice_count * 0.04, hi=0.20)

    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    title_relevant = any(t in current_title for t in config.RELEVANT_TITLE_TERMS)
    title_irrelevant = any(t in current_title for t in config.IRRELEVANT_TITLE_TERMS)
    title_bonus = 0.08 if title_relevant and not title_irrelevant else 0.0

    raw = _clamp(must_score + nice_bonus + title_bonus)

    return {
        "score": raw,
        "must_count": must_count,
        "must_matched": must_matched[:6],
        "nice_count": nice_count,
        "nice_matched": nice_matched[:4],
        "title_relevant": title_relevant,
    }


def score_production_depth(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    all_desc = " ".join(j.get("description", "") for j in career).lower()

    prod_count, prod_matched = _count_matches(all_desc, config.PRODUCTION_SIGNALS)
    prod_signal_score = _clamp(prod_count / 6.0)

    exp_years = float(profile.get("years_of_experience", 0) or 0)
    exp_score = _clamp(exp_years / 7.0)

    all_companies = " ".join(j.get("company", "").lower() for j in career)
    tier1 = any(c in all_companies for c in config.TIER1_COMPANIES)
    tier2 = any(c in all_companies for c in config.TIER2_COMPANIES)
    company_bonus = 0.15 if tier1 else (0.08 if tier2 else 0.0)

    companies = {j.get("company", "").strip().lower() for j in career if j.get("company")}
    breadth_bonus = _clamp((len(companies) - 1) * 0.04, hi=0.08)

    raw = _clamp(0.45 * prod_signal_score + 0.35 * exp_score + company_bonus + breadth_bonus)

    return {
        "score": raw,
        "prod_signal_count": prod_count,
        "prod_matched": prod_matched[:4],
        "exp_years": exp_years,
        "tier1": tier1,
        "tier2": tier2,
        "company_count": len(companies),
    }


def score_technical_alignment(candidate: Dict[str, Any]) -> Dict[str, Any]:
    corpus = build_text_corpus(candidate)

    ret_count, ret_matched = _count_matches(corpus, config.RETRIEVAL_STACK_TERMS)
    ret_score = _clamp(ret_count * 0.10, hi=0.60)

    eval_count, eval_matched = _count_matches(corpus, config.EVALUATION_TERMS)
    eval_score = _clamp(eval_count * 0.10, hi=0.25)

    py_count, py_matched = _count_matches(corpus, config.PYTHON_DEPTH_TERMS)
    py_score = _clamp(py_count * 0.05, hi=0.20)

    raw = _clamp(ret_score + eval_score + py_score)

    return {
        "score": raw,
        "retrieval_count": ret_count,
        "retrieval_matched": ret_matched[:5],
        "eval_count": eval_count,
        "eval_matched": eval_matched[:3],
        "python_count": py_count,
        "python_matched": py_matched[:3],
    }


def score_behavioral(candidate: Dict[str, Any]) -> Dict[str, Any]:
    sig = candidate.get("redrob_signals", {})
    bw = config.BEHAVIORAL_WEIGHTS
    bt = config.BEHAVIORAL

    otw_score = 1.0 if sig.get("open_to_work_flag", False) else 0.5

    last_active = _parse_date(str(sig.get("last_active_date", "")))
    if last_active:
        days_ago = (date.today() - last_active).days
        recency_score = _lerp_inv(days_ago, bt["active_fresh_d"], bt["active_stale_d"])
    else:
        recency_score = 0.3

    # response rate is already 0-1
    rr_fraction = float(sig.get("recruiter_response_rate", 0.5) or 0.5)
    rr = rr_fraction * 100
    response_rate_score = _lerp_inv(rr, bt["response_rate_best"], bt["response_rate_worst"])

    art = float(sig.get("avg_response_time_hours", 12) or 12)
    response_time_score = _lerp_inv(art, bt["response_best_h"], bt["response_worst_h"])

    np_ = int(sig.get("notice_period_days", 60) or 60)
    notice_score = _lerp_inv(np_, bt["notice_best_d"], bt["notice_worst_d"])
    notice_score = _clamp(notice_score, lo=0.40)

    gh = float(sig.get("github_activity_score", 40) or 40)
    if gh == -1:
        gh = 20 # no github linked -> penalty
    github_score = _lerp_inv(gh, bt["github_best"], bt["github_worst"])

    ic_fraction = float(sig.get("interview_completion_rate", 0.8) or 0.8)
    ic = ic_fraction * 100
    interview_score = _lerp_inv(ic, bt["interview_best"], bt["interview_worst"])

    weighted = (
        bw["open_to_work"]   * otw_score
        + bw["recency"]        * recency_score
        + bw["response_rate"]  * response_rate_score
        + bw["response_time"]  * response_time_score
        + bw["notice_period"]  * notice_score
        + bw["github"]         * github_score
        + bw["interview_rate"] * interview_score
    )

    return {
        "score": _clamp(weighted),
        "open_to_work": sig.get("open_to_work_flag", False),
        "recency_score": round(recency_score, 3),
        "response_rate": rr,
        "response_time_h": art,
        "notice_days": np_,
        "github": gh,
        "interview_rate": ic,
        "saved_30d": sig.get("saved_by_recruiters_30d", 0),
    }


def _count_relevant_prior_roles(career: list[dict]) -> tuple[int, list[str]]:
    """Count how many prior (non-current) career roles match engineering-relevant titles."""
    relevant_titles: list[str] = []
    for job in career:
        if job.get("is_current", False):
            continue
        title = job.get("title", "").lower()
        if any(term in title for term in config.CAREER_HISTORY_RELEVANT_TERMS):
            relevant_titles.append(job.get("title", ""))
    return len(relevant_titles), relevant_titles


def compute_negative_multiplier(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a negative multiplier based on red flags.

    Career-pivot-aware: if the current title is irrelevant but the candidate
    has prior engineering roles, the penalty is reduced or removed entirely.
    """
    multiplier = 1.0
    flags: List[str] = []

    corpus = build_text_corpus(candidate)
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    all_desc = " ".join(j.get("description", "") for j in career).lower()

    # ── No production signals ────────────────────────────────────────────
    prod_count, _ = _count_matches(all_desc, config.PRODUCTION_SIGNALS)
    if prod_count == 0:
        multiplier *= config.NEG_NO_PRODUCTION
        flags.append("no_production_signals")

    # ── Current title relevance (career-pivot-aware) ─────────────────────
    current_title = profile.get("current_title", "").lower()
    title_is_irrelevant = any(t in current_title for t in config.IRRELEVANT_TITLE_TERMS)

    relevant_prior_count = 0
    relevant_prior_titles: list[str] = []

    if title_is_irrelevant:
        relevant_prior_count, relevant_prior_titles = _count_relevant_prior_roles(career)

        if relevant_prior_count >= 2:
            # Career pivot: strong engineering background — NO penalty
            flags.append("career_pivot_noted")
        elif relevant_prior_count == 1:
            # Single prior relevant role — mild penalty
            multiplier *= config.NEG_IRRELEVANT_ROLE_MILD
            flags.append("irrelevant_current_role_mild")
        else:
            # Zero relevant prior roles — full penalty
            multiplier *= config.NEG_IRRELEVANT_ROLE_FULL
            flags.append("irrelevant_current_role")

    # ── API wrapper only ─────────────────────────────────────────────────
    has_wrapper = any(kw in corpus for kw in ["langchain", "openai api", "chatgpt"])
    has_systems = any(kw in corpus for kw in ["faiss", "elasticsearch", "pipeline", "serving", "infrastructure"])
    if has_wrapper and not has_systems:
        multiplier *= config.NEG_WRAPPER_ONLY
        flags.append("api_wrapper_only")

    # ── Under-experienced ────────────────────────────────────────────────
    exp = float(profile.get("years_of_experience", 0) or 0)
    if exp < 2.0:
        multiplier *= config.NEG_UNDER_EXPERIENCE
        flags.append("under_experienced")

    multiplier = max(multiplier, config.NEG_FLOOR)

    return {
        "multiplier": round(multiplier, 4),
        "flags": flags,
        "relevant_prior_count": relevant_prior_count,
        "relevant_prior_titles": relevant_prior_titles,
    }


def score_career_history_relevance(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score how relevant the full career history is, independent of current title.

    Returns a blended title_score:
        title_score = 0.40 * current_title_relevance + 0.60 * career_history_relevance

    This ensures career history matters MORE than just the current title.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Current title relevance (0 or 1)
    current_title = profile.get("current_title", "").lower()
    current_relevant = 1.0 if any(t in current_title for t in config.RELEVANT_TITLE_TERMS) else 0.0
    current_irrelevant = any(t in current_title for t in config.IRRELEVANT_TITLE_TERMS)
    if current_irrelevant:
        current_relevant = 0.0

    # Career history relevance: proportion of roles with relevant titles
    total_roles = len(career)
    if total_roles == 0:
        career_relevance = 0.0
        relevant_count = 0
        relevant_titles = []
    else:
        relevant_titles = []
        for job in career:
            title = job.get("title", "").lower()
            if any(term in title for term in config.CAREER_HISTORY_RELEVANT_TERMS):
                relevant_titles.append(job.get("title", ""))
        relevant_count = len(relevant_titles)
        # Score: at least 2 relevant roles = 1.0, 1 = 0.6, 0 = 0.0
        if relevant_count >= 2:
            career_relevance = 1.0
        elif relevant_count == 1:
            career_relevance = 0.6
        else:
            career_relevance = 0.0

    # Blended title score: career history weighted MORE than current title
    blended = 0.40 * current_relevant + 0.60 * career_relevance
    is_career_pivot = current_irrelevant and relevant_count >= 2

    return {
        "score": round(_clamp(blended), 4),
        "current_title_relevant": current_relevant > 0,
        "career_history_relevant_count": relevant_count,
        "career_history_relevant_titles": relevant_titles[:5],
        "is_career_pivot": is_career_pivot,
        "total_roles": total_roles,
    }


def extract_all_features(candidate: Dict[str, Any], parsed_jd: Any = None) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "name": candidate.get("profile", {}).get("anonymized_name", "Unknown"),
        "relevance":   score_skill_relevance(candidate, parsed_jd),
        "production":  score_production_depth(candidate),
        "technical":   score_technical_alignment(candidate),
        "behavioral":  score_behavioral(candidate),
        "career_history_relevance": score_career_history_relevance(candidate),
        "negative":    compute_negative_multiplier(candidate),
    }
