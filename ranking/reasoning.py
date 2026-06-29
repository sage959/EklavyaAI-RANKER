"""
reasoning.py — Generates grounded, single-paragraph reasoning strings based on sub-scores.
Does not hallucinate; only uses data passed via the feature extraction dict.
"""
from __future__ import annotations

from typing import Any, Dict


def generate_reasoning(features: Dict[str, Any]) -> str:
    """
    Builds a grounded reasoning string explaining the final score.
    Returns a single string.
    """
    pts = []
    
    # 1. Identify the top 2 score drivers
    comps = [
        ("relevance", features["relevance"]["score"] * 100),
        ("production", features["production"]["score"] * 100),
        ("technical", features["technical"]["score"] * 100)
    ]
    comps.sort(key=lambda x: x[1], reverse=True)
    
    top_driver = comps[0][0]
    
    if top_driver == "relevance":
        m_count = features["relevance"]["must_count"]
        top_skills = ", ".join(features["relevance"]["must_matched"][:2])
        if m_count >= 4:
            pts.append(f"Strong JD match ({m_count} must-haves including {top_skills})")
        else:
            pts.append(f"Partial JD match ({m_count} must-haves)")
            
    elif top_driver == "production":
        exp = features["production"]["exp_years"]
        if features["production"]["tier1"]:
            pts.append(f"Deep production ML experience ({exp} yrs) with Tier-1 tech background")
        elif features["production"]["prod_signal_count"] >= 3:
            pts.append(f"Proven history of shipping ML systems ({exp} yrs exp)")
        else:
            pts.append(f"Solid ML experience ({exp} yrs)")
            
    elif top_driver == "technical":
        ret = ", ".join(features["technical"]["retrieval_matched"][:2])
        if ret:
            pts.append(f"Deep retrieval/ranking stack expertise ({ret})")
        else:
            pts.append("Strong technical evaluation skills")

    # Second driver
    second_driver = comps[1][0]
    if second_driver == "production" and top_driver != "production":
        exp = features["production"]["exp_years"]
        pts.append(f"shipped ML systems ({exp} yrs exp)")
    elif second_driver == "technical" and top_driver != "technical":
        ret = ", ".join(features["technical"]["retrieval_matched"][:2])
        if ret:
            pts.append(f"solid retrieval stack ({ret})")
    elif second_driver == "relevance" and top_driver != "relevance":
        m_count = features["relevance"]["must_count"]
        pts.append(f"matched {m_count} key skills")

    # 2. Add 1 availability/behavioral fact
    beh = features["behavioral"]
    if beh["open_to_work"] and beh["notice_days"] <= 30:
        pts.append(f"Available fast ({beh['notice_days']}d notice)")
    elif beh["response_time_h"] <= 6:
        pts.append(f"Highly responsive ({beh['response_time_h']}h avg)")
    elif beh["github"] >= 80:
        pts.append("High GitHub activity")
        
    # 3. If score < 75 or negative flags present, add a gap signal
    flags = features["negative"]["flags"]
    if flags:
        if "no_production_signals" in flags:
            pts.append("Gap: missing production/deployment signals")
        elif "irrelevant_current_role" in flags:
            pts.append("Gap: current role lacks engineering relevance")
        elif "api_wrapper_only" in flags:
            pts.append("Gap: primarily API wrapper experience, missing systems depth")
        elif "under_experienced" in flags:
            pts.append("Gap: below minimum experience requirement")
    elif features.get("final_score", 0) < 75:
        # Find lowest component
        lowest = comps[-1][0]
        if lowest == "technical":
            pts.append("Gap: limited specific retrieval/ranking tech stack")
        elif lowest == "production":
            pts.append("Gap: lighter on production scale evidence")
        elif lowest == "relevance":
            pts.append("Gap: missing several JD must-have skills")

    # Capitalize first letter, join with " · ", end with "."
    if not pts:
        return "Profile analyzed."
        
    reasoning = " · ".join(pts)
    reasoning = reasoning[0].upper() + reasoning[1:]
    if not reasoning.endswith("."):
        reasoning += "."
        
    return reasoning
