"""
data_loader.py — Utilities for parsing the real Redrob Candidate Schema.
"""
from __future__ import annotations
from typing import Any, Dict, List

def build_text_corpus(candidate: Dict[str, Any]) -> str:
    """
    Concatenate all searchable text from a candidate into one lowercase string.
    Adapts to the real Redrob candidate_schema.json format.
    """
    parts: List[str] = []

    # Profile
    profile = candidate.get("profile", {})
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("current_title", ""))

    # Skills (Array of objects)
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    # Career history descriptions + titles (Array of objects)
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
        parts.append(job.get("company", ""))
        parts.append(job.get("industry", ""))

    # Education
    for edu in candidate.get("education", []):
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))

    return " ".join(filter(None, parts)).lower()
