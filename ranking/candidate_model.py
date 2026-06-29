"""
candidate_model.py — Candidate-centric document model.
Converts raw JSON candidate records into structured, retrieval-ready representations
with evidence units, career events, skill inventories, and proof-of-work blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvidenceChunk:
    """A single piece of candidate evidence, ready for retrieval indexing."""
    candidate_id: str
    source_type: str          # "summary" | "headline" | "career" | "skill" | "education" | "project"
    section: str              # human label e.g. "Career @ Google"
    title: str = ""           # job title if applicable
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    is_current: bool = False
    importance_weight: float = 1.0
    text: str = ""
    normalized_text: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source_type": self.source_type,
            "section": self.section,
            "title": self.title,
            "company": self.company,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "duration_months": self.duration_months,
            "is_current": self.is_current,
            "importance_weight": self.importance_weight,
            "text": self.text,
        }


@dataclass
class CareerEvent:
    """A structured career timeline entry."""
    company: str
    title: str
    start_date: str
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "title": self.title,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "duration_months": self.duration_months,
            "is_current": self.is_current,
            "industry": self.industry,
            "company_size": self.company_size,
            "description": self.description,
        }


@dataclass
class SkillEntry:
    """A normalised skill from the candidate's profile."""
    name: str
    proficiency: str = "unknown"   # beginner | intermediate | advanced | unknown
    endorsements: int = 0
    duration_months: int = 0

    @property
    def normalized_name(self) -> str:
        return self.name.strip().lower()


@dataclass
class CandidateProfile:
    """Normalized candidate representation."""
    candidate_id: str
    name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""

    skills: list[SkillEntry] = field(default_factory=list)
    career_events: list[CareerEvent] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    certifications: list[dict] = field(default_factory=list)
    redrob_signals: dict = field(default_factory=dict)

    # Computed retrieval views
    evidence_chunks: list[EvidenceChunk] = field(default_factory=list)
    skill_names_lower: list[str] = field(default_factory=list)
    full_text: str = ""  # concatenated retrieval-ready text

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "headline": self.headline,
            "current_title": self.current_title,
            "current_company": self.current_company,
            "years_of_experience": self.years_of_experience,
            "skill_count": len(self.skills),
            "career_event_count": len(self.career_events),
            "chunk_count": len(self.evidence_chunks),
        }


def build_candidate_profile(raw: dict[str, Any]) -> CandidateProfile:
    """Convert a raw JSON candidate dict into a fully structured CandidateProfile."""
    profile_raw = raw.get("profile") or {}

    cp = CandidateProfile(
        candidate_id=raw.get("candidate_id", ""),
        name=profile_raw.get("anonymized_name", ""),
        headline=profile_raw.get("headline", ""),
        summary=profile_raw.get("summary", ""),
        location=profile_raw.get("location", ""),
        country=profile_raw.get("country", ""),
        years_of_experience=float(profile_raw.get("years_of_experience", 0) or 0),
        current_title=profile_raw.get("current_title", ""),
        current_company=profile_raw.get("current_company", ""),
        current_company_size=profile_raw.get("current_company_size", ""),
        current_industry=profile_raw.get("current_industry", ""),
        redrob_signals=raw.get("redrob_signals") or {},
    )

    # Skills
    for s in raw.get("skills") or []:
        cp.skills.append(SkillEntry(
            name=s.get("name", ""),
            proficiency=s.get("proficiency", "unknown"),
            endorsements=int(s.get("endorsements", 0) or 0),
            duration_months=int(s.get("duration_months", 0) or 0),
        ))
    cp.skill_names_lower = [s.normalized_name for s in cp.skills]

    # Career events
    for job in raw.get("career_history") or []:
        cp.career_events.append(CareerEvent(
            company=job.get("company", ""),
            title=job.get("title", ""),
            start_date=job.get("start_date", ""),
            end_date=job.get("end_date"),
            duration_months=int(job.get("duration_months", 0) or 0),
            is_current=bool(job.get("is_current", False)),
            industry=job.get("industry", ""),
            company_size=job.get("company_size", ""),
            description=job.get("description", ""),
        ))

    # Education
    cp.education = raw.get("education") or []
    cp.certifications = raw.get("certifications") or []

    # Build evidence chunks
    cp.evidence_chunks = _build_evidence_chunks(cp)

    # Full text view
    cp.full_text = _build_full_text(cp)

    return cp


def _build_evidence_chunks(cp: CandidateProfile) -> list[EvidenceChunk]:
    """Build retrieval-indexable chunks from the candidate profile."""
    chunks: list[EvidenceChunk] = []
    cid = cp.candidate_id

    # Summary chunk
    if cp.summary:
        chunks.append(EvidenceChunk(
            candidate_id=cid,
            source_type="summary",
            section="Professional Summary",
            text=cp.summary,
            normalized_text=cp.summary.lower(),
            importance_weight=1.2,
        ))

    # Headline
    if cp.headline:
        chunks.append(EvidenceChunk(
            candidate_id=cid,
            source_type="headline",
            section="Headline",
            text=cp.headline,
            normalized_text=cp.headline.lower(),
            importance_weight=1.0,
        ))

    # Career history chunks (most important — weighted by recency)
    for i, event in enumerate(cp.career_events):
        weight = 1.5 if event.is_current else max(0.6, 1.2 - i * 0.15)
        section_label = f"Career @ {event.company}" if event.company else f"Career #{i+1}"
        text_parts = [event.title, event.company, event.description]
        text = " | ".join(filter(None, text_parts))
        chunks.append(EvidenceChunk(
            candidate_id=cid,
            source_type="career",
            section=section_label,
            title=event.title,
            company=event.company,
            start_date=event.start_date,
            end_date=event.end_date or "",
            duration_months=event.duration_months,
            is_current=event.is_current,
            importance_weight=weight,
            text=text,
            normalized_text=text.lower(),
        ))

    # Skills chunk (aggregate)
    if cp.skills:
        skill_text = ", ".join(s.name for s in cp.skills)
        chunks.append(EvidenceChunk(
            candidate_id=cid,
            source_type="skill",
            section="Skills",
            text=skill_text,
            normalized_text=skill_text.lower(),
            importance_weight=1.0,
        ))

    # Education chunks
    for edu in cp.education:
        edu_text = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
        chunks.append(EvidenceChunk(
            candidate_id=cid,
            source_type="education",
            section="Education",
            text=edu_text.strip(),
            normalized_text=edu_text.strip().lower(),
            importance_weight=0.5,
        ))

    return chunks


def _build_full_text(cp: CandidateProfile) -> str:
    """Concatenate all searchable text into one lowercase string."""
    parts: list[str] = [
        cp.summary,
        cp.headline,
        cp.current_title,
    ]
    parts.extend(s.name for s in cp.skills)
    for ev in cp.career_events:
        parts.extend([ev.description, ev.title, ev.company, ev.industry])
    for edu in cp.education:
        parts.extend([edu.get("field_of_study", ""), edu.get("degree", "")])
    return " ".join(filter(None, parts)).lower()
