"""
chunking.py — Chunking pipeline and event assembler.
Produces normalised chunk metadata for retrieval indexing.
"""
from __future__ import annotations

from typing import Any
from ranking.candidate_model import CandidateProfile, EvidenceChunk


def chunk_candidate(profile: CandidateProfile, max_chunk_len: int = 512) -> list[EvidenceChunk]:
    """
    Split long evidence chunks into retrieval-sized pieces.
    Short chunks are passed through as-is.
    Returns a flat list of chunks ready for indexing.
    """
    output: list[EvidenceChunk] = []

    for chunk in profile.evidence_chunks:
        text = chunk.text.strip()
        if not text:
            continue

        if len(text) <= max_chunk_len:
            output.append(chunk)
            continue

        # Split on sentence boundaries
        sentences = _split_sentences(text)
        current_buf: list[str] = []
        current_len = 0
        sub_idx = 0

        for sent in sentences:
            if current_len + len(sent) > max_chunk_len and current_buf:
                sub_text = " ".join(current_buf)
                output.append(EvidenceChunk(
                    candidate_id=chunk.candidate_id,
                    source_type=chunk.source_type,
                    section=f"{chunk.section} [part {sub_idx + 1}]",
                    title=chunk.title,
                    company=chunk.company,
                    start_date=chunk.start_date,
                    end_date=chunk.end_date,
                    duration_months=chunk.duration_months,
                    is_current=chunk.is_current,
                    importance_weight=chunk.importance_weight,
                    text=sub_text,
                    normalized_text=sub_text.lower(),
                ))
                current_buf = []
                current_len = 0
                sub_idx += 1

            current_buf.append(sent)
            current_len += len(sent) + 1

        if current_buf:
            sub_text = " ".join(current_buf)
            suffix = f" [part {sub_idx + 1}]" if sub_idx > 0 else ""
            output.append(EvidenceChunk(
                candidate_id=chunk.candidate_id,
                source_type=chunk.source_type,
                section=f"{chunk.section}{suffix}",
                title=chunk.title,
                company=chunk.company,
                start_date=chunk.start_date,
                end_date=chunk.end_date,
                duration_months=chunk.duration_months,
                is_current=chunk.is_current,
                importance_weight=chunk.importance_weight,
                text=sub_text,
                normalized_text=sub_text.lower(),
            ))

    return output


def assemble_career_timeline(profile: CandidateProfile) -> list[dict[str, Any]]:
    """
    Assemble a chronologically ordered career timeline from career events.
    Returns a list of event dicts ordered by start_date descending (most recent first).
    """
    events = []
    for ev in profile.career_events:
        events.append({
            "company": ev.company,
            "title": ev.title,
            "start_date": ev.start_date,
            "end_date": ev.end_date,
            "duration_months": ev.duration_months,
            "is_current": ev.is_current,
            "industry": ev.industry,
        })
    # Sort by start_date descending (most recent first); current roles first
    events.sort(key=lambda e: (not e["is_current"], -(e.get("start_date") or "0").__hash__()))
    return events


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
