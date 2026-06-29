"""
resume_parser.py — Local, deterministic resume parsing.
Converts PDF, DOCX, and JSON/JSONL files into internal candidate schema dicts
compatible with build_candidate_profile().

No external LLM/API calls. Uses heuristic extraction only.
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Known skill terms for extraction ─────────────────────────────────────────

_TECH_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "scala", "kotlin", "ruby", "php", "swift", "r", "matlab", "julia",
    "perl", "shell", "bash", "sql", "nosql", "html", "css",
    # ML/AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "neural networks",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "catboost", "huggingface", "transformers", "bert", "gpt",
    "llm", "large language model", "rag", "retrieval augmented generation",
    "embedding", "vector search", "faiss", "elasticsearch", "opensearch",
    "pinecone", "weaviate", "chromadb", "qdrant", "milvus",
    "sentence transformers", "cross-encoder", "reranking", "bm25",
    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "spark", "pyspark", "hadoop", "hive", "kafka", "airflow", "dbt",
    "snowflake", "bigquery", "redshift", "databricks",
    # Cloud & DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "github actions", "mlflow", "kubeflow", "sagemaker",
    # Web
    "react", "angular", "vue", "node.js", "express", "fastapi", "flask",
    "django", "spring boot", "graphql", "rest api",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "cassandra", "dynamodb",
    "neo4j", "sqlite",
    # Other
    "git", "linux", "agile", "scrum", "jira", "a/b testing",
    "data analysis", "data engineering", "data science", "mlops",
    "microservices", "api design", "system design",
}

_ROLE_KEYWORDS = re.compile(
    r"\b(engineer|developer|scientist|analyst|architect|manager|lead|director|"
    r"designer|consultant|administrator|specialist|intern|associate|"
    r"researcher|professor|instructor|coordinator|head|vp|cto|ceo|coo)\b",
    re.I,
)

_DEGREE_PATTERNS = re.compile(
    r"\b(b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|mba|bachelor|master|"
    r"doctorate|diploma|associate|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?)\b",
    re.I,
)

_EXPERIENCE_YEARS_PATTERN = re.compile(
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
    re.I,
)

_DATE_RANGE_PATTERN = re.compile(
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|"
    r"\d{4})\s*[-–—to]+\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?\s*\d{4}|\d{4}|present|current|now)",
    re.I,
)

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_PATTERN = re.compile(r"[\+]?[\d\s\-\(\)]{7,15}")
_URL_PATTERN = re.compile(r"https?://[\w\-\.]+\.\w+[/\w\-\.?=&#%]*")
_GITHUB_PATTERN = re.compile(r"github\.com/[\w\-]+", re.I)
_LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)


# ── PDF parsing ──────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str = "resume.pdf") -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed, cannot parse PDF")
        return ""

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.error(f"Failed to parse PDF '{filename}': {e}")
        return ""


# ── DOCX parsing ─────────────────────────────────────────────────────────────

def parse_docx(file_bytes: bytes, filename: str = "resume.docx") -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx not installed, cannot parse DOCX")
        return ""

    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        logger.error(f"Failed to parse DOCX '{filename}': {e}")
        return ""


# ── JSON / JSONL ingestion ───────────────────────────────────────────────────

def parse_json_candidates(file_bytes: bytes, filename: str = "") -> list[dict]:
    """Parse a JSON or JSONL file into candidate dicts.

    If the data already matches the internal schema, returns it directly.
    Otherwise wraps raw records with minimal normalization.
    """
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    # Try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_normalize_json_candidate(r, i) for i, r in enumerate(data)]
        elif isinstance(data, dict):
            return [_normalize_json_candidate(data, 0)]
    except json.JSONDecodeError:
        pass

    # Try JSONL (one JSON object per line)
    results = []
    for i, line in enumerate(text.split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            results.append(_normalize_json_candidate(obj, i))
        except json.JSONDecodeError:
            continue

    return results


def _normalize_json_candidate(raw: dict, index: int) -> dict:
    """Ensure a JSON record has the required internal schema fields."""
    # If it already has candidate_id + profile, return as-is
    if "candidate_id" in raw and "profile" in raw:
        return raw

    # Otherwise, build a compatible wrapper
    profile = raw.get("profile") or {}
    return {
        "candidate_id": raw.get("candidate_id", ""),
        "profile": {
            "anonymized_name": raw.get("name", profile.get("anonymized_name", "")),
            "headline": raw.get("headline", profile.get("headline", "")),
            "summary": raw.get("summary", profile.get("summary", "")),
            "current_title": raw.get("current_title", raw.get("title", profile.get("current_title", ""))),
            "current_company": raw.get("current_company", raw.get("company", profile.get("current_company", ""))),
            "years_of_experience": raw.get("years_of_experience", profile.get("years_of_experience", 0)),
            "location": raw.get("location", profile.get("location", "")),
            "country": raw.get("country", profile.get("country", "")),
        },
        "skills": raw.get("skills", []),
        "career_history": raw.get("career_history", []),
        "education": raw.get("education", []),
        "redrob_signals": raw.get("redrob_signals", {}),
    }


# ── Core text structuring ────────────────────────────────────────────────────

def structure_resume_text(raw_text: str, candidate_id: str) -> dict:
    """Convert unstructured resume text into the internal candidate schema dict.

    Uses deterministic heuristics — no LLM calls.
    """
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if not lines:
        return _empty_candidate(candidate_id)

    # ── Name: first short line that looks like a name ──
    name = _extract_name(lines)

    # ── Contact info (for filtering, not schema) ──
    emails = _EMAIL_PATTERN.findall(raw_text)
    github = _GITHUB_PATTERN.findall(raw_text)
    linkedin = _LINKEDIN_PATTERN.findall(raw_text)

    # ── Current title: look for role-like phrases near the top ──
    title = _extract_title(lines[:10])

    # ── Skills ──
    skills = _extract_skills(raw_text)

    # ── Career history ──
    career = _extract_career_history(lines)

    # ── Education ──
    education = _extract_education(lines)

    # ── Years of experience ──
    years_exp = _extract_years_experience(raw_text, career)

    # ── Summary: first paragraph-like block ──
    summary = _extract_summary(lines)

    # ── Current company from career ──
    current_company = ""
    if career:
        current_company = career[0].get("company", "")

    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": name,
            "headline": title,
            "summary": summary,
            "current_title": title,
            "current_company": current_company,
            "years_of_experience": years_exp,
            "location": "",
            "country": "",
        },
        "skills": [{"name": s} for s in skills],
        "career_history": career,
        "education": education,
        "redrob_signals": {},
    }


def _empty_candidate(candidate_id: str) -> dict:
    """Return an empty candidate record."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Unknown",
            "headline": "",
            "summary": "",
            "current_title": "",
            "current_company": "",
            "years_of_experience": 0,
            "location": "",
            "country": "",
        },
        "skills": [],
        "career_history": [],
        "education": [],
        "redrob_signals": {},
    }


def _extract_name(lines: list[str]) -> str:
    """Extract candidate name from the first few lines."""
    for line in lines[:5]:
        # Skip lines that are obviously not names
        if "@" in line or "http" in line.lower() or len(line) > 60:
            continue
        if _PHONE_PATTERN.fullmatch(line.replace(" ", "")):
            continue
        # A name is typically 2-4 words, all starting with uppercase
        words = line.split()
        if 1 <= len(words) <= 4:
            if all(w[0].isupper() or w in ("van", "de", "von", "al", "el") for w in words if w):
                # Not a title/header
                if not _DEGREE_PATTERNS.search(line) and not any(
                    kw in line.lower() for kw in ("resume", "curriculum", "cv", "phone", "email")
                ):
                    return line
    return lines[0][:50] if lines else "Unknown"


def _extract_title(lines: list[str]) -> str:
    """Extract the most likely current job title from top lines."""
    for line in lines:
        if _ROLE_KEYWORDS.search(line) and len(line) < 80:
            # Clean common prefixes
            clean = re.sub(r"^(current|present|title|role|position)\s*[:|-]\s*", "", line, flags=re.I)
            return clean.strip()
    return ""


def _extract_skills(text: str) -> list[str]:
    """Extract skills by matching against known tech terms."""
    text_lower = text.lower()
    found = []

    # Match multi-word skills first
    for skill in sorted(_TECH_SKILLS, key=len, reverse=True):
        if skill in text_lower and skill not in [s.lower() for s in found]:
            # Capitalize nicely
            found.append(skill.title() if len(skill) > 3 else skill.upper())

    # Also look for a Skills section and extract additional terms
    skills_section = _extract_section(text, r"(?:skills|technologies|technical\s+skills|core\s+competencies)")
    if skills_section:
        for item in re.split(r"[,;|•·\n]", skills_section):
            item = item.strip().strip("-").strip("*").strip()
            if 2 <= len(item) <= 40 and item.lower() not in [s.lower() for s in found]:
                found.append(item)

    return found[:30]  # cap at 30 skills


def _extract_section(text: str, header_pattern: str) -> str:
    """Extract text from a labeled section."""
    pat = rf"(?:^|\n)\s*{header_pattern}\s*[:\-]?\s*\n(.*?)(?:\n\s*(?:experience|education|projects|certifications|awards|references|interests)\s*[:\-]?\s*\n|\Z)"
    m = re.search(pat, text, re.I | re.S)
    return m.group(1).strip() if m else ""


def _extract_career_history(lines: list[str]) -> list[dict]:
    """Extract career history entries from resume lines."""
    careers = []
    i = 0

    # Find experience section
    exp_start = -1
    for idx, line in enumerate(lines):
        if re.match(r"^(?:experience|work\s+experience|employment|professional\s+experience|career)\s*[:\-]?\s*$", line, re.I):
            exp_start = idx + 1
            break

    if exp_start < 0:
        # Try to find date ranges anywhere as career signals
        for idx, line in enumerate(lines):
            dates = _DATE_RANGE_PATTERN.search(line)
            if dates and _ROLE_KEYWORDS.search(line):
                exp_start = max(0, idx - 1)
                break

    if exp_start < 0:
        return []

    # Parse entries from the experience section
    current_entry = None
    for line in lines[exp_start:]:
        # Stop at next major section
        if re.match(r"^(?:education|skills|projects|certifications|awards|references|interests|languages)\s*[:\-]?\s*$", line, re.I):
            break

        dates = _DATE_RANGE_PATTERN.search(line)
        has_role = _ROLE_KEYWORDS.search(line)

        if dates or (has_role and len(line) < 100):
            # Save previous entry
            if current_entry:
                careers.append(current_entry)

            start_date = dates.group(1) if dates else ""
            end_date = dates.group(2) if dates else ""
            is_current = bool(end_date and re.match(r"present|current|now", end_date, re.I))

            # Try to split "Title at Company" or "Company — Title"
            title, company = _split_role_company(line, dates)

            current_entry = {
                "company": company,
                "title": title,
                "start_date": start_date,
                "end_date": None if is_current else end_date,
                "duration_months": 0,
                "is_current": is_current,
                "industry": "",
                "company_size": "",
                "description": "",
            }
        elif current_entry:
            # Append as description
            if line and not line.startswith(("•", "-", "*", "–")):
                current_entry["description"] += " " + line
            else:
                current_entry["description"] += " " + line.lstrip("•-*– ").strip()

    if current_entry:
        careers.append(current_entry)

    # Clean descriptions
    for c in careers:
        c["description"] = c["description"].strip()[:500]

    return careers[:10]  # cap at 10 entries


def _split_role_company(line: str, dates_match) -> tuple[str, str]:
    """Try to split a line into (title, company)."""
    # Remove the date range portion
    clean = line
    if dates_match:
        clean = line[:dates_match.start()] + line[dates_match.end():]
    clean = clean.strip().strip("-–—|,").strip()

    # Try "Title at Company" or "Title, Company"
    for sep in [" at ", " @ ", " - ", " – ", " — ", " | ", ", "]:
        if sep in clean:
            parts = clean.split(sep, 1)
            if _ROLE_KEYWORDS.search(parts[0]):
                return parts[0].strip(), parts[1].strip()
            elif _ROLE_KEYWORDS.search(parts[1]):
                return parts[1].strip(), parts[0].strip()
            return parts[0].strip(), parts[1].strip()

    # Just use the whole line as title
    return clean, ""


def _extract_education(lines: list[str]) -> list[dict]:
    """Extract education entries."""
    education = []

    # Find education section
    edu_start = -1
    for idx, line in enumerate(lines):
        if re.match(r"^(?:education|academic|qualifications)\s*[:\-]?\s*$", line, re.I):
            edu_start = idx + 1
            break

    if edu_start < 0:
        # Scan entire text for degree patterns
        for line in lines:
            if _DEGREE_PATTERNS.search(line):
                edu = _parse_edu_line(line)
                if edu:
                    education.append(edu)
        return education[:5]

    for line in lines[edu_start:]:
        if re.match(r"^(?:experience|skills|projects|certifications|awards)\s*[:\-]?\s*$", line, re.I):
            break
        if _DEGREE_PATTERNS.search(line) or any(kw in line.lower() for kw in ("university", "college", "institute", "school")):
            edu = _parse_edu_line(line)
            if edu:
                education.append(edu)

    return education[:5]


def _parse_edu_line(line: str) -> Optional[dict]:
    """Parse a single education line into a structured dict."""
    degree_match = _DEGREE_PATTERNS.search(line)
    degree = degree_match.group(0) if degree_match else ""

    # Try to extract field of study and institution
    field = ""
    institution = ""

    # "B.S. in Computer Science from MIT" pattern
    in_match = re.search(r"\bin\s+(.+?)(?:\s+(?:from|at)\s+|\s*[,|]\s*)", line, re.I)
    if in_match:
        field = in_match.group(1).strip()

    from_match = re.search(r"(?:from|at)\s+(.+?)(?:\s*[,|]\s*\d|\s*$)", line, re.I)
    if from_match:
        institution = from_match.group(1).strip()

    if not institution:
        # Check for known institution keywords
        for part in re.split(r"[,|;]", line):
            part = part.strip()
            if any(kw in part.lower() for kw in ("university", "college", "institute", "school", "iit", "mit", "stanford")):
                institution = part
                break

    if not degree and not institution:
        return None

    return {
        "degree": degree,
        "field_of_study": field,
        "institution": institution,
    }


def _extract_years_experience(text: str, career: list[dict]) -> float:
    """Extract years of experience from text or career history."""
    # Direct mention
    matches = _EXPERIENCE_YEARS_PATTERN.findall(text)
    if matches:
        return float(max(int(m) for m in matches))

    # Estimate from career history count
    if career:
        return float(len(career) * 2.5)  # rough estimate

    return 0.0


def _extract_summary(lines: list[str]) -> str:
    """Extract the professional summary from early lines."""
    # Look for explicit summary section
    for i, line in enumerate(lines):
        if re.match(r"^(?:summary|about|profile|objective|about\s+me)\s*[:\-]?\s*$", line, re.I):
            # Collect next few lines
            summary_lines = []
            for j in range(i + 1, min(i + 6, len(lines))):
                if re.match(r"^(?:experience|education|skills|work)\s*[:\-]?\s*$", lines[j], re.I):
                    break
                summary_lines.append(lines[j])
            return " ".join(summary_lines)[:500]

    # Fallback: use lines 2-4 if they look like prose (>30 chars, not a header)
    prose = []
    for line in lines[1:6]:
        if len(line) > 30 and not _ROLE_KEYWORDS.search(line) and "@" not in line:
            prose.append(line)
    return " ".join(prose)[:500] if prose else ""


# ── Google Drive public folder fetch ─────────────────────────────────────────

def fetch_drive_folder(folder_url: str) -> list[tuple[str, bytes]]:
    """Fetch files from a public Google Drive folder.

    Returns a list of (filename, file_bytes) tuples.
    Only fetches PDF and DOCX files.
    """
    import requests

    # Extract folder ID from URL
    folder_id = _extract_drive_folder_id(folder_url)
    if not folder_id:
        raise ValueError(f"Could not extract Google Drive folder ID from: {folder_url}")

    # Use the Drive API files list endpoint (works for public folders)
    api_url = f"https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType)",
        "key": "",  # works without key for public folders in many cases
    }

    # Alternative: scrape the public folder page
    # For truly public folders, use the export endpoint directly
    files = []

    try:
        # Try listing via the public share page
        share_url = f"https://drive.google.com/drive/folders/{folder_id}"
        resp = requests.get(share_url, timeout=15)
        if resp.status_code != 200:
            raise ValueError(f"Cannot access Drive folder (HTTP {resp.status_code})")

        # Extract file IDs from the page content
        file_ids = re.findall(r'/file/d/([a-zA-Z0-9_-]+)', resp.text)
        file_ids = list(dict.fromkeys(file_ids))  # deduplicate preserving order

        for fid in file_ids[:20]:  # cap at 20 files
            try:
                # Download via direct download link
                dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
                dl_resp = requests.get(dl_url, timeout=30, allow_redirects=True)
                if dl_resp.status_code == 200 and len(dl_resp.content) > 100:
                    # Guess filename from content-disposition or use ID
                    cd = dl_resp.headers.get("content-disposition", "")
                    fname_match = re.search(r'filename="?(.+?)"?$', cd)
                    fname = fname_match.group(1) if fname_match else f"drive_{fid}"

                    # Only keep PDF/DOCX
                    if fname.lower().endswith((".pdf", ".docx")):
                        files.append((fname, dl_resp.content))
                    elif dl_resp.content[:4] == b"%PDF":
                        files.append((f"{fname}.pdf", dl_resp.content))
            except Exception as e:
                logger.warning(f"Failed to download Drive file {fid}: {e}")
                continue

    except Exception as e:
        raise ValueError(f"Failed to access Drive folder: {e}")

    return files


def _extract_drive_folder_id(url: str) -> Optional[str]:
    """Extract the folder ID from various Google Drive URL formats."""
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/.*[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


# ── Main entry point ─────────────────────────────────────────────────────────

def parse_resume_file(file_bytes: bytes, filename: str, candidate_id: str) -> list[dict]:
    """Parse a single resume file into one or more candidate dicts.

    Returns a list because JSON/JSONL files may contain multiple candidates.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text = parse_pdf(file_bytes, filename)
        if not text:
            logger.warning(f"Empty PDF: {filename}")
            return []
        cand = structure_resume_text(text, candidate_id)
        if not cand["profile"]["anonymized_name"] or cand["profile"]["anonymized_name"] == "Unknown":
            cand["profile"]["anonymized_name"] = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        return [cand]

    elif ext == ".docx":
        text = parse_docx(file_bytes, filename)
        if not text:
            logger.warning(f"Empty DOCX: {filename}")
            return []
        cand = structure_resume_text(text, candidate_id)
        if not cand["profile"]["anonymized_name"] or cand["profile"]["anonymized_name"] == "Unknown":
            cand["profile"]["anonymized_name"] = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        return [cand]

    elif ext in (".json", ".jsonl"):
        candidates = parse_json_candidates(file_bytes, filename)
        return candidates

    else:
        logger.warning(f"Unsupported file type: {filename} ({ext})")
        return []
