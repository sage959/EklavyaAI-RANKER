"""
jd_parser.py — Structured JD parser.
Converts raw job description text into a normalized, typed schema.
Uses deterministic heuristic extraction with keyword/phrase normalization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Normalisation tables ─────────────────────────────────────────────────────

_SKILL_ALIASES: dict[str, str] = {
    "sentence-transformers": "sentence_transformers",
    "sentencetransformers": "sentence_transformers",
    "scikit-learn": "sklearn",
    "sci-kit learn": "sklearn",
    "hugging face": "huggingface",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "elastic search": "elasticsearch",
    "open search": "opensearch",
    "vector db": "vector_database",
    "vector store": "vector_database",
    "langchain": "langchain",
    "lang chain": "langchain",
    "llm": "large_language_model",
    "large language model": "large_language_model",
    "gpt": "large_language_model",
}

# Proper casing for known tech terms (used in display)
_TECH_CASING: dict[str, str] = {
    "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "ci/cd": "CI/CD", "c++": "C++", "c#": "C#",
    "javascript": "JavaScript", "typescript": "TypeScript",
    "graphql": "GraphQL", "fastapi": "FastAPI", "github": "GitHub",
    "gitlab": "GitLab", "mysql": "MySQL", "postgresql": "PostgreSQL",
    "mongodb": "MongoDB", "dynamodb": "DynamoDB", "bigquery": "BigQuery",
    "elasticsearch": "Elasticsearch", "opensearch": "OpenSearch",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "sklearn": "Scikit-learn",
    "pandas": "Pandas", "numpy": "NumPy", "scipy": "SciPy",
    "docker": "Docker", "kubernetes": "Kubernetes", "terraform": "Terraform",
    "jenkins": "Jenkins", "airflow": "Airflow", "kafka": "Kafka",
    "redis": "Redis", "spark": "Spark", "hadoop": "Hadoop",
    "pyspark": "PySpark", "snowflake": "Snowflake", "databricks": "Databricks",
    "faiss": "FAISS", "pinecone": "Pinecone", "weaviate": "Weaviate",
    "chromadb": "ChromaDB", "qdrant": "Qdrant", "milvus": "Milvus",
    "mlflow": "MLflow", "kubeflow": "Kubeflow", "sagemaker": "SageMaker",
    "huggingface": "HuggingFace", "langchain": "LangChain",
    "react": "React", "angular": "Angular", "vue": "Vue",
    "node.js": "Node.js", "express": "Express", "django": "Django",
    "flask": "Flask", "spring boot": "Spring Boot",
    "rest apis": "REST APIs", "rest api": "REST API",
    "sql": "SQL", "nosql": "NoSQL", "html": "HTML", "css": "CSS",
    "python": "Python", "java": "Java", "go": "Go", "golang": "Go",
    "rust": "Rust", "scala": "Scala", "kotlin": "Kotlin",
    "r": "R", "matlab": "MATLAB", "bash": "Bash",
    "llm": "LLM", "llms": "LLMs", "nlp": "NLP", "ml": "ML", "ai": "AI",
    "rag": "RAG", "bm25": "BM25", "hnsw": "HNSW",
    "ndcg": "nDCG", "mrr": "MRR", "api": "API", "apis": "APIs",
    "etl": "ETL", "mlops": "MLOps", "devops": "DevOps",
    "a/b testing": "A/B Testing", "ab testing": "A/B Testing",
    "microservices": "Microservices", "agile": "Agile", "scrum": "Scrum",
    "git": "Git", "linux": "Linux", "jira": "Jira",
    "sentence_transformers": "Sentence Transformers",
    "large_language_model": "LLM", "vector_database": "Vector DB",
    "cross-encoder": "Cross-Encoder", "bi-encoder": "Bi-Encoder",
    "colbert": "ColBERT", "bert": "BERT", "gpt": "GPT",
    "neo4j": "Neo4j", "cassandra": "Cassandra", "sqlite": "SQLite",
    "tableau": "Tableau", "power bi": "Power BI",
    "figma": "Figma", "jira": "Jira",
    "deep learning": "Deep Learning", "machine learning": "Machine Learning",
    "computer vision": "Computer Vision", "data science": "Data Science",
    "data engineering": "Data Engineering",
    "natural language processing": "NLP",
    "information retrieval": "Information Retrieval",
    "semantic search": "Semantic Search", "vector search": "Vector Search",
    "embedding": "Embeddings", "embeddings": "Embeddings",
    "reranking": "Reranking", "ranking": "Ranking",
    "retrieval": "Retrieval", "search": "Search",
}

# Phrases to strip from the beginning of extracted tags
_STRIP_PREFIXES = re.compile(
    r"^(?:experience\s+(?:with|in)\s+|proficiency\s+(?:with|in)\s+|knowledge\s+of\s+|"
    r"familiarity\s+with\s+|understanding\s+of\s+|strong\s+(?:understanding|knowledge|experience)\s+(?:of|in|with)\s+|"
    r"ability\s+to\s+|proven\s+(?:experience|ability|track\s+record)\s+(?:in|with|of)?\s*|"
    r"hands-on\s+experience\s+(?:with|in)\s+|expertise\s+in\s+|background\s+in\s+)",
    re.I,
)

# Filler words/phrases to exclude entirely from tags
_FILLER_EXCLUSIONS = {
    "benefits", "competitive salary", "health", "wellness", "health and wellness",
    "flexible work", "flexible work arrangements", "work-life balance",
    "collaborative environment", "learning opportunities", "career growth",
    "professional development", "team player", "fast-paced environment",
    "equal opportunity", "diversity", "inclusive", "company culture",
    "compensation", "bonus", "equity", "stock options", "401k",
    "dental", "vision", "insurance", "pto", "vacation",
    "such as", "and", "or", "etc", "related field",
    "a related field", "or a related field", "and more",
    "strong analytical abilities", "excellent communication",
    "excellent communication skills", "strong communication skills",
    "problem solving", "problem-solving skills", "attention to detail",
    "self-motivated", "team-oriented", "results-driven",
    "bachelor's degree", "master's degree", "phd",
}

_SENIORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(staff|principal|distinguished)\b", re.I), "staff"),
    (re.compile(r"\bsenior\b", re.I), "senior"),
    (re.compile(r"\b(mid[- ]?level|intermediate)\b", re.I), "mid"),
    (re.compile(r"\b(junior|entry[- ]?level|associate)\b", re.I), "junior"),
]

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "retrieval": ["retrieval", "search", "ranking", "information retrieval", "ir "],
    "ml_production": ["production ml", "ml engineering", "mlops", "deployment", "serving"],
    "nlp": ["nlp", "natural language", "text processing", "language model"],
    "recommendation": ["recommendation", "recommender", "collaborative filtering"],
    "cv": ["computer vision", "image", "object detection"],
    "data_engineering": ["data pipeline", "etl", "data engineering", "airflow", "spark"],
}

_INTENT_PATTERNS: dict[str, list[str]] = {
    "builder": ["ship", "build", "hands-on", "deploy", "production", "launch", "end-to-end"],
    "researcher": ["research", "experiment", "publish", "paper", "novel", "state-of-the-art"],
    "retrieval_specialist": ["retrieval", "search", "ranking", "reranking", "vector", "faiss", "elasticsearch"],
    "platform_engineer": ["platform", "infrastructure", "scalab", "microservice", "api", "fastapi"],
    "evaluator": ["evaluation", "metrics", "a/b test", "ndcg", "mrr", "benchmark"],
}


@dataclass
class ParsedJD:
    """Normalised job description representation."""
    role_title: str = ""
    seniority: str = "mid"  # junior | mid | senior | staff
    min_experience_years: float = 0.0
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    tooling: list[str] = field(default_factory=list)
    education_requirements: list[str] = field(default_factory=list)
    location: str = ""
    work_mode: str = ""  # remote | hybrid | onsite | ""
    intent_facets: dict[str, float] = field(default_factory=dict)
    ranking_priorities: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "role_title": self.role_title,
            "seniority": self.seniority,
            "min_experience_years": self.min_experience_years,
            "must_have_skills": self.must_have_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "responsibilities": self.responsibilities,
            "domains": self.domains,
            "tooling": self.tooling,
            "education_requirements": self.education_requirements,
            "location": self.location,
            "work_mode": self.work_mode,
            "intent_facets": self.intent_facets,
            "ranking_priorities": self.ranking_priorities,
        }


def _normalise_skill(s: str) -> str:
    """Lowercase, strip, and resolve aliases."""
    s = s.strip().lower()
    return _SKILL_ALIASES.get(s, s)


def _clean_tag(raw: str) -> str:
    """Clean a raw extracted phrase into a display-ready tag."""
    s = raw.strip()
    # Strip leading "experience with", "knowledge of", etc.
    s = _STRIP_PREFIXES.sub("", s).strip()
    # Strip trailing punctuation junk
    s = re.sub(r"[\.,;:!?]+$", "", s).strip()
    # Strip leading bullets/dashes
    s = re.sub(r"^[-–—•*\d\.]+\s*", "", s).strip()
    # Strip trailing "such as" leftovers
    s = re.sub(r"\s+such\s+as\s*$", "", s, flags=re.I).strip()
    # Strip wrapping parentheses
    s = re.sub(r"^\((.+)\)$", r"\1", s).strip()
    return s


def _split_compound_tag(tag: str) -> list[str]:
    """Split a long compound phrase into shorter individual tags.
    
    E.g. 'proficiency in one or more programming languages such as Java, Python, JavaScript, C++ or Go.' 
         → ['Java', 'Python', 'JavaScript', 'C++', 'Go']
    E.g. 'cloud platforms (AWS, GCP or Azure)' → ['AWS', 'GCP', 'Azure']
    E.g. 'frontend frameworks such as React, Angular or Vue' → ['React', 'Angular', 'Vue']
    """
    tag = _clean_tag(tag)
    if not tag:
        return []
    
    # If short enough already, return as-is
    words = tag.split()
    if len(words) <= 3:
        return [tag]
    
    # Step 1: Check for "such as X, Y, Z" pattern — extract just the listed items
    such_as = re.search(r"such\s+as\s+(.+)$", tag, re.I)
    if such_as:
        listed = such_as.group(1)
        items = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", listed)
        results = [_clean_tag(p) for p in items if _clean_tag(p) and len(_clean_tag(p)) >= 1]
        if results:
            return results
    
    # Step 2: Check for parenthetical lists: "cloud platforms (AWS, GCP or Azure)"
    paren_match = re.search(r"(.+?)\s*\((.+?)\)", tag)
    if paren_match:
        inner = paren_match.group(2)
        inner_items = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", inner)
        results = [_clean_tag(p) for p in inner_items if _clean_tag(p) and len(_clean_tag(p)) >= 1]
        if results:
            return results
    
    # Step 3: Strip leading filler phrases and check if remainder is short
    stripped = re.sub(
        r"^(?:\d+\+?\s*years?\s*(?:of\s+)?(?:experience\s+(?:in|with)\s+)?|"
        r"strong\s+(?:proficiency|understanding|knowledge|experience)\s+(?:in|of|with)\s+|"
        r"(?:proficiency|experience|familiarity|knowledge)\s+(?:in|of|with)\s+)",
        "", tag, flags=re.I
    ).strip()
    stripped = _clean_tag(stripped)
    
    if stripped and len(stripped.split()) <= 4:
        return [stripped]
    
    # Step 4: Split on commas/or/and
    if stripped:
        parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", stripped)
    else:
        parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", tag)
    
    results = [_clean_tag(p) for p in parts if _clean_tag(p) and len(_clean_tag(p)) >= 2]
    
    # If splitting produced good short results, use them
    if results and all(len(r.split()) <= 4 for r in results):
        return results
    
    # Step 5: Last resort — try to extract just the last meaningful phrase (up to 4 words)
    if len(words) > 4:
        return [" ".join(words[-4:]).strip(".,;:")]
    return [tag]


def _apply_tech_casing(tag: str) -> str:
    """Apply proper casing for display."""
    lower = tag.strip().lower()
    # Exact match in casing table
    if lower in _TECH_CASING:
        return _TECH_CASING[lower]
    # Check for multi-word prefix matches (e.g., "a/b testing" inside "a/b testing frameworks")
    for key, cased in _TECH_CASING.items():
        if " " in key and lower.startswith(key + " "):
            remainder = tag[len(key):].strip()
            return cased + " " + " ".join(w.capitalize() for w in remainder.split())
        if " " in key and lower.startswith(key):
            return cased
    # Title case fallback, preserving known sub-terms
    words = tag.split()
    result = []
    for w in words:
        wl = w.lower()
        if wl in _TECH_CASING:
            result.append(_TECH_CASING[wl])
        elif len(wl) <= 3 and wl.isalpha():
            result.append(w.upper() if wl in {"api", "sql", "aws", "gcp", "rag", "nlp", "ml", "ai", "etl"} else w.capitalize())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _is_filler(tag: str) -> bool:
    """Check if a tag is generic filler / non-skill content."""
    lower = tag.strip().lower()
    if lower in _FILLER_EXCLUSIONS:
        return True
    if len(lower) < 2:
        return True
    if len(lower) > 200:
        return True
    # Single generic words
    if lower in {"the", "a", "an", "is", "are", "be", "to", "of", "in", "with", "for", "on", "at", "by"}:
        return True
    return False


def _is_near_duplicate(a: str, b: str) -> bool:
    """Check if two tags are near-duplicates (word-boundary match, not partial)."""
    if a == b:
        return True
    # Only treat as duplicate if one is a complete word match within the other
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 3:
        return False
    # Check for whole-word containment
    pattern = rf"\b{re.escape(shorter)}\b"
    return bool(re.search(pattern, longer, re.I))


def _dedup_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate tags, handling near-duplicates and substrings."""
    seen_norms: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = _clean_tag(item)
        if not cleaned or _is_filler(cleaned):
            continue
        norm = _normalise_skill(cleaned)
        if not norm or norm in seen_norms:
            continue
        # Check substring containment (avoid "python" and "proficiency in python")
        already_covered = any(_is_near_duplicate(norm, existing) for existing in seen_norms)
        if already_covered:
            continue
        seen_norms.add(norm)
        out.append(norm)
    return out


def _cross_dedup(must: list[str], nice: list[str]) -> tuple[list[str], list[str]]:
    """Remove items from nice-to-have that already appear in must-have."""
    must_norms = {_normalise_skill(s) for s in must}
    deduped_nice = []
    for item in nice:
        norm = _normalise_skill(item)
        # Skip if this exact norm or a very similar one is in must-have
        if norm in must_norms:
            continue
        # Check near-duplicate overlap
        if any(_is_near_duplicate(norm, m) for m in must_norms):
            continue
        deduped_nice.append(item)
    return must, deduped_nice


def clean_tags_for_display(tags: list[str], max_count: int = 12) -> list[str]:
    """Final pass: split compound tags, apply casing, enforce limits."""
    result: list[str] = []
    seen_lower: set[str] = set()
    
    for tag in tags:
        sub_tags = _split_compound_tag(tag)
        for st in sub_tags:
            st = _clean_tag(st)
            if _is_filler(st):
                continue
            display = _apply_tech_casing(st)
            dl = display.lower()
            if dl in seen_lower:
                continue
            # Check near-duplicate
            skip = any(_is_near_duplicate(dl, existing) for existing in seen_lower)
            if skip:
                continue
            seen_lower.add(dl)
            result.append(display)
            if len(result) >= max_count:
                return result
    return result


def _extract_experience(text: str) -> float:
    """Extract minimum experience requirement from text."""
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience",
        r"minimum\s+(\d+)\s*(?:years?|yrs?)",
        r"at\s+least\s+(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\+\s*(?:years?|yrs?)",  # catch "3+ years" without trailing "experience"
    ]
    years: list[int] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            years.append(int(m.group(1)))
    return float(min(years)) if years else 0.0


# Regex to detect a new section header line (used as extraction boundary)
_NEXT_SECTION_RE = re.compile(
    r"^(?:must[- ]?have|nice[- ]?to[- ]?have|required(?:\s+skills)?|requirements?|"
    r"qualifications?|preferred(?:\s+(?:qualifications|skills))?|desired(?:\s+(?:qualifications|skills))?|"
    r"responsibilities?|education|key\s+skills|technical\s+skills|essential\s+skills|"
    r"mandatory\s+skills|good[- ]?to[- ]?have|additional\s+(?:qualifications|skills)|"
    r"bonus\s+skills|benefits?|compensation|about\s+(?:us|the)|what\s+you|how\s+to\s+apply)"
    r"\s*[:\-\u2013\u2014]\s*$",
    re.I,
)

# Generic header-fragment words to filter out of extracted items
_HEADER_FRAGMENT_RE = re.compile(
    r"^(?:skills?|tools?|technologies?|frameworks?|stack)\s*:?\s*$", re.I
)


def _extract_skills_from_section(text: str, section_header: str) -> list[str]:
    """Pull bullet-point or comma-separated skills from a labelled section."""
    pattern = rf"{section_header}\s*[:\-]?\s*(.*?)(?:\r?\n\r?\n|\Z)"
    m = re.search(pattern, text, re.I | re.S)
    if not m:
        return []
    block = m.group(1)
    items: list[str] = []
    for line in block.split("\n"):
        raw_line = line.strip().rstrip("\r")
        # Stop if we hit a new section header
        if raw_line and _NEXT_SECTION_RE.match(raw_line):
            break
        line = re.sub(r"^[\s\-\*\•\d\.]+", "", raw_line).strip()
        if not line:
            continue
        # Skip lines that are just leftover header fragments (e.g. "Skills:")
        if _HEADER_FRAGMENT_RE.match(line):
            continue
        # Keep the full line intact — let _split_compound_tag handle splitting
        items.append(line)
    return items


def _detect_seniority(text: str) -> str:
    for pat, level in _SENIORITY_PATTERNS:
        if pat.search(text):
            return level
    return "mid"


def _detect_domains(text: str) -> list[str]:
    text_lower = text.lower()
    found: list[str] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(domain)
    return found


def _detect_intents(text: str) -> dict[str, float]:
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for intent, keywords in _INTENT_PATTERNS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[intent] = min(1.0, hits / len(keywords))
    return scores


def _detect_work_mode(text: str) -> str:
    text_lower = text.lower()
    if "remote" in text_lower:
        return "remote"
    if "hybrid" in text_lower:
        return "hybrid"
    if "onsite" in text_lower or "on-site" in text_lower or "in-office" in text_lower:
        return "onsite"
    return ""

# Patterns that signal secondary/optional requirements within any line.
# Each pattern captures the relevant skill/technology payload.
_SECONDARY_PAYLOAD_PATTERNS = [
    # "Familiarity with X" → X
    re.compile(r"(?:familiarity|familiar)\s+with\s+(.+?)(?:\s+is\s+|\s*$)", re.I),
    # "Exposure to X" → X
    re.compile(r"exposure\s+to\s+(.+?)(?:\s+is\s+|\s*$)", re.I),
    # "Nice/good to have: X" → X
    re.compile(r"(?:nice|good)\s+to\s+have[:\s]+(.+)", re.I),
    # "Preferred: X" → X
    re.compile(r"preferred[:\s]+(.+)", re.I),
    # "Bonus: X" → X
    re.compile(r"bonus[:\s]+(.+)", re.I),
    # "Ideally with X" → X
    re.compile(r"ideally\s+(?:with|having)\s+(.+)", re.I),
    # "Some experience with X" → X
    re.compile(r"(?:some|basic)\s+(?:experience|knowledge)\s+(?:with|of|in)\s+(.+?)(?:\s+is\s+|\s*$)", re.I),
    # "Awareness/understanding of X" → X
    re.compile(r"(?:awareness|understanding)\s+of\s+(.+?)(?:\s+is\s+|\s*$)", re.I),
]

# Pattern to detect "X is a plus/bonus" and extract X
_PLUS_BONUS_PATTERN = re.compile(
    r"(.+?)\s+(?:is|would\s+be)\s+a\s+(?:plus|bonus)\b",
    re.I,
)


def _infer_secondary_tags(raw_text: str, must_have_lower: set[str]) -> list[str]:
    """Scan raw JD text for secondary requirement signals and extract tags.

    Used as a fallback when no explicit nice-to-have section header is found.
    """
    found: list[str] = []

    for line in raw_text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\s\-\*\•\d\.]+", "", line).strip()
        if not line or len(line) < 5:
            continue

        # Check "X is a plus/bonus" first
        pm = _PLUS_BONUS_PATTERN.search(line)
        if pm:
            subject = _clean_tag(pm.group(1))
            if subject and not _is_filler(subject):
                found.append(subject)
            continue

        # Check payload patterns
        for pat in _SECONDARY_PAYLOAD_PATTERNS:
            m = pat.search(line)
            if m and m.group(1):
                payload = _clean_tag(m.group(1))
                # Strip trailing "is a plus/bonus" remnants
                payload = re.sub(r"\s*(?:is\s+)?(?:a\s+)?(?:plus|bonus)\s*$", "", payload, flags=re.I).strip()
                if payload and not _is_filler(payload):
                    found.append(payload)
                break  # only match first signal per line

    # Deduplicate against must-have
    result = []
    seen_lower: set[str] = set()
    for item in found:
        norm = _normalise_skill(item)
        lower = item.lower()
        if lower in must_have_lower or norm in must_have_lower:
            continue
        if any(_is_near_duplicate(lower, m) for m in must_have_lower):
            continue
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        result.append(item)

    return result[:8]


def parse_jd(raw_text: str) -> ParsedJD:
    """Parse a raw job description string into a structured ParsedJD object."""
    jd = ParsedJD(raw_text=raw_text)

    lines = raw_text.strip().split("\n")
    # Role title: first non-empty line or first line with 'engineer/scientist/developer'
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 120:
            jd.role_title = line
            break

    jd.seniority = _detect_seniority(raw_text)
    jd.min_experience_years = _extract_experience(raw_text)
    jd.domains = _detect_domains(raw_text)
    jd.work_mode = _detect_work_mode(raw_text)
    jd.intent_facets = _detect_intents(raw_text)

    # Extract must-have and nice-to-have sections
    # NOTE: compound headers like "Required Skills" must appear BEFORE bare
    # "required" so the regex engine matches the longer form first.
    must_haves = (
        _extract_skills_from_section(raw_text,
            r"(?:must[- ]?have(?:\s+skills)?|required\s+skills|skills\s+required|"
            r"requirements?|required|qualifications?|key\s+skills|technical\s+skills|"
            r"essential\s+skills|mandatory\s+skills|core\s+competencies)")
    )
    nice_to_haves = (
        _extract_skills_from_section(raw_text,
            r"(?:nice[- ]?to[- ]?have(?:\s+skills)?|preferred\s+skills|"
            r"preferred(?:\s+qualifications)?|good[- ]?to[- ]?have|"
            r"desired\s+skills|desired(?:\s+qualifications)?|"
            r"additional\s+(?:qualifications|skills)|bonus\s+skills|"
            r"helpful|advantageous|desirable)")
    )

    # Deduplicate within each list
    must_deduped = _dedup_preserve_order(must_haves)
    nice_deduped = _dedup_preserve_order(nice_to_haves)

    # Cross-dedup: remove nice items that already appear in must
    must_deduped, nice_deduped = _cross_dedup(must_deduped, nice_deduped)

    # Clean for display: split compounds, apply casing, enforce limits
    jd.must_have_skills = clean_tags_for_display(must_deduped, max_count=10)
    jd.nice_to_have_skills = clean_tags_for_display(nice_deduped, max_count=8)

    # ── Fallback: infer nice-to-have if section extraction found nothing ──
    if not jd.nice_to_have_skills:
        inferred = _infer_secondary_tags(raw_text, set(s.lower() for s in jd.must_have_skills))
        jd.nice_to_have_skills = clean_tags_for_display(inferred, max_count=6)

    # ── Last-resort fallback: promote tail of must-have ──
    if not jd.nice_to_have_skills and len(jd.must_have_skills) > 6:
        promoted = jd.must_have_skills[-3:]
        jd.must_have_skills = jd.must_have_skills[:-3]
        jd.nice_to_have_skills = promoted

    # Responsibilities
    resp = _extract_skills_from_section(raw_text, r"(?:responsibilities?|what you.?ll do|role)")
    jd.responsibilities = resp[:15]

    # Education
    edu = _extract_skills_from_section(raw_text, r"(?:education|degree)")
    jd.education_requirements = edu[:5]

    # Ranking priorities from intents
    jd.ranking_priorities = sorted(jd.intent_facets.keys(), key=lambda k: -jd.intent_facets[k])

    return jd
