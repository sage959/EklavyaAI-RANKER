"""
config.py — JD definition, keyword lists, scoring weights, behavioral thresholds.
All tunable constants live here. Never scatter magic numbers elsewhere.
"""
from __future__ import annotations
from typing import Any, Dict, List

# ── JOB DESCRIPTION ──────────────────────────────────────────────────────────

JD_ROLE    = "Senior AI Engineer"
JD_COMPANY = "Redrob"
JD_MIN_EXPERIENCE_YEARS = 4

# Hard requirements — each match carries full weight
JD_MUST_HAVE: List[str] = [
    "python",
    "retrieval",
    "vector search",
    "faiss",
    "elasticsearch",
    "dense retrieval",
    "sparse retrieval",
    "bm25",
    "production ml",
    "ml engineering",
    "ranking",
    "information retrieval",
    "embedding",
    "semantic search",
    "search",
]

# Nice-to-have — partial credit
JD_NICE_TO_HAVE: List[str] = [
    "cross-encoder",
    "reranking",
    "rag",
    "retrieval augmented generation",
    "sentencetransformers",
    "sentence transformers",
    "fastapi",
    "large language model",
    "llm",
    "evaluation",
    "ndcg",
    "mrr",
    "hnsw",
    "ann",
    "approximate nearest neighbor",
    "pinecone",
    "weaviate",
    "chromadb",
    "qdrant",
    "milvus",
    "two-tower",
    "bi-encoder",
    "colbert",
]

# Evidence of shipping to real users
PRODUCTION_SIGNALS: List[str] = [
    "production",
    "deployed",
    "deployment",
    "serving",
    "served",
    "latency",
    "qps",
    "queries per second",
    "queries per day",
    "throughput",
    "real-time",
    "at scale",
    "infrastructure",
    "million",
    "billion",
    "microservice",
    "api endpoint",
    "pipeline",
    "shipped",
    "launched",
    "live system",
    "live service",
    "users",
    "daily active",
]

# Retrieval/ranking-specific technology stack
RETRIEVAL_STACK_TERMS: List[str] = [
    "faiss",
    "hnsw",
    "annoy",
    "scann",
    "elasticsearch",
    "opensearch",
    "solr",
    "lucene",
    "bm25",
    "tf-idf",
    "tfidf",
    "dense retrieval",
    "sparse retrieval",
    "dual encoder",
    "bi-encoder",
    "cross-encoder",
    "reranking",
    "reranker",
    "two-tower",
    "colbert",
    "splade",
    "pinecone",
    "weaviate",
    "chromadb",
    "qdrant",
    "milvus",
    "vector database",
    "vector store",
    "ann",
    "approximate nearest",
]

# Evaluation metric / experimental rigour signals
EVALUATION_TERMS: List[str] = [
    "ndcg",
    "map@",
    "mrr",
    "precision@",
    "recall@",
    "hit rate",
    "offline eval",
    "online eval",
    "a/b test",
    "ab testing",
    "metrics",
    "benchmark",
    "evaluation framework",
]

# Python ecosystem depth signals
PYTHON_DEPTH_TERMS: List[str] = [
    "fastapi",
    "flask",
    "django",
    "pytorch",
    "tensorflow",
    "jax",
    "huggingface",
    "transformers",
    "sentence-transformers",
    "sentencetransformers",
    "sklearn",
    "scikit-learn",
    "asyncio",
    "pydantic",
    "celery",
    "ray",
    "dask",
]

# ML/AI/Search engineering — relevant current titles
RELEVANT_TITLE_TERMS: List[str] = [
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "research engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "retrieval engineer",
    "nlp engineer",
    "applied ml",
    "data scientist",
    "senior engineer",
]

# Broader set for career-history title matching (superset of RELEVANT_TITLE_TERMS)
# Used to detect engineering background even when current title is irrelevant.
CAREER_HISTORY_RELEVANT_TERMS: List[str] = [
    # Everything in RELEVANT_TITLE_TERMS
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "research engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "retrieval engineer",
    "nlp engineer",
    "applied ml",
    "data scientist",
    # Additional engineering roles that signal technical background
    "software engineer",
    "backend engineer",
    "data engineer",
    "platform engineer",
    "infrastructure engineer",
    "devops engineer",
    "site reliability",
    "full stack engineer",
    "systems engineer",
    "analytics engineer",
    "senior data engineer",
    "lead engineer",
    "staff engineer",
    "principal engineer",
    "tech lead",
    "engineering manager",
    "senior software",
    "senior backend",
    "senior data",
]

# Negative signal on current title (weak role-relevance to JD)
IRRELEVANT_TITLE_TERMS: List[str] = [
    "data analyst",
    "business analyst",
    "product manager",
    "business intelligence",
    "bi analyst",
    "operations analyst",
    "sales",
    "recruiter",
    "marketing",
    "account manager",
]

# Red-flag patterns in career descriptions (pure wrapper/no systems work)
NEGATIVE_SIGNALS: List[str] = [
    "prompt engineering only",
    "chatgpt wrapper",
    "langchain wrapper",
    "no deployment",
]

# Company tiers — implied production ML depth
TIER1_COMPANIES: List[str] = [
    "google", "meta", "microsoft", "amazon", "apple",
    "netflix", "openai", "anthropic", "deepmind", "nvidia",
]

TIER2_COMPANIES: List[str] = [
    "flipkart", "swiggy", "zomato", "razorpay", "phonepe",
    "paytm", "myntra", "meesho", "cred", "groww",
    "dream11", "ola", "uber", "stripe", "shopify",
    "atlassian", "adobe", "salesforce", "samsung", "qualcomm",
    "intuit", "oracle", "sap", "infosys", "tcs", "wipro",
    "thoughtworks", "freshworks", "zoho",
]

# ── SCORING WEIGHTS ───────────────────────────────────────────────────────────

WEIGHTS: Dict[str, float] = {
    "relevance":   0.35,
    "production":  0.25,
    "technical":   0.20,
    "behavioral":  0.20,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── BEHAVIORAL THRESHOLDS ────────────────────────────────────────────────────

BEHAVIORAL: Dict[str, float] = {
    # Response time: ≤ best → 1.0,  ≥ worst → 0.0
    "response_best_h":  2.0,
    "response_worst_h": 24.0,
    # Notice period: ≤ best → 1.0, ≥ worst → 0.4 (floor, not 0, still hirable)
    "notice_best_d":    30,
    "notice_worst_d":   90,
    # Last active recency
    "active_fresh_d":   7,
    "active_stale_d":   90,
    # GitHub activity: ≥ best → 1.0, ≤ worst → 0.0
    "github_best":      80,
    "github_worst":     20,
    # Recruiter response rate: ≥ best → 1.0, ≤ worst → 0.0
    "response_rate_best":  90,
    "response_rate_worst": 40,
    # Interview completion: ≥ best → 1.0, ≤ worst → 0.0
    "interview_best":  95,
    "interview_worst": 50,
}

# Weights within behavioral component (must sum to 1.0)
BEHAVIORAL_WEIGHTS: Dict[str, float] = {
    "open_to_work":    0.20,
    "recency":         0.20,
    "response_rate":   0.15,
    "response_time":   0.20,
    "notice_period":   0.10,
    "github":          0.10,
    "interview_rate":  0.05,
}
assert abs(sum(BEHAVIORAL_WEIGHTS.values()) - 1.0) < 1e-9

# ── NEGATIVE MULTIPLIER FACTORS ──────────────────────────────────────────────

NEG_NO_PRODUCTION   = 0.88   # No production signals anywhere in career history
NEG_WRAPPER_ONLY    = 0.90   # Only API-wrapper signals, no systems work
NEG_UNDER_EXPERIENCE = 0.85  # Experience < 2 years
NEG_FLOOR           = 0.65   # Minimum compounded multiplier

# ── CAREER-PIVOT-AWARE TITLE PENALTIES ───────────────────────────────────────
# Tiered penalties: current title irrelevant BUT career history may show relevance.
# Rule:
#   2+ prior relevant roles → NO penalty (career pivot)
#   1  prior relevant role  → mild penalty
#   0  prior relevant roles → full penalty
NEG_IRRELEVANT_ROLE_FULL = 0.85   # Zero relevant prior roles
NEG_IRRELEVANT_ROLE_MILD = 0.95   # Exactly 1 relevant prior role
# (No penalty when 2+ relevant prior roles)

# ── XGBOOST BLENDED SCORING ─────────────────────────────────────────────────
# XGBoost is an additive layer on top of the existing rule-based scoring.
# If disabled or model missing, the system falls back to rule-based scoring only.

XGB_ENABLED: bool = True             # Set to False to disable XGBoost entirely
XGB_MODEL_PATH: str = "models/xgb_ranker.json"  # Path to trained model artifact

# Blend formula: final = alpha * rule_score + beta * xgb_score
XGB_BLEND_ALPHA: float = 0.70        # Weight for current rule-based score
XGB_BLEND_BETA: float = 0.30         # Weight for XGBoost predicted score

# Training hyperparameters (used by xgb_train.py)
XGB_TRAIN_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}
