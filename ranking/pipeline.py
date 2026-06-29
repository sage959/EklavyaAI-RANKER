"""
pipeline.py — End-to-end ranking pipeline.
Orchestrates: load → parse JD → build candidate models → chunk → index →
hybrid retrieve → rerank → score features → proof-of-work →
contradictions → explanations → export.

Integrates the existing feature_engineering + scorer as a feature component.
"""
from __future__ import annotations

import csv
import json
import heapq
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ranking.candidate_model import CandidateProfile, build_candidate_profile
from ranking.chunking import chunk_candidate
from ranking.retrieval_lexical import BM25Index
from ranking.retrieval_semantic import SemanticIndex
from ranking.hybrid import fuse_retrieval_scores
from ranking.rerank import rerank_candidates
from ranking.proof_of_work import score_proof_of_work
from ranking.contradictions import check_contradictions
from ranking.credibility import check_resume_credibility
from ranking.explanations import generate_explanation
from ranking.jd_parser import ParsedJD, parse_jd
from ranking import feature_engineering
from ranking import scorer as legacy_scorer
from ranking import config
from ranking import xgb_model

logger = logging.getLogger(__name__)

# Default JD text for the Redrob hackathon (Senior AI Engineer role)
DEFAULT_JD_TEXT = """Senior AI Engineer

Requirements:
- 4+ years of experience in ML engineering or AI/ML research
- Strong Python skills with production experience
- Experience with retrieval systems, search, ranking, or recommendation
- Familiarity with vector search, FAISS, Elasticsearch, or similar
- Experience with production ML deployment and serving
- Knowledge of evaluation metrics (nDCG, MRR, precision/recall)
- Experience building end-to-end ML pipelines

Nice to have:
- Experience with cross-encoders, reranking, RAG
- Sentence transformers or embedding models
- FastAPI or similar Python web frameworks
- Experience with large language models
- Contributions to open-source ML projects
- Experience with A/B testing and online evaluation

Responsibilities:
- Build and maintain retrieval and ranking systems
- Deploy ML models to production at scale
- Design evaluation frameworks for search quality
- Collaborate with product teams to ship AI features
"""


class RankingPipeline:
    """Full ranking pipeline orchestrator."""

    def __init__(self, jd_text: str | None = None, use_semantic: bool = True,
                 use_reranker: bool = True):
        self.jd_text = jd_text or DEFAULT_JD_TEXT
        self.parsed_jd: ParsedJD = parse_jd(self.jd_text)
        self.use_semantic = use_semantic
        self.use_reranker = use_reranker

        self.bm25_index = BM25Index()
        self.semantic_index = SemanticIndex()
        self.profiles: dict[str, CandidateProfile] = {}  # cid -> profile
        self.raw_candidates: dict[str, dict] = {}  # cid -> raw JSON

        self._indexed = False
        self._precomputed = False
        self._precomputed_texts: dict[str, str] = {}

        # ── XGBoost model loading (graceful fallback if missing) ──
        self._xgb_available = False
        if config.XGB_ENABLED:
            self._xgb_available = xgb_model.load_model(config.XGB_MODEL_PATH)
            if self._xgb_available:
                logger.info("XGBoost blended scoring enabled.")
            else:
                logger.info("XGBoost model not found — using rule-based scoring only.")

    def load_candidates_from_jsonl(self, path: str, limit: int | None = None) -> int:
        """Stream and parse candidates from a JSONL file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Candidates file not found: {p}")

        count = 0
        logger.info(f"Loading candidates from {p}...")
        t0 = time.time()

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cid = raw.get("candidate_id", "")
                if not cid:
                    continue

                profile = build_candidate_profile(raw)
                self.profiles[cid] = profile
                self.raw_candidates[cid] = raw
                count += 1

                if count % 10000 == 0:
                    logger.info(f"  Loaded {count} candidates...")

                if limit and count >= limit:
                    break

        elapsed = time.time() - t0
        logger.info(f"Loaded {count} candidates in {elapsed:.1f}s")
        return count

    def load_candidates_from_list(self, candidates: list[dict]) -> int:
        """Load candidates from an in-memory list of raw dicts."""
        count = 0
        for raw in candidates:
            cid = raw.get("candidate_id", "")
            if not cid:
                continue
            profile = build_candidate_profile(raw)
            self.profiles[cid] = profile
            self.raw_candidates[cid] = raw
            count += 1
        logger.info(f"Loaded {count} candidates from list.")
        return count

    def build_index(self) -> dict[str, Any]:
        """Build retrieval indices (BM25 + semantic)."""
        profiles_list = list(self.profiles.values())
        if not profiles_list:
            return {"status": "error", "message": "No candidates loaded."}

        t0 = time.time()

        # BM25
        logger.info("Building BM25 index...")
        self.bm25_index.index_candidates(profiles_list)
        bm25_time = time.time() - t0

        # Semantic
        semantic_built = False
        sem_time = 0.0
        if self.use_semantic:
            t1 = time.time()
            logger.info("Building semantic index...")
            semantic_built = self.semantic_index.build_index(profiles_list)
            sem_time = time.time() - t1

        self._indexed = True
        total_time = time.time() - t0

        return {
            "status": "ok",
            "candidate_count": len(profiles_list),
            "bm25_docs": self.bm25_index.doc_count,
            "semantic_available": semantic_built,
            "bm25_time_s": round(bm25_time, 2),
            "semantic_time_s": round(sem_time, 2),
            "total_time_s": round(total_time, 2),
        }

    def load_precomputed_index(self, data_dir: str) -> dict[str, Any]:
        """
        Load pre-computed BM25 + FAISS indices from disk.

        Also loads the candidate data JSONL path for on-demand candidate loading
        during the scoring phase (avoids holding all 100k in memory).
        """
        d = Path(data_dir)
        t0 = time.time()

        # Load BM25 index
        bm25_path = d / "bm25_index.pkl"
        bm25_loaded = self.bm25_index.load_index(str(bm25_path))

        # Load semantic index
        semantic_dir = d / "semantic"
        semantic_loaded = False
        if self.use_semantic:
            semantic_loaded = self.semantic_index.load_index(str(semantic_dir))

        # Store path to candidate data for on-demand loading
        self._candidate_data_path = d / "candidate_data.jsonl"
        self._candidate_texts_path = d / "candidate_texts.json"

        # Load candidate texts map for reranker
        if self._candidate_texts_path.exists():
            with open(self._candidate_texts_path, "r", encoding="utf-8") as f:
                self._precomputed_texts = json.load(f)
        else:
            self._precomputed_texts = {}

        self._indexed = True
        self._precomputed = True
        total_time = time.time() - t0

        logger.info(f"Pre-computed indices loaded in {total_time:.1f}s")

        return {
            "status": "ok",
            "bm25_loaded": bm25_loaded,
            "semantic_loaded": semantic_loaded,
            "bm25_docs": self.bm25_index.doc_count,
            "total_time_s": round(total_time, 2),
        }

    def _load_candidates_for_scoring(self, cids: list[str]) -> int:
        """
        Load only the specified candidates from the pre-computed data JSONL.

        This avoids holding all 100k candidates in memory -- only the
        shortlisted candidates (typically ~200) are loaded and parsed.
        """
        if not hasattr(self, '_candidate_data_path') or \
           not self._candidate_data_path.exists():
            logger.warning("No pre-computed candidate data file found.")
            return 0

        needed = set(cids) - set(self.raw_candidates.keys())
        if not needed:
            return 0  # All already loaded

        count = 0
        with open(self._candidate_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cid = raw.get("candidate_id", "")
                if cid in needed:
                    profile = build_candidate_profile(raw)
                    self.profiles[cid] = profile
                    self.raw_candidates[cid] = raw
                    count += 1
                    needed.discard(cid)
                    if not needed:
                        break  # Found all we need

        logger.info(f"Loaded {count} candidates on-demand for scoring.")
        return count

    def rank(self, top_k: int = 100) -> list[dict[str, Any]]:
        """
        Run the full ranking pipeline.
        Returns a list of top_k ranked candidate result dicts.
        """
        if not self._indexed:
            self.build_index()

        # ── Ensure deterministic reproducibility ──────────────────────────
        import numpy as _np
        _np.random.seed(42)
        try:
            import torch as _torch
            _torch.manual_seed(42)
            _torch.set_num_threads(1)
            if hasattr(_torch, 'use_deterministic_algorithms'):
                _torch.use_deterministic_algorithms(True, warn_only=True)
        except ImportError:
            pass

        t0 = time.time()

        # ── 1. Hybrid retrieval ───────────────────────────────────────────
        jd_query = self.jd_text

        # In precomputed mode, profiles dict may be empty; use index size
        total_candidates = len(self.profiles) if self.profiles else self.bm25_index.doc_count

        t0_hybrid = time.time()
        logger.info("Starting BM25 query...")
        lexical_results = self.bm25_index.query_candidates(
            query_text=jd_query,
            must_have_skills=self.parsed_jd.must_have_skills,
            relevant_titles=config.RELEVANT_TITLE_TERMS,
            top_k=min(top_k * 3, max(total_candidates, 1)),
        )
        logger.info(f"BM25 query finished in {time.time() - t0_hybrid:.2f}s")

        if self.semantic_index.available:
            logger.info("Starting semantic FAISS query...")
            t0_sem = time.time()
            semantic_results = self.semantic_index.query_candidates(
                query_text=jd_query,
                top_k=min(top_k * 3, max(total_candidates, 1)),
            )
            logger.info(f"Semantic query finished in {time.time() - t0_sem:.2f}s")
            
            logger.info("Fusing results...")
            t0_fuse = time.time()
            fused = fuse_retrieval_scores(lexical_results, semantic_results)
            logger.info(f"Fusion finished in {time.time() - t0_fuse:.2f}s")
        else:
            # Lexical-only fallback
            fused = []
            for r in lexical_results:
                r["hybrid_score"] = r.get("lexical_score", 0.0)
                fused.append(r)

        logger.info(f"Hybrid retrieval returned {len(fused)} candidates in {time.time() - t0_hybrid:.2f}s total.")

        # ---- 2. Cross-encoder reranking ----
        # Use precomputed texts if available, else build from profiles
        if self._precomputed and self._precomputed_texts:
            candidate_texts = self._precomputed_texts
        else:
            candidate_texts = {cid: p.full_text[:512] for cid, p in self.profiles.items()}

        if self.use_reranker and len(fused) > 0:
            fused = rerank_candidates(
                query=jd_query[:512],
                candidates=fused,
                candidate_texts=candidate_texts,
                top_n=min(top_k * 2, len(fused)),
            )

        # ---- 3. Score features for top candidates ----
        # Limit to top candidates for detailed scoring
        shortlist_cids = [r["candidate_id"] for r in fused[:top_k * 2]]

        # In precomputed mode, load only the shortlisted candidates on-demand
        if self._precomputed:
            self._load_candidates_for_scoring(shortlist_cids)

        results: list[dict[str, Any]] = []
        for cid in shortlist_cids:
            if cid not in self.raw_candidates:
                continue

            raw = self.raw_candidates[cid]
            profile = self.profiles[cid]

            # Legacy feature extraction (reused from existing system)
            features = feature_engineering.extract_all_features(raw, self.parsed_jd)
            features = legacy_scorer.compute_final_score(features)

            # Proof of work
            pow_result = score_proof_of_work(profile)

            # Contradiction checks
            contradiction_result = check_contradictions(profile, features, pow_result)

            # Credibility checks
            credibility_result = check_resume_credibility(profile, features)

            # Retrieval info for this candidate
            retrieval_info = next(
                (r for r in fused if r["candidate_id"] == cid), {}
            )

            # Generate explanation
            explanation = generate_explanation(
                features, pow_result, contradiction_result, retrieval_info,
                credibility_result=credibility_result,
            )

            # ── Compute final combined score ──────────────────────────────
            legacy_score = features.get("final_score", 0.0)
            retrieval_score = retrieval_info.get("final_retrieval_score",
                             retrieval_info.get("hybrid_score", 0.0))

            # Normalise retrieval score to 0-1
            retrieval_norm = min(1.0, max(0.0, retrieval_score))

            # Proof-of-work contribution
            pow_score = pow_result.get("pow_score", 0.0)

            # Contradiction penalty
            contra_penalty = contradiction_result.get("penalty", 1.0)

            # Credibility penalty
            credibility_mult = credibility_result.get("credibility_multiplier", 1.0)

            # Combined rule-based score: blend legacy features + retrieval + PoW
            rule_combined = (
                0.45 * legacy_score
                + 0.30 * retrieval_norm
                + 0.15 * pow_score
                + 0.10 * features.get("behavioral", {}).get("score", 0.5)
            ) * contra_penalty * credibility_mult

            rule_combined = round(max(0.0, min(1.0, rule_combined)), 4)

            # ── XGBoost blended scoring ──
            xgb_score_val = None
            final_combined = rule_combined

            if self._xgb_available:
                xgb_score_val = xgb_model.predict_score(features)
                if xgb_score_val is not None:
                    final_combined = round(max(0.0, min(1.0,
                        config.XGB_BLEND_ALPHA * rule_combined
                        + config.XGB_BLEND_BETA * xgb_score_val
                    )), 4)

            results.append({
                "candidate_id": cid,
                "name": profile.name,
                "current_title": profile.current_title,
                "final_score": final_combined,
                "rule_score": rule_combined,
                "xgb_score": round(xgb_score_val, 4) if xgb_score_val is not None else None,
                "legacy_score": legacy_score,
                "retrieval_score": round(retrieval_norm, 4),
                "pow_score": pow_result["pow_score"],
                "behavioral_score": round(features.get("behavioral", {}).get("score", 0.5), 4),
                "contra_penalty": contra_penalty,
                "credibility_penalty": credibility_result.get("credibility_penalty", 0.0),
                "reasoning_text": explanation["reasoning_text"],
                "explanation": explanation,
                "features": features,
                "pow_detail": pow_result,
                "contradiction_detail": contradiction_result,
                "credibility_detail": credibility_result,
            })

        # ── 4. Sort and assign ranks ──────────────────────────────────────
        # Sort by final_score descending, tie-break by candidate_id ascending
        results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

        # Take top_k
        results = results[:top_k]

        # Assign ranks
        for i, r in enumerate(results, 1):
            r["rank"] = i

        elapsed = time.time() - t0
        logger.info(f"Ranking complete: {len(results)} candidates in {elapsed:.1f}s")

        return results

    def export_submission_csv(self, results: list[dict], output_path: str) -> str:
        """
        Export results to the exact submission CSV format.
        Header: candidate_id,rank,score,reasoning
        Exactly 100 rows.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])

            for r in results[:100]:
                writer.writerow([
                    r["candidate_id"],
                    r["rank"],
                    r["final_score"],
                    r["reasoning_text"],
                ])

        logger.info(f"Submission CSV exported to {out}")
        return str(out)

    def export_detailed_json(self, results: list[dict], output_path: str) -> str:
        """Export rich JSON output with full score breakdowns."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Serialise — strip non-serialisable data
        export_data = []
        for r in results:
            cid = r["candidate_id"]
            profile = self.profiles.get(cid)
            
            entry = {
                "candidate_id": cid,
                "rank": r["rank"],
                "name": r.get("name", ""),
                "current_title": r.get("current_title", ""),
                "final_score": r["final_score"],
                "rule_score": r.get("rule_score", r["final_score"]),
                "xgb_score": r.get("xgb_score"),
                "legacy_score": r.get("legacy_score", 0),
                "retrieval_score": r.get("retrieval_score", 0),
                "pow_score": r.get("pow_score", 0),
                "behavioral_score": r.get("behavioral_score", 0.5),
                "contra_penalty": r.get("contra_penalty", 1.0),
                "credibility_penalty": r.get("credibility_penalty", 0.0),
                "reasoning": r["reasoning_text"],
                "explanation": r.get("explanation", {}),
                "pow_detail": r.get("pow_detail", {}),
                "contradiction_detail": r.get("contradiction_detail", {}),
                "credibility_detail": r.get("credibility_detail", {}),
            }

            if profile:
                entry.update({
                    "skills": [s.name for s in profile.skills],
                    "location": profile.location,
                    "country": profile.country,
                    "experience": f"{profile.years_of_experience:.1f} yrs",
                    "current_company": profile.current_company,
                    "education": ", ".join([f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}" for edu in profile.education if edu]),
                    "careerHistory": [
                        {
                            "role": e.title,
                            "company": e.company,
                            "period": f"{e.start_date} - {e.end_date or 'Present'}",
                            "note": e.description,
                        }
                        for e in profile.career_events
                    ],
                    "redrob_signals": profile.redrob_signals,
                })
            
            export_data.append(entry)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Detailed JSON exported to {out}")
        return str(out)


def run_full_pipeline(
    candidates_path: str,
    output_csv: str = "output/TIER3.csv",
    output_json: str = "output/detailed_results.json",
    jd_text: str | None = None,
    limit: int | None = None,
    use_semantic: bool = True,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to run the full pipeline end-to-end.
    Returns a summary dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pipeline = RankingPipeline(
        jd_text=jd_text,
        use_semantic=use_semantic,
        use_reranker=use_reranker,
    )

    # Load
    count = pipeline.load_candidates_from_jsonl(candidates_path, limit=limit)

    # Index
    index_info = pipeline.build_index()

    # Rank
    results = pipeline.rank(top_k=100)

    # Export
    csv_path = pipeline.export_submission_csv(results, output_csv)
    json_path = pipeline.export_detailed_json(results, output_json)

    return {
        "candidates_loaded": count,
        "index_info": index_info,
        "ranked_count": len(results),
        "csv_path": csv_path,
        "json_path": json_path,
        "top_3": [
            {
                "rank": r["rank"],
                "candidate_id": r["candidate_id"],
                "score": r["final_score"],
                "reasoning": r["reasoning_text"][:120],
            }
            for r in results[:3]
        ],
    }
