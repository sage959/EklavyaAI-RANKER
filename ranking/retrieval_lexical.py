"""
retrieval_lexical.py — BM25-based lexical retrieval over candidate chunks.
Includes title-aware boosts, exact skill matching, and phrase overlap scoring.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
import operator
import numpy as np
from pathlib import Path
from typing import Any

from ranking.candidate_model import CandidateProfile, EvidenceChunk


class BM25Index:
    """
    Okapi BM25 index over candidate text documents.
    Supports per-document scoring and candidate-level aggregation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens: list[int] = []
        self.doc_tf: list[Counter] = []       # term frequencies per doc
        self.df: Counter = Counter()          # document frequency per term
        self.doc_meta: list[dict] = []        # candidate_id, source_type, etc.
        self._candidate_doc_map: dict[str, list[int]] = {}  # cid -> list of doc indices
        self.inverted_index: dict[str, list[tuple[int, int]]] = {} # term -> list of (doc_idx, tf)

    def index_candidates(self, profiles: list[CandidateProfile]) -> None:
        """Build the BM25 index from candidate evidence chunks."""
        all_docs: list[tuple[list[str], dict]] = []

        for profile in profiles:
            for chunk in profile.evidence_chunks:
                tokens = _tokenize(chunk.normalized_text)
                meta = {
                    "candidate_id": chunk.candidate_id,
                    "source_type": chunk.source_type,
                    "section": chunk.section,
                    "importance_weight": chunk.importance_weight,
                    "title": chunk.title,
                    "company": chunk.company,
                }
                all_docs.append((tokens, meta))

        self.doc_count = len(all_docs)
        if self.doc_count == 0:
            return

        total_len = 0
        for tokens, meta in all_docs:
            idx = len(self.doc_tf)
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_lens.append(len(tokens))
            self.doc_meta.append(meta)
            total_len += len(tokens)

            for term in set(tokens):
                self.df[term] += 1
                
            for term, count in tf.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, count))

            cid = meta["candidate_id"]
            if cid not in self._candidate_doc_map:
                self._candidate_doc_map[cid] = []
            self._candidate_doc_map[cid].append(idx)

        self.avg_dl = total_len / self.doc_count if self.doc_count > 0 else 1.0
        self._build_numpy_arrays()

    def _build_numpy_arrays(self):
        """Convert postings and doc_lens to numpy arrays for vectorised scoring.

        Called once after index_candidates() or load_index().  The BM25
        formula is identical — only the loop execution moves from Python
        to C via numpy, giving ~20-50× speed-up on large corpora.
        """
        import logging
        _logger = logging.getLogger(__name__)

        if self.doc_count == 0:
            self._np_doc_lens = None
            self._np_postings = {}
            return

        t0 = time.time() if 'time' in dir() else None
        import time as _time
        t0 = _time.time()

        self._np_doc_lens = np.array(self.doc_lens, dtype=np.float64)

        self._np_postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in self.inverted_index.items():
            if not postings:
                continue
            doc_indices = np.array([p[0] for p in postings], dtype=np.int32)
            term_tfs = np.array([p[1] for p in postings], dtype=np.float64)
            self._np_postings[term] = (doc_indices, term_tfs)

        _logger.info(
            f"Numpy BM25 arrays built: {len(self._np_postings)} terms "
            f"in {_time.time() - t0:.1f}s"
        )

    def save_index(self, path: str) -> None:
        """Save the BM25 index state to a pickle file."""
        state = {
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_dl": self.avg_dl,
            "doc_lens": self.doc_lens,
            "doc_tf": self.doc_tf,
            "df": self.df,
            "doc_meta": self.doc_meta,
            "_candidate_doc_map": self._candidate_doc_map,
            "inverted_index": getattr(self, "inverted_index", {}),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        import logging
        logging.getLogger(__name__).info(f"BM25 index saved to {out}")

    def load_index(self, path: str) -> bool:
        """Load BM25 index state from a pickle file."""
        p = Path(path)
        if not p.exists():
            import logging
            logging.getLogger(__name__).warning(f"BM25 index file not found: {p}")
            return False
        try:
            with open(p, "rb") as f:
                state = pickle.load(f)
            self.k1 = state["k1"]
            self.b = state["b"]
            self.doc_count = state["doc_count"]
            self.avg_dl = state["avg_dl"]
            self.doc_lens = state["doc_lens"]
            self.doc_tf = state["doc_tf"]
            self.df = state["df"]
            self.doc_meta = state["doc_meta"]
            self._candidate_doc_map = state["_candidate_doc_map"]
            
            # Dynamically build inverted index if not present in older index files
            self.inverted_index = state.get("inverted_index", {})
            if not self.inverted_index:
                import logging
                logging.getLogger(__name__).info("Building BM25 inverted index dynamically...")
                for doc_idx, tf in enumerate(self.doc_tf):
                    for term, count in tf.items():
                        if term not in self.inverted_index:
                            self.inverted_index[term] = []
                        self.inverted_index[term].append((doc_idx, count))

            import logging
            logging.getLogger(__name__).info(
                f"BM25 index loaded: {self.doc_count} docs"
            )
            self._build_numpy_arrays()
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load BM25 index: {e}")
            return False

    def query(self, query_text: str, top_k: int = 200) -> list[dict[str, Any]]:
        """
        Score all documents against the query.
        Returns top_k results sorted by score descending.
        """
        query_tokens = _tokenize(query_text.lower())
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        doc_scores: dict[int, float] = {}

        k1_plus_1 = self.k1 + 1.0
        k1_1_minus_b = self.k1 * (1.0 - self.b)
        k1_b_over_avg_dl = (self.k1 * self.b) / self.avg_dl

        for term, q_tf in query_tf.items():
            if term not in self.inverted_index:
                continue

            term_df = self.df.get(term, 0)
            idf = math.log((self.doc_count - term_df + 0.5) / (term_df + 0.5) + 1.0)
            if idf < 0.2:
                continue

            base_score = idf * q_tf
            for doc_idx, term_tf in self.inverted_index[term]:
                dl = self.doc_lens[doc_idx]
                tf_norm = (term_tf * k1_plus_1) / (
                    term_tf + k1_1_minus_b + k1_b_over_avg_dl * dl
                )
                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + base_score * tf_norm

        scores = [(s, d) for d, s in doc_scores.items() if s > 0]
        scores.sort(key=lambda x: (-x[0], x[1]))
        results = []
        for score, doc_idx in scores[:top_k]:
            meta = self.doc_meta[doc_idx]
            results.append({
                "doc_idx": doc_idx,
                "bm25_score": score,
                **meta,
            })
        return results

    def query_candidates(
        self,
        query_text: str,
        must_have_skills: list[str] | None = None,
        relevant_titles: list[str] | None = None,
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Aggregate BM25 scores at the candidate level.
        Applies title-aware and skill boosts.
        Returns top_k candidates sorted by aggregated score.
        """
        query_tokens = _tokenize(query_text.lower())
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)

        k1_plus_1 = self.k1 + 1.0
        k1_1_minus_b = self.k1 * (1.0 - self.b)
        k1_b_over_avg_dl = (self.k1 * self.b) / self.avg_dl

        # ── Vectorised BM25 scoring using numpy ──────────────────────────
        use_numpy = hasattr(self, '_np_doc_lens') and self._np_doc_lens is not None

        if use_numpy:
            all_scores = np.zeros(self.doc_count, dtype=np.float64)

            for term, q_tf in query_tf.items():
                np_data = self._np_postings.get(term)
                if np_data is None:
                    continue

                term_df = self.df.get(term, 0)
                idf = math.log((self.doc_count - term_df + 0.5) / (term_df + 0.5) + 1.0)
                if idf < 0.2:
                    continue

                base_score = idf * q_tf
                doc_indices, term_tfs = np_data
                dls = self._np_doc_lens[doc_indices]
                tf_norms = (term_tfs * k1_plus_1) / (
                    term_tfs + k1_1_minus_b + k1_b_over_avg_dl * dls
                )
                np.add.at(all_scores, doc_indices, base_score * tf_norms)

            # Convert numpy scores to dict for candidate aggregation
            nonzero_indices = np.nonzero(all_scores)[0]
            doc_scores = {int(idx): float(all_scores[idx]) for idx in nonzero_indices}
        else:
            # Fallback: pure-Python scoring (used if numpy arrays not built)
            doc_scores: dict[int, float] = {}

            for term, q_tf in query_tf.items():
                if term not in self.inverted_index:
                    continue

                term_df = self.df.get(term, 0)
                idf = math.log((self.doc_count - term_df + 0.5) / (term_df + 0.5) + 1.0)
                if idf < 0.2:
                    continue

                base_score = idf * q_tf

                postings = self.inverted_index[term]

                for doc_idx, term_tf in postings:
                    dl = self.doc_lens[doc_idx]
                    tf_norm = (term_tf * k1_plus_1) / (
                        term_tf + k1_1_minus_b + k1_b_over_avg_dl * dl
                    )
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + base_score * tf_norm

        candidate_scores: dict[str, float] = {}
        candidate_details: dict[str, dict] = {}

        for doc_idx, score in doc_scores.items():
            if score <= 0:
                continue

            meta = self.doc_meta[doc_idx]
            cid = meta["candidate_id"]
            weight = meta.get("importance_weight", 1.0)

            candidate_scores[cid] = candidate_scores.get(cid, 0.0) + score * weight
            if cid not in candidate_details:
                candidate_details[cid] = {"candidate_id": cid, "top_chunks": []}
            candidate_details[cid]["top_chunks"].append({
                "section": meta["section"],
                "bm25_score": round(score, 4),
            })

        # Sort candidates by raw BM25 score
        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        
        # Optimization: Only apply expensive boosts to the top candidates
        top_cids_for_boosting = set([cid for cid, score in ranked[:max(top_k * 5, 2000)]])

        # Apply boosts
        for cid in top_cids_for_boosting:
            boost = 1.0

            # Title boost
            if relevant_titles:
                title_chunks = [
                    self.doc_meta[idx] for idx in self._candidate_doc_map.get(cid, [])
                    if self.doc_meta[idx].get("title")
                ]
                for tc in title_chunks:
                    title_lower = tc["title"].lower()
                    if any(rt.lower() in title_lower for rt in relevant_titles):
                        boost += 0.15
                        break

            # Skill boost
            if must_have_skills:
                skill_docs = [
                    idx for idx in self._candidate_doc_map.get(cid, [])
                    if self.doc_meta[idx]["source_type"] == "skill"
                ]
                for idx in skill_docs:
                    doc_text = " ".join(self.doc_tf[idx].keys())
                    hits = sum(1 for s in must_have_skills if s.lower() in doc_text)
                    if hits > 0:
                        boost += min(0.20, hits * 0.04)

            candidate_scores[cid] *= boost

        # Sort again after boosts
        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        results = []
        for cid, score in ranked[:top_k]:
            detail = candidate_details[cid]
            detail["lexical_score"] = round(score, 4)
            detail["top_chunks"] = sorted(
                detail["top_chunks"], key=lambda c: -c["bm25_score"]
            )[:5]
            results.append(detail)

        return results

    def _score_document(self, doc_idx: int, query_tokens: list[str]) -> float:
        """BM25 score for a single document."""
        tf = self.doc_tf[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue
            term_tf = tf[term]
            term_df = self.df.get(term, 0)

            idf = math.log((self.doc_count - term_df + 0.5) / (term_df + 0.5) + 1.0)
            tf_norm = (term_tf * (self.k1 + 1)) / (
                term_tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            )
            score += idf * tf_norm

        return score


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'[a-z0-9_\-\.]+', text.lower())
