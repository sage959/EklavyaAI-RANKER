"""
retrieval_semantic.py — Semantic retrieval using sentence-transformers + FAISS.
Provides chunk-level embedding, indexing, and candidate-aggregated retrieval.
Falls back gracefully to lexical-only if models are unavailable.
"""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Optional

from ranking.candidate_model import CandidateProfile, EvidenceChunk

logger = logging.getLogger(__name__)

# ── Model paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_DIR = _PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
_HUB_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded globals
_model = None


def _load_model():
    """Load the sentence-transformers model lazily from local dir or hub."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        if _LOCAL_MODEL_DIR.exists():
            model_path = str(_LOCAL_MODEL_DIR)
            logger.info(f"Loading embedding model from local: {model_path}")
            _model = SentenceTransformer(model_path, device="cpu",
                                         local_files_only=True)
        else:
            logger.info(f"Local model not found, loading from hub: {_HUB_MODEL_NAME}")
            _model = SentenceTransformer(_HUB_MODEL_NAME, device="cpu")
        logger.info("Embedding model loaded successfully.")
        return _model
    except ImportError:
        logger.warning("sentence-transformers not installed. Semantic retrieval disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}. Semantic retrieval disabled.")
        return None


def _load_faiss():
    """Import faiss lazily."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.warning("faiss not installed. Semantic retrieval disabled.")
        return None


class SemanticIndex:
    """FAISS-based semantic retrieval index over candidate evidence chunks."""

    def __init__(self):
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_meta: list[dict] = []
        self._candidate_chunk_map: dict[str, list[int]] = {}
        self._available = False
        self._dimension = 0

    @property
    def available(self) -> bool:
        return self._available

    def build_index(self, profiles: list[CandidateProfile]) -> bool:
        """
        Build FAISS index from candidate evidence chunks.
        Returns True if successful, False if dependencies unavailable.
        """
        model = _load_model()
        faiss = _load_faiss()

        if model is None or faiss is None:
            logger.warning("Semantic index not built: missing dependencies.")
            return False

        # Collect all chunk texts and metadata
        texts: list[str] = []
        metas: list[dict] = []

        for profile in profiles:
            for chunk in profile.evidence_chunks:
                if not chunk.text.strip():
                    continue
                texts.append(chunk.text)
                metas.append({
                    "candidate_id": chunk.candidate_id,
                    "source_type": chunk.source_type,
                    "section": chunk.section,
                    "importance_weight": chunk.importance_weight,
                    "idx": len(texts) - 1,
                })

        if not texts:
            return False

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = model.encode(texts, batch_size=128, show_progress_bar=False,
                                  normalize_embeddings=True)
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self._dimension = self.embeddings.shape[1]

        # Build FAISS index (Inner Product for cosine sim on normalized vectors)
        self.index = faiss.IndexFlatIP(self._dimension)
        self.index.add(self.embeddings)

        self.chunk_meta = metas

        # Build candidate -> chunk index mapping
        for i, meta in enumerate(metas):
            cid = meta["candidate_id"]
            if cid not in self._candidate_chunk_map:
                self._candidate_chunk_map[cid] = []
            self._candidate_chunk_map[cid].append(i)

        self._available = True
        logger.info(f"Semantic index built: {len(texts)} chunks, dim={self._dimension}")
        return True

    def save_index(self, dir_path: str) -> None:
        """Save FAISS index, embeddings, and chunk metadata to disk."""
        faiss = _load_faiss()
        if faiss is None or self.index is None:
            raise RuntimeError("Cannot save: index not built or faiss unavailable.")

        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(out / "faiss_index.bin"))

        # Save embeddings
        np.save(str(out / "embeddings.npy"), self.embeddings)

        # Save chunk metadata
        with open(out / "chunk_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_meta, f)

        # Save candidate chunk map
        with open(out / "candidate_chunk_map.json", "w", encoding="utf-8") as f:
            json.dump(self._candidate_chunk_map, f)

        logger.info(f"Semantic index saved to {out}")

    def load_index(self, dir_path: str) -> bool:
        """Load pre-computed FAISS index from disk."""
        faiss = _load_faiss()
        if faiss is None:
            return False

        d = Path(dir_path)
        index_path = d / "faiss_index.bin"
        emb_path = d / "embeddings.npy"
        meta_path = d / "chunk_meta.json"
        map_path = d / "candidate_chunk_map.json"

        if not index_path.exists():
            logger.warning(f"Pre-computed index not found at {index_path}")
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            self.embeddings = np.load(str(emb_path))
            self._dimension = self.embeddings.shape[1]

            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunk_meta = json.load(f)

            if map_path.exists():
                with open(map_path, "r", encoding="utf-8") as f:
                    self._candidate_chunk_map = json.load(f)

            self._available = True
            logger.info(
                f"Semantic index loaded from disk: {len(self.chunk_meta)} chunks, "
                f"dim={self._dimension}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load pre-computed index: {e}")
            return False

    def query(self, query_text: str, top_k: int = 200) -> list[dict[str, Any]]:
        """Query the FAISS index and return top_k chunk results."""
        if not self._available:
            return []

        model = _load_model()
        if model is None:
            return []

        # Pin PyTorch determinism for reproducible embeddings
        try:
            import torch
            torch.manual_seed(42)
            torch.set_num_threads(1)
        except ImportError:
            pass

        q_emb = model.encode([query_text], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunk_meta)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.chunk_meta[idx]
            results.append({
                "chunk_idx": int(idx),
                "semantic_score": float(score),
                **meta,
            })
        return results

    def query_candidates(self, query_text: str, top_k: int = 100) -> list[dict[str, Any]]:
        """
        Query and aggregate at the candidate level.
        Uses importance-weighted sum of chunk similarities.
        """
        chunk_results = self.query(query_text, top_k=top_k * 5)
        if not chunk_results:
            return []

        candidate_scores: dict[str, float] = {}
        candidate_details: dict[str, dict] = {}

        for r in chunk_results:
            cid = r["candidate_id"]
            weight = r.get("importance_weight", 1.0)
            score = r["semantic_score"] * weight

            candidate_scores[cid] = candidate_scores.get(cid, 0.0) + score
            if cid not in candidate_details:
                candidate_details[cid] = {"candidate_id": cid, "top_chunks": []}
            candidate_details[cid]["top_chunks"].append({
                "section": r["section"],
                "semantic_score": round(r["semantic_score"], 4),
            })

        ranked = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
        results = []
        for cid, score in ranked[:top_k]:
            detail = candidate_details[cid]
            detail["semantic_score"] = round(score, 4)
            detail["top_chunks"] = sorted(
                detail["top_chunks"], key=lambda c: -c["semantic_score"]
            )[:5]
            results.append(detail)

        return results
