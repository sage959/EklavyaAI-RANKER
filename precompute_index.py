"""
precompute_index.py -- Offline index builder for the ranking pipeline.

Pre-computes FAISS + BM25 indices and saves lightweight candidate data to disk,
so the ranking phase can load them instantly without re-embedding 100k candidates.

Usage:
    python precompute_index.py --input dataset/candidates.jsonl

Outputs (in data/ directory):
    data/semantic/faiss_index.bin    FAISS index
    data/semantic/embeddings.npy     Embedding matrix
    data/semantic/chunk_meta.json    Chunk metadata
    data/bm25_index.pkl              BM25 index (pickled)
    data/candidate_data.jsonl        Raw candidate JSON lines (for scoring)
    data/candidate_texts.json        cid -> full_text map (for reranker)
"""
import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.candidate_model import build_candidate_profile
from ranking.retrieval_lexical import BM25Index
from ranking.retrieval_semantic import SemanticIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Output directory
DATA_DIR = PROJECT_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description="Pre-compute indices for ranking pipeline")
    parser.add_argument(
        "--input", type=str, default="dataset/candidates.jsonl",
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="Batch size for processing candidates",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Prepare output directories
    semantic_dir = DATA_DIR / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ---- Phase 1: Stream candidates, build profiles, collect data --------
    logger.info("=" * 60)
    logger.info("Phase 1: Loading and parsing candidates...")
    logger.info("=" * 60)

    bm25_index = BM25Index()
    semantic_index = SemanticIndex()

    all_profiles = []
    candidate_texts = {}  # cid -> full_text (for reranker)

    # Stream and parse in batches
    count = 0
    batch_profiles = []

    candidate_data_path = DATA_DIR / "candidate_data.jsonl"
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(candidate_data_path, "w", encoding="utf-8") as fout:
        for line in fin:
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

            # Save raw candidate data for later scoring
            fout.write(line + "\n")

            # Build profile
            profile = build_candidate_profile(raw)
            batch_profiles.append(profile)
            candidate_texts[cid] = profile.full_text[:512]
            count += 1

            if count % 10000 == 0:
                logger.info(f"  Parsed {count} candidates...")

    all_profiles = batch_profiles
    logger.info(f"Parsed {count} candidates in {time.time() - t_start:.1f}s")

    # ---- Phase 2: Build BM25 index --------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Building BM25 index...")
    logger.info("=" * 60)

    t_bm25 = time.time()
    bm25_index.index_candidates(all_profiles)
    bm25_path = str(DATA_DIR / "bm25_index.pkl")
    bm25_index.save_index(bm25_path)
    logger.info(f"BM25 index: {bm25_index.doc_count} docs in {time.time() - t_bm25:.1f}s")

    # ---- Phase 3: Build semantic (FAISS) index ---------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Building semantic (FAISS) index...")
    logger.info("=" * 60)

    t_sem = time.time()
    semantic_built = semantic_index.build_index(all_profiles)
    if semantic_built:
        semantic_index.save_index(str(semantic_dir))
        logger.info(f"Semantic index built and saved in {time.time() - t_sem:.1f}s")
    else:
        logger.warning("Semantic index could not be built (missing dependencies?)")

    # ---- Phase 4: Save candidate texts for reranker ----------------------
    logger.info("Saving candidate texts...")
    texts_path = DATA_DIR / "candidate_texts.json"
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(candidate_texts, f)

    # ---- Summary ---------------------------------------------------------
    total_time = time.time() - t_start

    # Check disk usage
    data_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    data_size_mb = data_size / (1024 * 1024)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Candidates processed: {count}")
    logger.info(f"BM25 docs indexed:    {bm25_index.doc_count}")
    logger.info(f"Semantic index:       {'built' if semantic_built else 'NOT built'}")
    logger.info(f"Data directory:       {DATA_DIR}")
    logger.info(f"Disk usage:           {data_size_mb:.1f} MB")
    logger.info(f"Total time:           {total_time:.1f}s")
    logger.info("")
    logger.info("Files created:")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.relative_to(DATA_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
