"""
run_pipeline.py -- CLI entry point for the full ranking pipeline.
Runs end-to-end: load -> index -> rank -> export -> validate.

Supports two modes:
  Normal:       python run_pipeline.py --limit 100
  Precomputed:  python run_pipeline.py --precomputed
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.pipeline import RankingPipeline, run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="Eklavya AI -- Full Ranking Pipeline")
    parser.add_argument(
        "--input", type=str, default="dataset/candidates.jsonl",
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--output-csv", type=str, default="output/TIER3.csv",
        help="Path to submission CSV output",
    )
    parser.add_argument(
        "--output-json", type=str, default="output/detailed_results.json",
        help="Path to detailed JSON output",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of candidates to process (for testing)",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Disable semantic retrieval (faster, lexical-only)",
    )
    parser.add_argument(
        "--no-reranker", action="store_true",
        help="Disable cross-encoder reranking (faster)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run the validator on the output CSV after generation",
    )
    parser.add_argument(
        "--precomputed", action="store_true",
        help="Use pre-computed indices from data/ directory (fast mode)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing pre-computed indices (default: data/)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = PROJECT_ROOT / args.input
    output_csv = PROJECT_ROOT / args.output_csv
    output_json = PROJECT_ROOT / args.output_json
    data_dir = PROJECT_ROOT / args.data_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Eklavya AI -- Full Ranking Pipeline")
    logger.info("=" * 60)

    t_start = time.time()

    if args.precomputed:
        # ---- FAST MODE: Load pre-computed indices ----
        logger.info("Mode: PRECOMPUTED (loading indices from disk)")
        logger.info(f"Data directory: {data_dir}")

        pipeline = RankingPipeline(
            use_semantic=not args.no_semantic,
            use_reranker=not args.no_reranker,
        )

        # Load pre-built indices (instant)
        index_info = pipeline.load_precomputed_index(str(data_dir))
        logger.info(f"Index load: {index_info}")

        # Rank (retrieval + on-demand candidate loading + scoring)
        results = pipeline.rank(top_k=100)

        # Export
        csv_path = pipeline.export_submission_csv(results, str(output_csv))
        json_path = pipeline.export_detailed_json(results, str(output_json))

        total_time = time.time() - t_start

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULTS (PRECOMPUTED MODE)")
        logger.info("=" * 60)
        logger.info(f"BM25 docs in index:   {index_info.get('bm25_docs', 'N/A')}")
        logger.info(f"Candidates ranked:    {len(results)}")
        logger.info(f"CSV output:           {csv_path}")
        logger.info(f"JSON output:          {json_path}")
        logger.info(f"Total wall-clock:     {total_time:.1f}s")

        if results:
            logger.info("\nTop 3:")
            for r in results[:3]:
                logger.info(
                    f"  #{r['rank']} {r['candidate_id']} "
                    f"score={r['final_score']} -- {r['reasoning_text'][:120]}"
                )

    else:
        # ---- NORMAL MODE: Full pipeline from scratch ----
        logger.info("Mode: FULL (loading candidates and building indices)")

        result = run_full_pipeline(
            candidates_path=str(input_path),
            output_csv=str(output_csv),
            output_json=str(output_json),
            limit=args.limit,
            use_semantic=not args.no_semantic,
            use_reranker=not args.no_reranker,
        )

        total_time = time.time() - t_start

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Candidates loaded: {result['candidates_loaded']}")
        logger.info(f"Candidates ranked: {result['ranked_count']}")
        logger.info(f"CSV output: {result['csv_path']}")
        logger.info(f"JSON output: {result['json_path']}")
        logger.info(f"Total wall-clock:  {total_time:.1f}s")

        if result.get("top_3"):
            logger.info("\nTop 3:")
            for t in result["top_3"]:
                logger.info(f"  #{t['rank']} {t['candidate_id']} score={t['score']} -- {t['reasoning']}")

    # Validate if requested
    if args.validate:
        validator_path = PROJECT_ROOT / "dataset" / "validate_submission.py"
        if validator_path.exists():
            logger.info(f"\nRunning validator: {validator_path}")
            proc = subprocess.run(
                [sys.executable, str(validator_path), str(output_csv)],
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                logger.info(f"PASS: {proc.stdout.strip()}")
            else:
                logger.error(f"FAIL: Validation failed:\n{proc.stdout}\n{proc.stderr}")
        else:
            logger.warning(f"Validator not found at {validator_path}")


if __name__ == "__main__":
    main()

