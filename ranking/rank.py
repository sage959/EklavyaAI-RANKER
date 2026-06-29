"""
rank.py — Main entry point for the ranking engine.
Streams a massive JSONL file, keeps the top 100 in a min-heap, and outputs exactly 4 columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import heapq
from pathlib import Path

from ranking import feature_engineering
from ranking import scorer
from ranking import reasoning

# Custom tuple wrapper to handle tie-breaking correctly in a min-heap.
# We want to keep the TOP 100 highest scoring candidates.
# In a Python min-heap (which pops the smallest item), we need the "worst" candidate 
# among the top 100 to be at the root (index 0).
# A candidate is "worse" if:
# 1. Its score is LOWER.
# 2. Or, if scores are tied, its candidate_id is HIGHER (lexicographically descending, so CAND_0000002 is worse than CAND_0000001 because the rules say CAND_0000001 must rank higher).
# Therefore, our sort key should naturally order worse candidates as smaller values.
class HeapItem:
    def __init__(self, score: float, candidate_id: str, features: dict):
        self.score = score
        self.candidate_id = candidate_id
        self.features = features

    def __lt__(self, other):
        if self.score != other.score:
            return self.score < other.score
        # If scores are equal, the "smaller" (worse) candidate is the one with the larger ID string.
        # This means CAND_0000002 < CAND_0000001 is evaluated as True.
        # This keeps the worst candidate at the root of the min-heap so it gets popped.
        return self.candidate_id > other.candidate_id


def process_jsonl(input_file: str, top_k: int = 100):
    """
    Stream a massive JSONL file line-by-line.
    Score candidates on the fly and maintain a min-heap of the top K.
    Returns a list of the top K features dicts, sorted descending by rank.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    top_candidates = [] # min-heap
    count = 0

    print(f"Streaming candidates from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} candidates...")

            # 1. Extract features
            features = feature_engineering.extract_all_features(candidate)
            
            # 2. Compute final score (0-100)
            features = scorer.compute_final_score(features)
            score_val = features["final_score"]
            cid = features["candidate_id"]
            
            # 3. Maintain top-K heap
            item = HeapItem(score_val, cid, features)
            
            if len(top_candidates) < top_k:
                heapq.heappush(top_candidates, item)
            else:
                heapq.heappushpop(top_candidates, item)

    print(f"Total candidates evaluated: {count}")
    
    # Extract from heap and sort descending (best first)
    # The heap gives items in arbitrary order, but sorted() will use our __lt__ 
    # which orders worse items first. We reverse it to get best items first.
    best_items = sorted(top_candidates, reverse=True)
    return [item.features for item in best_items]


def run_ranking(input_jsonl: str, output_csv: str):
    """
    Full pipeline execution.
    """
    try:
        ranked_features = process_jsonl(input_jsonl, top_k=100)
    except Exception as e:
        print(f"Fatal error processing data: {e}")
        return

    if not ranked_features:
        print("No candidates were ranked.")
        return

    # Generate reasoning for only the top 100
    for features in ranked_features:
        features["reasoning_text"] = reasoning.generate_reasoning(features)

    # Export exactly 4 columns: candidate_id,rank,score,reasoning
    print(f"Exporting to {output_csv}...")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank_idx, f_data in enumerate(ranked_features, start=1):
            writer.writerow([
                f_data["candidate_id"],
                rank_idx,
                f_data["final_score"],
                f_data["reasoning_text"]
            ])

    print(f"Done! Top {len(ranked_features)} candidates exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eklavya AI - Deterministic Ranking Engine")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dataset/candidates.jsonl",
        help="Path to candidates.jsonl file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/TIER3.csv",
        help="Path to output CSV file"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_path = project_root / args.input
    output_path = project_root / args.output
    
    run_ranking(str(input_path), str(output_path))
