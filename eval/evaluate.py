"""
Compute NDCG@10, NDCG@50, MAP, and P@10 for a ranked submission.

Usage
-----
    python eval/evaluate.py <submission.csv> [--golden eval/golden_set.csv]

Submission CSV  : candidate_id, rank, score, reasoning  (validator format)
Golden set CSV  : candidate_id, relevance_label, notes
                  relevance_label ∈ {0, 1, 2, 3}  (0=irrelevant, 3=perfect)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score


def _load_submission(path: Path) -> list[tuple[int, str]]:
    """Return [(rank, candidate_id), ...] sorted by rank ascending."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip()))
    rows.sort(key=lambda x: x[0])
    return rows


def _load_golden(path: Path) -> dict[str, int]:
    """Return {candidate_id: relevance_label}."""
    labels: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["candidate_id"].strip()] = int(row["relevance_label"])
    return labels


def _relevance_vector(ranked_ids: list[str], labels: dict[str, int]) -> list[int]:
    """Map ranked list to relevance scores (0 for candidates not in golden set)."""
    return [labels.get(cid, 0) for cid in ranked_ids]


def _average_precision(relevance: list[int], threshold: int = 1) -> float:
    """AP — fraction of relevant docs × precision at each relevant position."""
    hits = precisions = 0.0
    for i, r in enumerate(relevance, 1):
        if r >= threshold:
            hits += 1
            precisions += hits / i
    total_relevant = sum(1 for r in relevance if r >= threshold)
    if total_relevant == 0:
        return 0.0
    return precisions / total_relevant


def _precision_at_k(relevance: list[int], k: int, threshold: int = 1) -> float:
    top_k = relevance[:k]
    return sum(1 for r in top_k if r >= threshold) / k


def evaluate(submission_path: Path, golden_path: Path) -> dict[str, float]:
    ranked = _load_submission(submission_path)
    labels = _load_golden(golden_path)

    ranked_ids = [cid for _, cid in ranked]
    rel = _relevance_vector(ranked_ids, labels)

    max_label = max(labels.values()) if labels else 3
    ideal = sorted(labels.values(), reverse=True)

    def _ndcg_at(k: int) -> float:
        # sklearn ndcg_score requires both arrays to be the same length
        # and at least 2 elements; we use the full list and pass k= cutoff.
        n = max(len(rel), len(ideal))
        true_vec  = np.zeros(n, dtype=float)
        ideal_vec = np.zeros(n, dtype=float)
        for i, v in enumerate(rel):
            true_vec[i] = v
        for i, v in enumerate(ideal):
            ideal_vec[i] = v
        if ideal_vec.sum() == 0:
            return 0.0
        return float(ndcg_score([ideal_vec], [true_vec], k=k))

    metrics = {
        "NDCG@10":  _ndcg_at(10),
        "NDCG@50":  _ndcg_at(50),
        "MAP":      _average_precision(rel),
        "P@10":     _precision_at_k(rel, 10),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked submission.")
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument(
        "--golden", default="eval/golden_set.csv",
        help="Path to golden set CSV (default: eval/golden_set.csv)",
    )
    args = parser.parse_args()

    metrics = evaluate(Path(args.submission), Path(args.golden))
    width = max(len(k) for k in metrics)
    print("\nEvaluation results")
    print("-" * (width + 12))
    for k, v in metrics.items():
        print(f"  {k:<{width}}  {v:.4f}")
    print()


if __name__ == "__main__":
    main()
