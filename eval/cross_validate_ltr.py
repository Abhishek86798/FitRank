import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score

from src.feature_builder import build_feature_vector
from eval.evaluate import _average_precision, _precision_at_k

# Feature column order must match scorer.py
FEATURE_ORDER = [
    "cosine_similarity",
    "experience_fit_score",
    "is_ml_engineer",
    "production_ml_score",
    "domain_alignment",
    "consulting_penalty",
    "behavioral_multiplier",
    "location_score",
    "notice_penalty",
    "github_activity",
    "ce_score",
    "response_rate_score",
    "active_job_seeking",
    "skill_depth_score",
    "profile_completeness",
]

def _load_golden(golden_path: Path) -> dict[str, int]:
    labels = {}
    with open(golden_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["candidate_id"].strip()] = int(row["relevance_label"])
    return labels

def _cosine_sim(emb: np.ndarray, jd: np.ndarray) -> float:
    return float(np.dot(emb.astype(np.float32), jd.astype(np.float32)))

def _stream_candidates(candidates_path: Path, target_ids: set[str]) -> dict[str, dict]:
    import orjson
    records = {}
    with open(candidates_path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw: continue
            cand = orjson.loads(raw)
            if cand["candidate_id"] in target_ids:
                records[cand["candidate_id"]] = cand
            if len(records) == len(target_ids):
                break
    return records

def _ndcg_at(rel: list[int], ideal: list[int], k: int) -> float:
    n = max(len(rel), len(ideal))
    true_vec = np.zeros(n, dtype=float)
    ideal_vec = np.zeros(n, dtype=float)
    for i, v in enumerate(rel): true_vec[i] = v
    for i, v in enumerate(ideal): ideal_vec[i] = v
    if ideal_vec.sum() == 0: return 0.0
    return float(ndcg_score([ideal_vec], [true_vec], k=k))

def main():
    golden_path = Path("eval/golden_set.csv")
    artifacts_dir = Path("artifacts")
    candidates_path = Path("data/candidates.jsonl")
    role_model_path = Path("role_model.yaml")

    print("Loading Golden Set...")
    golden = _load_golden(golden_path)
    
    print("Loading Artifacts...")
    embeddings = np.load(artifacts_dir / "embeddings.npy").astype(np.float32)
    candidate_ids = np.load(artifacts_dir / "candidate_ids.npy", allow_pickle=True)
    jd_vector = np.load(artifacts_dir / "jd_vector.npy").astype(np.float32).squeeze()
    
    import json
    ce_scores_path = artifacts_dir / "ce_scores.json"
    ce_scores = json.loads(ce_scores_path.read_bytes()) if ce_scores_path.exists() else {}

    id_to_idx = {str(cid): i for i, cid in enumerate(candidate_ids)}
    with open(role_model_path, encoding="utf-8") as f:
        role_model = yaml.safe_load(f)

    print("Extracting candidate records...")
    records = _stream_candidates(candidates_path, set(golden.keys()))

    print("Building Feature Matrix...")
    X_rows = []
    y = []
    ordered_ids = []

    for cid, label in sorted(golden.items()):
        cand = records.get(cid)
        if not cand: continue
        idx = id_to_idx.get(cid)
        cosine = _cosine_sim(embeddings[idx], jd_vector) if idx is not None else 0.0
        features = build_feature_vector(cand, role_model, cosine_sim=cosine, ce_scores=ce_scores)
        X_rows.append([features.get(k, 0.0) for k in FEATURE_ORDER])
        y.append(label)
        ordered_ids.append(cid)

    X = np.array(X_rows, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    ordered_ids = np.array(ordered_ids)

    print(f"Total valid examples: {len(y_arr)}")
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {"NDCG@10": [], "NDCG@50": [], "MAP": [], "P@10": []}
    
    print("\nStarting 5-Fold Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y_arr[train_idx]
        X_test, y_test = X[test_idx], y_arr[test_idx]
        test_cids = ordered_ids[test_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, group=[len(y_train)], free_raw_data=False)
        
        params = {
            "objective":        "lambdarank",
            "metric":           "ndcg",
            "ndcg_eval_at":     [10],
            "num_leaves":       7,
            "learning_rate":    0.01,
            "min_data_in_leaf": 4,
            "feature_fraction": 0.8,
            "feature_fraction_seed": 42,
            "verbose":          -1,
        }
        
        booster = lgb.train(params, train_data, num_boost_round=300)
        preds = booster.predict(X_test)
        
        # Sort predictions
        order = np.argsort(preds)[::-1]
        ranked_test_cids = test_cids[order]
        
        # We need to map ranked predictions to true labels (relevance vector)
        test_labels_map = {cid: lbl for cid, lbl in zip(test_cids, y_test)}
        rel = [test_labels_map[cid] for cid in ranked_test_cids]
        ideal = sorted(y_test, reverse=True)
        
        metrics["NDCG@10"].append(_ndcg_at(rel, ideal, 10))
        metrics["NDCG@50"].append(_ndcg_at(rel, ideal, 50))
        metrics["MAP"].append(_average_precision(rel))
        metrics["P@10"].append(_precision_at_k(rel, 10))
        
    print("\nPrimary metric: held-out, cross-validated (the honest number)")
    print("-" * 65)
    print(f"{'Metric':<15} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 65)
    
    for metric in ["NDCG@10", "NDCG@50", "MAP", "P@10"]:
        mean_val = np.mean(metrics[metric])
        std_val = np.std(metrics[metric])
        print(f"{metric:<15} {mean_val:<15.4f} ±{std_val:<14.4f}")

if __name__ == "__main__":
    main()
