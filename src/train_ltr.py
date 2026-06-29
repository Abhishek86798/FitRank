# Train a LightGBM LambdaMART model on the golden set and save to artifacts/ltr_model.txt.
#
# Inputs:
#   eval/golden_set.csv          — candidate_id, relevance_label (0-3 scale), notes
#   artifacts/embeddings.npy     — (n_candidates, 768) fp16, L2-normalised
#   artifacts/candidate_ids.npy  — (n_candidates,) string IDs aligned with embeddings
#   artifacts/jd_vector.npy      — (1, 768) fp16, L2-normalised JD embedding
#   data/candidates.jsonl        — full candidate records (streamed, not loaded in bulk)
#   role_model.yaml              — for build_feature_vector()
#
# Output:
#   artifacts/ltr_model.txt      — LightGBM booster, loaded by LTRScorer in scorer.py

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

# Feature column order must match scorer.py LTRScorer.FEATURE_ORDER exactly
FEATURE_ORDER = [
    "cosine_similarity",
    "experience_fit_score",
    "is_ml_engineer",
    "production_ml_score",
    "domain_alignment",
    "consulting_penalty",
    "behavioral_multiplier",
    "consistency_score",
    "location_score",
    "notice_penalty",
    "github_activity",
    "ce_score",
    "open_to_work_score",
    "response_rate_score",
    "recency_score",
    "response_speed_score",
    "interview_reliability",
    "active_job_seeking",
    "market_validation",
    "skill_depth_score",
    "education_tier_score",
    "profile_completeness",
]


def _load_golden(golden_path: Path) -> dict[str, int]:
    """Return {candidate_id: relevance_label} from golden_set.csv."""
    labels: dict[str, int] = {}
    with open(golden_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"].strip()
            label = int(row["relevance_label"])
            labels[cid] = label
    return labels


def _cosine_sim(emb: np.ndarray, jd: np.ndarray) -> float:
    """Dot product of two L2-normalised fp32 vectors → cosine similarity."""
    return float(np.dot(emb.astype(np.float32), jd.astype(np.float32)))


def _stream_candidates(candidates_path: Path, target_ids: set[str]) -> dict[str, dict]:
    """Stream candidates.jsonl and return records for target_ids only."""
    import orjson
    records: dict[str, dict] = {}
    with open(candidates_path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            cand = orjson.loads(raw)
            cid = cand["candidate_id"]
            if cid in target_ids:
                records[cid] = cand
            if len(records) == len(target_ids):
                break
    return records


def run(
    golden_path: Path,
    artifacts_dir: Path,
    candidates_path: Path,
    role_model_path: Path,
    output_path: Path,
) -> None:
    import lightgbm as lgb
    import yaml

    from src.feature_builder import build_feature_vector

    # ── 1. Load golden labels ─────────────────────────────────────────────────
    golden = _load_golden(golden_path)
    print(f"Golden set: {len(golden)} candidates  "
          f"(labels: {sorted(set(golden.values()))})", flush=True)

    # ── 2. Load artifacts ─────────────────────────────────────────────────────
    print("Loading embeddings ...", flush=True)
    embeddings    = np.load(artifacts_dir / "embeddings.npy").astype(np.float32)
    candidate_ids = np.load(artifacts_dir / "candidate_ids.npy", allow_pickle=True)
    jd_vector     = np.load(artifacts_dir / "jd_vector.npy").astype(np.float32).squeeze()

    import json as _json_ce
    ce_scores_path = artifacts_dir / "ce_scores.json"
    ce_scores: dict = {}
    if ce_scores_path.exists():
        ce_scores = _json_ce.loads(ce_scores_path.read_bytes())
        print(f"  Loaded ce_scores for {len(ce_scores)} candidates", flush=True)

    print(f"  embeddings shape={embeddings.shape}  ids={len(candidate_ids)}", flush=True)

    # Build id→row index for fast cosine lookup
    id_to_idx: dict[str, int] = {str(cid): i for i, cid in enumerate(candidate_ids)}

    # ── 3. Load role model ────────────────────────────────────────────────────
    with open(role_model_path, encoding="utf-8") as f:
        role_model: dict = yaml.safe_load(f)

    # ── 4. Stream candidate records for golden IDs ────────────────────────────
    print(f"Streaming {candidates_path.name} for golden candidate records ...", flush=True)
    records = _stream_candidates(candidates_path, set(golden.keys()))
    missing = set(golden.keys()) - set(records.keys())
    if missing:
        print(f"  [warn] {len(missing)} golden IDs not found in candidates file: "
              f"{sorted(missing)[:5]}", flush=True)

    # ── 5. Build feature matrix ───────────────────────────────────────────────
    print("Building feature matrix ...", flush=True)

    ordered_ids: list[str] = []
    X_rows: list[list[float]] = []
    y: list[int] = []

    for cid, label in sorted(golden.items()):
        cand = records.get(cid)
        if cand is None:
            continue

        idx = id_to_idx.get(cid)
        cosine = _cosine_sim(embeddings[idx], jd_vector) if idx is not None else 0.0

        features = build_feature_vector(cand, role_model, cosine_sim=cosine, ce_scores=ce_scores)
        X_rows.append([features.get(k, 0.0) for k in FEATURE_ORDER])
        y.append(label)
        ordered_ids.append(cid)

    X = np.array(X_rows, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    n = len(y_arr)
    print(f"  {n} training examples  X.shape={X.shape}", flush=True)

    if n == 0:
        print("ERROR: no training examples — check golden_set.csv and candidates.jsonl paths",
              flush=True)
        sys.exit(1)

    # ── 6. Train LambdaMART ───────────────────────────────────────────────────
    # Single query group: all candidates belong to the same JD query
    group = [n]

    print(f"Training LightGBM LambdaMART  "
          f"(group=[{n}], num_boost_round=100) ...", flush=True)

    train_data = lgb.Dataset(
        X, label=y_arr,
        group=group,
        feature_name=FEATURE_ORDER,
        free_raw_data=False,
    )

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [10],
        "num_leaves":       31,
        "learning_rate":    0.05,
        "min_data_in_leaf": 1,   # golden set is small; allow single-example leaves
        "verbose":          -1,
    }

    callbacks = [lgb.log_evaluation(period=20)]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        valid_names=["train"],
        callbacks=callbacks,
    )

    # ── 7. Save model ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    print(f"\nSaved model: {output_path}", flush=True)

    # ── 8. Feature importance (gain) ──────────────────────────────────────────
    importance = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(FEATURE_ORDER, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop-5 features by gain importance:")
    for fname, gain in feat_imp[:5]:
        print(f"  {fname:<28}  gain={gain:.2f}")

    # ── 9. NDCG@10 on training set ────────────────────────────────────────────
    preds = booster.predict(X)
    order = np.argsort(preds)[::-1]
    y_sorted = y_arr[order]

    def _dcg_at_k(labels: np.ndarray, k: int) -> float:
        top = labels[:k].astype(float)
        gains = (2.0 ** top - 1.0) / np.log2(np.arange(2, len(top) + 2))
        return float(gains.sum())

    ideal_order = np.sort(y_arr)[::-1]
    dcg  = _dcg_at_k(y_sorted, k=10)
    idcg = _dcg_at_k(ideal_order, k=10)
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    print(f"\nNDCG@10 on training set: {ndcg10:.4f}  "
          f"(expected high — single-query overfit confirms training worked)")

    print("\nTop-10 predicted ranking (id | pred_score | true_label):")
    for rank_i, idx_i in enumerate(order[:10], 1):
        print(f"  #{rank_i:2d}  {ordered_ids[idx_i]}  "
              f"pred={preds[idx_i]:+.4f}  label={y_arr[idx_i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaMART on golden set.")
    parser.add_argument("--golden",        default="eval/golden_set.csv",    help="Golden set CSV")
    parser.add_argument("--artifacts-dir", default="artifacts",               help="Artifacts directory")
    parser.add_argument("--candidates",    default="data/candidates.jsonl",   help="Candidate data file")
    parser.add_argument("--role-model",    default="role_model.yaml",         help="Role model YAML")
    parser.add_argument("--output",        default="artifacts/ltr_model.txt", help="Output model path")
    args = parser.parse_args()

    run(
        golden_path     = Path(args.golden),
        artifacts_dir   = Path(args.artifacts_dir),
        candidates_path = Path(args.candidates),
        role_model_path = Path(args.role_model),
        output_path     = Path(args.output),
    )


if __name__ == "__main__":
    main()
