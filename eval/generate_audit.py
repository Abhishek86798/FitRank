"""
Generate a counterfactual decision audit for the top-20 candidates in team_xxx.csv.

Usage
-----
    python eval/generate_audit.py [--submission team_xxx.csv] [--top-n 20]
                                  [--candidates data/candidates.jsonl]
                                  [--role-model role_model.yaml]
                                  [--output eval/decision_audit.json]

Writes eval/decision_audit.json and prints a readable summary for one example.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_submission(path: Path, top_n: int) -> list[tuple[int, str, float]]:
    """Return [(rank, candidate_id, score), ...] for the top-n rows."""
    rows: list[tuple[int, str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["rank"]), row["candidate_id"].strip(), float(row["score"])))
    rows.sort(key=lambda x: x[0])
    return rows[:top_n]


def _load_candidates(path: Path, needed_ids: set[str]) -> dict[str, dict]:
    """Stream candidates file and return {id: record} for needed IDs only."""
    records: dict[str, dict] = {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_bytes())
        for cand in data:
            cid = cand["candidate_id"]
            if cid in needed_ids:
                records[cid] = cand
    else:
        import orjson
        with open(path, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cand = orjson.loads(line)
                cid = cand["candidate_id"]
                if cid in needed_ids:
                    records[cid] = cand
                if records.keys() >= needed_ids:
                    break
    return records


def _print_audit_summary(audit: dict) -> None:
    """Print a human-readable summary of one audit record."""
    print(f"\n{'='*60}")
    print(f"Candidate : {audit['candidate_id']}")
    print(f"Base rank : #{audit['base_rank']}  (score={audit['base_score']:.4f})")
    print(f"Confidence: {audit['confidence']:.4f}")

    print("\nTop reasons (features that matter most):")
    for i, r in enumerate(audit["top_reasons"], 1):
        print(
            f"  {i}. {r['feature']:<25s}  rank_drop={r['rank_drop']:+d}  "
            f"score_drop={r['score_drop']:+.4f}"
        )

    print("\nFull counterfactual table:")
    print(f"  {'Feature':<25s}  {'rank_if_removed':>15s}  {'rank_drop':>9s}  {'score_drop':>10s}")
    print(f"  {'-'*65}")
    cf = audit["counterfactuals"]
    for feat in sorted(cf, key=lambda f: -cf[f]["rank_drop"]):
        row = cf[feat]
        print(
            f"  {feat:<25s}  {row['rank_if_removed']:>15d}  "
            f"{row['rank_drop']:>+9d}  {row['score_drop']:>+10.4f}"
        )

    if audit["risk_flags"]:
        print("\nRisk flags:")
        for flag in audit["risk_flags"]:
            print(f"  ⚑  {flag}")
    else:
        print("\nRisk flags: none")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate counterfactual decision audit")
    parser.add_argument("--submission",   default="team_xxx.csv",          help="Ranked submission CSV")
    parser.add_argument("--top-n",        type=int, default=20,             help="Number of top candidates to audit")
    parser.add_argument("--candidates",   default="data/candidates.jsonl",  help="Candidate data file")
    parser.add_argument("--role-model",   default="role_model.yaml",        help="Role model YAML")
    parser.add_argument("--output",       default="eval/decision_audit.json", help="Output JSON path")
    parser.add_argument("--ltr-model",    default="artifacts/ltr_model.txt", help="LambdaMART model")
    args = parser.parse_args()

    import yaml
    from src.feature_builder import build_feature_vector
    from src.scorer import LTRScorer
    from src.counterfactual import explain_candidate

    print(f"Loading role model from {args.role_model} ...")
    with open(args.role_model, encoding="utf-8") as f:
        role_model: dict = yaml.safe_load(f)

    print(f"Loading submission from {args.submission} ...")
    top_rows = _load_submission(Path(args.submission), args.top_n)
    needed_ids = {cid for _, cid, _ in top_rows}

    print(f"Loading {len(needed_ids)} candidate records from {args.candidates} ...")
    records = _load_candidates(Path(args.candidates), needed_ids)
    missing = needed_ids - records.keys()
    if missing:
        print(f"  [warn] {len(missing)} IDs not found in candidates file: {missing}", file=sys.stderr)

    scorer = LTRScorer(args.ltr_model, role_model)
    print(f"  Scorer mode: {'LambdaMART' if scorer.is_ltr else 'weighted-sum'}")

    # Build all_scored from the submission — full list for rank computation.
    # We only have scores for the top-N rows from the CSV; the counterfactual
    # rank is relative to this pool (the same pool used for the submission).
    all_scored: list[tuple[str, float]] = [(cid, score) for _, cid, score in top_rows]

    print(f"\nGenerating counterfactual audits for top-{args.top_n} candidates ...")
    audits: list[dict] = []
    for rank, cid, score in top_rows:
        if cid not in records:
            continue
        cand = records[cid]
        # cosine_similarity is not stored in the CSV; approximate from score rank as 0.
        # In a real pipeline you'd persist cosine sims alongside the submission.
        # We use 0.0 here; feature_builder fills all other features from the record.
        # For a more accurate audit, pass the actual cosine sim if available.
        features = build_feature_vector(cand, role_model, cosine_sim=0.0)
        audit = explain_candidate(
            candidate_id=cid,
            features=features,
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=cand,
            role_model=role_model,
        )
        audits.append(audit)
        print(f"  #{rank:3d}  {cid}  base_score={audit['base_score']:.4f}  "
              f"confidence={audit['confidence']:.4f}  "
              f"top_reason={audit['top_reasons'][0]['feature'] if audit['top_reasons'] else 'n/a'}")

    # Write JSON output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"\nWrote {len(audits)} audit records to {out_path}")

    # Print a full example (first candidate)
    if audits:
        print("\n--- Full audit example (rank #1) ---")
        _print_audit_summary(audits[0])


if __name__ == "__main__":
    main()
