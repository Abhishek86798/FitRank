# Smoke test for feature_builder.py against sample_candidates.json.
# Run: python test_feature_builder.py

import json
import yaml
from pathlib import Path
from src.feature_builder import build_feature_vector

candidates = json.loads(Path("data/sample_candidates.json").read_bytes())
with open("role_model.yaml") as f:
    role_model = yaml.safe_load(f)

FEATURES = [
    "cosine_similarity", "experience_fit_score", "is_ml_engineer",
    "title_disqualified", "production_ml_score", "domain_alignment",
    "consulting_penalty", "behavioral_multiplier", "consistency_score",
    "location_score", "notice_penalty", "github_activity",
]

print(f"{'ID':<15} {'exp':>4} {'ml':>4} {'disq':>5} {'prod':>5} {'dom':>5} "
      f"{'cons':>5} {'beh':>5} {'cst':>5} {'loc':>4} {'ntc':>4} {'gh':>4}")
print("-" * 85)

errors = []
for c in candidates:
    feats = build_feature_vector(c, role_model, cosine_sim=0.5)

    # Sanity: all expected keys present
    for k in FEATURES:
        if k not in feats:
            errors.append(f"{c['candidate_id']}: missing key {k}")

    # Sanity: all values are float
    for k, v in feats.items():
        if not isinstance(v, (int, float)):
            errors.append(f"{c['candidate_id']}.{k}: not a float ({v!r})")

    p = c["profile"]
    print(
        f"{c['candidate_id']:<15} "
        f"{feats['experience_fit_score']:>4.2f} "
        f"{feats['is_ml_engineer']:>4.2f} "
        f"{feats['title_disqualified']:>5.1f} "
        f"{feats['production_ml_score']:>5.2f} "
        f"{feats['domain_alignment']:>5.2f} "
        f"{feats['consulting_penalty']:>5.2f} "
        f"{feats['behavioral_multiplier']:>5.2f} "
        f"{feats['consistency_score']:>5.2f} "
        f"{feats['location_score']:>4.1f} "
        f"{feats['notice_penalty']:>4.2f} "
        f"{feats['github_activity']:>4.2f} "
        f"  {p.get('current_title','')[:30]}"
    )

print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
else:
    print("All checks passed.")

# Spot-check: CAND_0000031 should be clearly best
print()
best = next(c for c in candidates if c["candidate_id"] == "CAND_0000031")
feats = build_feature_vector(best, role_model, cosine_sim=0.85)
print("=== Spot-check CAND_0000031 (expected: high scores everywhere) ===")
for k, v in feats.items():
    print(f"  {k:<25} {v}")

# Spot-check: marketing manager should be disqualified
trap = next(c for c in candidates if c["profile"].get("current_title","").lower().startswith("marketing manager"))
feats_trap = build_feature_vector(trap, role_model, cosine_sim=0.7)
print(f"\n=== Spot-check {trap['candidate_id']} ({trap['profile']['current_title']}) — expect title_disqualified=-1.0 ===")
for k, v in feats_trap.items():
    print(f"  {k:<25} {v}")
