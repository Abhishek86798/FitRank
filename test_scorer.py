# End-to-end test: feature_builder + scorer on 5 representative sample candidates.
# Run: python test_scorer.py

import json
import yaml
from pathlib import Path

from src.feature_builder import build_feature_vector
from src.scorer import score_with_weighted_sum, LTRScorer

candidates = json.loads(Path("data/sample_candidates.json").read_bytes())
with open("role_model.yaml") as f:
    role_model = yaml.safe_load(f)

# Pick 5 representative candidates (by hand from notes.txt analysis)
SAMPLE_IDS = [
    "CAND_0000031",  # Recsys engineer Swiggy — should be clear #1
    "CAND_0000010",  # Data engineer Ola    — weak adjacent
    "CAND_0000001",  # Backend eng Mindtree — outside India, no ML
    "CAND_0000014",  # Frontend Zomato      — trap (FAISS in skills, career is frontend)
    "CAND_0000004",  # Marketing Manager    — hard disqualify
]

by_id = {c["candidate_id"]: c for c in candidates}

# ── Print full feature vectors ────────────────────────────────────────────────
FEATURES = [
    "cosine_similarity", "experience_fit_score", "is_ml_engineer",
    "title_disqualified", "production_ml_score", "domain_alignment",
    "consulting_penalty", "behavioral_multiplier", "consistency_score",
    "location_score", "notice_penalty", "github_activity",
]

scorer = LTRScorer("artifacts/ltr_model.txt", role_model)
print(f"Scorer mode: {'LTR (LightGBM)' if scorer.is_ltr else 'weighted-sum fallback'}\n")

results = []
for cid in SAMPLE_IDS:
    c = by_id[cid]
    p = c["profile"]
    feats = build_feature_vector(c, role_model, cosine_sim=0.5)  # neutral cosine for comparison
    ws_score = score_with_weighted_sum(feats, role_model)
    results.append((cid, p, feats, ws_score))

    print("=" * 70)
    print(f"{cid}  |  {p.get('current_title')} @ {p.get('current_company')}")
    print(f"YOE: {p.get('years_of_experience')}  Location: {p.get('location')}, {p.get('country')}")
    signals = c.get("redrob_signals", {})
    print(f"Notice: {signals.get('notice_period_days')}d  "
          f"Open: {signals.get('open_to_work_flag')}  "
          f"RespRate: {signals.get('recruiter_response_rate')}  "
          f"LastActive: {signals.get('last_active_date')}")
    print()
    print(f"  {'Feature':<25}  {'Value':>8}  Sanity")
    print(f"  {'-'*25}  {'-'*8}  {'------'}")
    for k in FEATURES:
        v = feats[k]
        flag = ""
        if k == "title_disqualified" and v == -1.0:
            flag = "  ** HARD GATE TRIGGERED"
        elif k == "is_ml_engineer" and v == 1.0:
            flag = "  ** ML engineering role confirmed"
        elif k == "behavioral_multiplier" and v > 0.7:
            flag = "  ** highly reachable"
        elif k == "consulting_penalty" and v > 0.8:
            flag = "  ** almost entire career at consulting"
        print(f"  {k:<25}  {v:>8.4f}{flag}")
    print()
    print(f"  WEIGHTED-SUM SCORE: {ws_score:.4f}")
    print()

# ── Ranking summary ────────────────────────────────────────────────────────────
print("=" * 70)
print("RANKING SUMMARY (weighted-sum, cosine=0.5 for all)")
print(f"{'Rank':<5} {'Score':>6}  {'ID':<15}  Title")
print("-" * 65)
ranked = sorted(results, key=lambda x: x[3], reverse=True)
for i, (cid, p, feats, score) in enumerate(ranked, 1):
    marker = " [BEST]" if cid == "CAND_0000031" else ("  [DISQ]" if feats["title_disqualified"] == -1.0 else "")
    print(f"  {i:<3} {score:>6.4f}  {cid:<15}  {p.get('current_title')}{marker}")

# ── Sanity assertions ──────────────────────────────────────────────────────────
print()
print("SANITY ASSERTIONS")

score_map = {cid: score for cid, _, _, score in results}
feat_map  = {cid: feats for cid, _, feats, _ in results}

fails = []

def check(condition: bool, msg: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        fails.append(msg)

# CAND_0000031 must rank #1
check(ranked[0][0] == "CAND_0000031",
      "CAND_0000031 (recsys engineer) ranks #1")

# Hard gate candidates must score near 0
for cid in ["CAND_0000004"]:
    check(score_map[cid] <= 0.02,
          f"{cid} (marketing mgr) score ≤ 0.02 due to hard gate")

# Hard gate flag must be -1.0 for disqualified titles
check(feat_map["CAND_0000004"]["title_disqualified"] == -1.0,
      "Marketing Manager triggers title_disqualified = -1.0")

# is_ml_engineer must be 1.0 only for the recsys engineer
check(feat_map["CAND_0000031"]["is_ml_engineer"] == 1.0,
      "CAND_0000031 is_ml_engineer = 1.0")
check(feat_map["CAND_0000004"]["is_ml_engineer"] == 0.0,
      "Marketing Manager is_ml_engineer = 0.0")

# CAND_0000031 must beat both adjacent candidates
check(score_map["CAND_0000031"] > score_map["CAND_0000010"],
      "CAND_0000031 scores higher than CAND_0000010 (data engineer)")
check(score_map["CAND_0000031"] > score_map["CAND_0000001"],
      "CAND_0000031 scores higher than CAND_0000001 (backend engineer)")

# All scores in valid range
for cid, _, _, score in results:
    check(0.0 <= score <= 1.0,
          f"{cid} score in [0.0, 1.0]: {score:.4f}")

# All feature values are floats and in expected range
for cid, _, feats, _ in results:
    for k, v in feats.items():
        if k == "title_disqualified":
            check(v in (0.0, -1.0), f"{cid}.{k} is 0.0 or -1.0")
        else:
            check(0.0 <= v <= 1.0, f"{cid}.{k} in [0,1]: {v:.4f}")

print()
if fails:
    print(f"RESULT: {len(fails)} FAILED")
    for f in fails:
        print(f"  ✗ {f}")
else:
    print("RESULT: ALL PASSED")
