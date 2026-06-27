import json
import yaml
from pathlib import Path

import pytest

from src.feature_builder import build_feature_vector
from src.scorer import score_with_weighted_sum, LTRScorer

SAMPLE_JSON = Path("data/sample_candidates.json")
LTR_MODEL = Path("artifacts/ltr_model.txt")

SAMPLE_IDS = [
    "CAND_0000031",  # Recsys engineer Swiggy — should rank #1
    "CAND_0000010",  # Data engineer Ola — weak adjacent
    "CAND_0000001",  # Backend eng Mindtree — outside India, no ML
    "CAND_0000014",  # Frontend Zomato — trap (FAISS in skills, career is frontend)
    "CAND_0000004",  # Marketing Manager — hard disqualify
]


@pytest.fixture(scope="module")
def role_model():
    with open("role_model.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_candidates():
    return json.loads(SAMPLE_JSON.read_bytes())


@pytest.fixture(scope="module")
def scored_results(sample_candidates, role_model):
    by_id = {c["candidate_id"]: c for c in sample_candidates}
    results = {}
    for cid in SAMPLE_IDS:
        c = by_id[cid]
        feats = build_feature_vector(c, role_model, cosine_sim=0.5)
        score = score_with_weighted_sum(feats, role_model)
        results[cid] = {"feats": feats, "score": score}
    return results


def test_best_candidate_ranks_first(scored_results):
    ranked = sorted(scored_results.items(), key=lambda x: x[1]["score"], reverse=True)
    assert ranked[0][0] == "CAND_0000031", "CAND_0000031 (recsys engineer) must rank #1"


def test_marketing_manager_score_near_zero(scored_results):
    score = scored_results["CAND_0000004"]["score"]
    assert score <= 0.02, f"Marketing manager score must be ≤0.02, got {score}"


def test_marketing_manager_title_disqualified(scored_results):
    assert scored_results["CAND_0000004"]["feats"]["title_disqualified"] == -1.0


def test_best_beats_adjacent(scored_results):
    best = scored_results["CAND_0000031"]["score"]
    assert best > scored_results["CAND_0000010"]["score"]
    assert best > scored_results["CAND_0000001"]["score"]


def test_all_scores_in_range(scored_results):
    for cid, r in scored_results.items():
        assert 0.0 <= r["score"] <= 1.0, f"{cid} score out of [0,1]: {r['score']}"


def test_is_ml_engineer_flags(scored_results):
    assert scored_results["CAND_0000031"]["feats"]["is_ml_engineer"] == 1.0
    assert scored_results["CAND_0000004"]["feats"]["is_ml_engineer"] == 0.0


def test_ltr_scorer_loads_and_scores(role_model, scored_results):
    if not LTR_MODEL.exists():
        pytest.skip("ltr_model.txt not present")
    scorer = LTRScorer(LTR_MODEL, role_model)
    assert scorer.is_ltr
    feats = scored_results["CAND_0000031"]["feats"]
    score = scorer.score(feats)
    assert 0.0 <= score <= 1.0


def test_ltr_scorer_fallback_when_model_missing(tmp_path, role_model, scored_results):
    scorer = LTRScorer(tmp_path / "nonexistent.txt", role_model)
    assert not scorer.is_ltr
    feats = scored_results["CAND_0000031"]["feats"]
    score = scorer.score(feats)
    assert 0.0 <= score <= 1.0


def test_compose_reasoning_nonempty_and_bounded():
    from src.feature_builder import build_feature_vector
    from src.reasoning import compose_reasoning
    import yaml, json
    candidates = json.loads(Path("data/sample_candidates.json").read_bytes())
    role_model = yaml.safe_load(open("role_model.yaml"))
    cand = next(c for c in candidates if c["candidate_id"] == "CAND_0000031")
    feats = build_feature_vector(cand, role_model, cosine_sim=0.85)
    text = compose_reasoning(cand, feats, rank=1)
    assert isinstance(text, str)
    assert 10 < len(text) <= 500
    assert text.strip()
