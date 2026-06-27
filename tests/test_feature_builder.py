import json
import yaml
from pathlib import Path

import pytest

from src.feature_builder import build_feature_vector

SAMPLE_JSON = Path("data/sample_candidates.json")

FEATURES = [
    "cosine_similarity", "experience_fit_score", "is_ml_engineer",
    "title_disqualified", "production_ml_score", "domain_alignment",
    "consulting_penalty", "behavioral_multiplier", "consistency_score",
    "location_score", "notice_penalty", "github_activity",
]


@pytest.fixture(scope="module")
def role_model():
    with open("role_model.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_candidates():
    return json.loads(SAMPLE_JSON.read_bytes())


def test_all_feature_keys_present(sample_candidates, role_model):
    for c in sample_candidates:
        feats = build_feature_vector(c, role_model, cosine_sim=0.5)
        for k in FEATURES:
            assert k in feats, f"{c['candidate_id']}: missing key {k}"


def test_all_feature_values_are_floats(sample_candidates, role_model):
    for c in sample_candidates:
        feats = build_feature_vector(c, role_model, cosine_sim=0.5)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"{c['candidate_id']}.{k}: not a float ({v!r})"


def test_feature_values_in_range(sample_candidates, role_model):
    for c in sample_candidates:
        feats = build_feature_vector(c, role_model, cosine_sim=0.5)
        for k, v in feats.items():
            if k == "title_disqualified":
                assert v in (0.0, -1.0), f"{c['candidate_id']}.{k} must be 0.0 or -1.0"
            else:
                assert 0.0 <= v <= 1.0, f"{c['candidate_id']}.{k} out of [0,1]: {v}"


def test_best_candidate_high_scores(sample_candidates, role_model):
    best = next(c for c in sample_candidates if c["candidate_id"] == "CAND_0000031")
    feats = build_feature_vector(best, role_model, cosine_sim=0.85)
    assert feats["is_ml_engineer"] == 1.0
    assert feats["title_disqualified"] == 0.0


def test_marketing_manager_disqualified(sample_candidates, role_model):
    trap = next(
        c for c in sample_candidates
        if c["profile"].get("current_title", "").lower().startswith("marketing manager")
    )
    feats = build_feature_vector(trap, role_model, cosine_sim=0.7)
    assert feats["title_disqualified"] == -1.0


def test_title_disqualified_redemption_by_past_ml_role(role_model):
    cand = {
        "candidate_id": "TEST_REDEMPTION",
        "profile": {"current_title": "Marketing Manager"},
        "career_history": [{"title": "ML Engineer", "description": "built ml models"}],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.0)
    assert feats["title_disqualified"] == 0.0
