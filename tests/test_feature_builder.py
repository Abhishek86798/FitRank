import json
import yaml
from pathlib import Path

import pytest

from src.feature_builder import build_feature_vector

SAMPLE_JSON = Path("data/sample_candidates.json")

FEATURES = [
    "cosine_similarity", "experience_fit_score", "is_ml_engineer",
    "title_disqualified", "impossibility_flag", "production_ml_score",
    "domain_alignment", "consulting_penalty", "behavioral_multiplier",
    "consistency_score", "location_score", "notice_penalty", "github_activity",
]

# Features whose valid values are {-1.0, 0.0} rather than [0.0, 1.0]
_GATE_FEATURES = {"title_disqualified", "impossibility_flag"}


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
            if k in _GATE_FEATURES:
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


# ── impossibility_flag tests ──────────────────────────────────────────────────

def test_impossibility_flag_clean_candidate(sample_candidates, role_model):
    """A well-formed candidate should not trigger the flag."""
    best = next(c for c in sample_candidates if c["candidate_id"] == "CAND_0000031")
    feats = build_feature_vector(best, role_model, cosine_sim=0.85)
    assert feats["impossibility_flag"] == 0.0


def test_impossibility_flag_expert_skill_zero_duration(role_model):
    """Expert skill with duration_months=0 is an impossibility — triggers flag."""
    cand = {
        "candidate_id": "TEST_ZERO_DURATION",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 5.0},
        "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 0}],
        "career_history": [],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.5)
    assert feats["impossibility_flag"] == -1.0


def test_impossibility_flag_advanced_skill_zero_duration(role_model):
    """Advanced skill with duration_months=0 also triggers flag."""
    cand = {
        "candidate_id": "TEST_ZERO_DURATION_ADV",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 4.0},
        "skills": [{"name": "LambdaMART", "proficiency": "advanced", "duration_months": 0}],
        "career_history": [],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.5)
    assert feats["impossibility_flag"] == -1.0


def test_impossibility_flag_intermediate_zero_duration_ok(role_model):
    """Intermediate skill with zero duration is unusual but not an impossibility."""
    cand = {
        "candidate_id": "TEST_INTERMEDIATE_ZERO",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 3.0},
        "skills": [{"name": "Docker", "proficiency": "intermediate", "duration_months": 0}],
        "career_history": [],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.5)
    assert feats["impossibility_flag"] == 0.0


def test_impossibility_flag_tenure_exceeds_company_age(role_model):
    """Role duration_months > months since start_date is impossible — triggers flag."""
    from datetime import date, timedelta
    # Role started 6 months ago but claims 24 months duration
    start = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    cand = {
        "candidate_id": "TEST_TENURE_IMPOSSIBLE",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 3.0},
        "skills": [],
        "career_history": [{
            "company": "Fake Corp",
            "title": "ML Engineer",
            "start_date": start,
            "duration_months": 24,   # claims 24 months at a company started 6 months ago
        }],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.5)
    assert feats["impossibility_flag"] == -1.0


def test_impossibility_flag_plausible_tenure_ok(role_model):
    """Role with plausible start date and matching duration does not trigger flag."""
    cand = {
        "candidate_id": "TEST_TENURE_OK",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 5.0},
        "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 24}],
        "career_history": [{
            "company": "Good Corp",
            "title": "ML Engineer",
            "start_date": "2021-01-01",
            "duration_months": 36,
        }],
    }
    feats = build_feature_vector(cand, role_model, cosine_sim=0.7)
    assert feats["impossibility_flag"] == 0.0


def test_impossibility_flag_gates_score_to_001(role_model):
    """When impossibility_flag fires, scorer must return exactly 0.01."""
    from src.scorer import score_with_weighted_sum
    feats = build_feature_vector(
        {
            "candidate_id": "TEST_GATE",
            "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0},
            "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 0}],
            "career_history": [],
        },
        role_model,
        cosine_sim=0.9,   # high cosine — gate must override this
    )
    assert feats["impossibility_flag"] == -1.0
    score = score_with_weighted_sum(feats, role_model)
    assert score == 0.01
