import json
import yaml
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.feature_builder import build_feature_vector, _role_recency_weight

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


# ── recency weighting tests ───────────────────────────────────────────────────

def test_role_recency_weight_current_role():
    """is_current=True always returns the recency boost."""
    role = {"is_current": True, "end_date": None}
    assert _role_recency_weight(role) == 3.0


def test_role_recency_weight_recent_ended_role():
    """Role that ended 12 months ago is within the 36-month window."""
    recent_end = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    role = {"is_current": False, "end_date": recent_end}
    assert _role_recency_weight(role) == 3.0


def test_role_recency_weight_old_role():
    """Role that ended 5 years ago gets weight 1.0."""
    old_end = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    role = {"is_current": False, "end_date": old_end}
    assert _role_recency_weight(role) == 1.0


def test_role_recency_weight_missing_end_date_treated_as_recent():
    """No end_date and not explicitly current → treated as recent (safe default)."""
    role = {"is_current": False, "end_date": None}
    assert _role_recency_weight(role) == 3.0


def test_production_ml_score_recent_beats_old_same_keywords(role_model):
    """
    Two candidates with identical keywords in descriptions, but one role is
    current and the other ended 5 years ago.  The recent candidate must score
    strictly higher.
    """
    desc = "shipped faiss-based dense retrieval system in production with embedding search"

    old_end = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    cand_old = {
        "candidate_id": "TEST_OLD",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0},
        "skills": [],
        "career_history": [{
            "title": "ML Engineer", "company": "OldCo",
            "start_date": "2015-01-01", "end_date": old_end,
            "is_current": False, "duration_months": 36,
            "description": desc,
        }],
    }
    cand_recent = {
        "candidate_id": "TEST_RECENT",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0},
        "skills": [],
        "career_history": [{
            "title": "ML Engineer", "company": "NewCo",
            "start_date": "2023-01-01", "end_date": None,
            "is_current": True, "duration_months": 18,
            "description": desc,
        }],
    }
    score_old    = build_feature_vector(cand_old,    role_model, cosine_sim=0.5)["production_ml_score"]
    score_recent = build_feature_vector(cand_recent, role_model, cosine_sim=0.5)["production_ml_score"]
    assert score_recent > score_old, (
        f"Recent role should outscore old role: {score_recent} vs {score_old}"
    )


def test_production_ml_score_old_role_still_positive(role_model):
    """An older role with strong keywords still gets a positive score (weight 1.0 not 0)."""
    old_end = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    cand = {
        "candidate_id": "TEST_OLD_POSITIVE",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 8.0},
        "skills": [],
        "career_history": [{
            "title": "ML Engineer", "company": "OldCo",
            "start_date": "2015-01-01", "end_date": old_end,
            "is_current": False, "duration_months": 48,
            "description": "deployed faiss ranking system production embedding a/b test ndcg",
        }],
    }
    score = build_feature_vector(cand, role_model, cosine_sim=0.5)["production_ml_score"]
    assert score > 0.0


def test_production_ml_score_capped_at_1(role_model):
    """Many keywords in a recent role should not exceed 1.0."""
    dense_desc = " ".join([
        "deployed shipped launched production serving faiss weaviate qdrant milvus pinecone",
        "ndcg mrr embedding semantic ranking recsys lambdamart experiment a/b test",
    ])
    cand = {
        "candidate_id": "TEST_CAP",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 7.0},
        "skills": [],
        "career_history": [{
            "title": "ML Engineer", "company": "BigCo",
            "start_date": "2022-01-01", "end_date": None,
            "is_current": True, "duration_months": 30,
            "description": dense_desc,
        }],
    }
    score = build_feature_vector(cand, role_model, cosine_sim=0.5)["production_ml_score"]
    assert score == 1.0


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
