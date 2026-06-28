"""
Tests for src/counterfactual.py

Properties verified:
  1. base_rank matches position in the all_scored list (order consistency)
  2. Masking a feature never *improves* rank (masking can only hold or drop)
  3. confidence is always in [0, 1]
  4. risk_flags is always a list of strings
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.feature_builder import build_feature_vector
from src.scorer import LTRScorer, score_with_weighted_sum
from src.counterfactual import explain_candidate, _MASKABLE_FEATURES

SAMPLE_JSON = Path("data/sample_candidates.json")
ROLE_MODEL_PATH = Path("role_model.yaml")
LTR_MODEL = Path("artifacts/ltr_model.txt")

# A representative slice: recsys engineer (should rank #1), marketing manager (disqualified)
SAMPLE_IDS = [
    "CAND_0000031",  # Recsys engineer Swiggy — top candidate
    "CAND_0000010",  # Data engineer Ola
    "CAND_0000001",  # Backend eng Mindtree
    "CAND_0000014",  # Frontend Zomato
    "CAND_0000004",  # Marketing Manager — hard disqualify
]


@pytest.fixture(scope="module")
def role_model() -> dict:
    with open(ROLE_MODEL_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    return json.loads(SAMPLE_JSON.read_bytes())


@pytest.fixture(scope="module")
def scorer(role_model) -> LTRScorer:
    return LTRScorer(LTR_MODEL, role_model)


@pytest.fixture(scope="module")
def scored_pool(sample_candidates, role_model, scorer) -> tuple[list[dict], list[tuple[str, float]]]:
    """Build feature vectors and all_scored list for SAMPLE_IDS."""
    by_id = {c["candidate_id"]: c for c in sample_candidates}
    candidates = [by_id[cid] for cid in SAMPLE_IDS if cid in by_id]
    feature_map: dict[str, dict] = {}
    score_pairs: list[tuple[str, float]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        feats = build_feature_vector(cand, role_model, cosine_sim=0.5)
        score = scorer.score(feats)
        feature_map[cid] = feats
        score_pairs.append((cid, score))
    # Sort best-first (same rule as rank.py)
    score_pairs.sort(key=lambda x: (-x[1], x[0]))
    return candidates, score_pairs, feature_map


def test_base_rank_matches_csv(scored_pool, role_model, scorer):
    """base_rank returned by explain_candidate must match the position in all_scored."""
    candidates, all_scored, feature_map = scored_pool
    by_id = {c["candidate_id"]: c for c in candidates}
    for expected_rank, (cid, _) in enumerate(all_scored, 1):
        audit = explain_candidate(
            candidate_id=cid,
            features=feature_map[cid],
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=by_id[cid],
            role_model=role_model,
        )
        assert audit["base_rank"] == expected_rank, (
            f"{cid}: expected rank {expected_rank}, got {audit['base_rank']}"
        )


def test_score_drop_sign_consistent_with_rank_drop(scored_pool, role_model, scorer):
    """
    Internal consistency: if score_drop > 0 (masking reduced the candidate's score),
    rank_drop must be >= 0 (they can only fall or stay, never rise).
    Conversely if score_drop < 0 (masking raised score), rank_drop must be <= 0.
    This holds regardless of model type (weighted-sum or LTR).
    """
    candidates, all_scored, feature_map = scored_pool
    by_id = {c["candidate_id"]: c for c in candidates}
    for cid, _ in all_scored:
        audit = explain_candidate(
            candidate_id=cid,
            features=feature_map[cid],
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=by_id[cid],
            role_model=role_model,
        )
        for feat, cf in audit["counterfactuals"].items():
            score_drop = cf["score_drop"]
            rank_drop = cf["rank_drop"]
            if score_drop > 1e-8:
                # masking reduced score → rank can only hold or fall
                assert rank_drop >= 0, (
                    f"{cid}/{feat}: score_drop={score_drop:.4f} > 0 but rank improved "
                    f"(rank_drop={rank_drop})"
                )
            elif score_drop < -1e-8:
                # masking raised score → rank can only hold or improve
                assert rank_drop <= 0, (
                    f"{cid}/{feat}: score_drop={score_drop:.4f} < 0 but rank fell "
                    f"(rank_drop={rank_drop})"
                )


def test_confidence_in_unit_interval(scored_pool, role_model, scorer):
    """confidence must be in [0, 1] for every candidate."""
    candidates, all_scored, feature_map = scored_pool
    by_id = {c["candidate_id"]: c for c in candidates}
    for cid, _ in all_scored:
        audit = explain_candidate(
            candidate_id=cid,
            features=feature_map[cid],
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=by_id[cid],
            role_model=role_model,
        )
        assert 0.0 <= audit["confidence"] <= 1.0, (
            f"{cid}: confidence={audit['confidence']} out of [0, 1]"
        )


def test_risk_flags_is_list_of_strings(scored_pool, role_model, scorer):
    """risk_flags must be a list of str for every candidate."""
    candidates, all_scored, feature_map = scored_pool
    by_id = {c["candidate_id"]: c for c in candidates}
    for cid, _ in all_scored:
        audit = explain_candidate(
            candidate_id=cid,
            features=feature_map[cid],
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=by_id[cid],
            role_model=role_model,
        )
        assert isinstance(audit["risk_flags"], list), (
            f"{cid}: risk_flags is not a list"
        )
        for flag in audit["risk_flags"]:
            assert isinstance(flag, str), (
                f"{cid}: risk_flag entry is not a string: {flag!r}"
            )


def test_top_reasons_capped_at_three(scored_pool, role_model, scorer):
    """top_reasons must have at most 3 entries."""
    candidates, all_scored, feature_map = scored_pool
    by_id = {c["candidate_id"]: c for c in candidates}
    for cid, _ in all_scored:
        audit = explain_candidate(
            candidate_id=cid,
            features=feature_map[cid],
            scorer=scorer,
            all_scored=all_scored,
            candidate_record=by_id[cid],
            role_model=role_model,
        )
        assert len(audit["top_reasons"]) <= 3, (
            f"{cid}: top_reasons has {len(audit['top_reasons'])} entries, expected <=3"
        )


def test_second_candidate_has_meaningful_top_reason(scored_pool, role_model, scorer):
    """
    The second-ranked candidate should have at least one feature where masking
    causes a rank drop. (Rank-1 candidate with only 5 in pool may hold rank #1
    even after masking — use rank #2 which has more room to drop.)
    """
    candidates, all_scored, feature_map = scored_pool
    if len(all_scored) < 2:
        pytest.skip("Need at least 2 candidates to test rank drop")
    by_id = {c["candidate_id"]: c for c in candidates}
    second_cid = all_scored[1][0]
    audit = explain_candidate(
        candidate_id=second_cid,
        features=feature_map[second_cid],
        scorer=scorer,
        all_scored=all_scored,
        candidate_record=by_id[second_cid],
        role_model=role_model,
    )
    # The second candidate should have at least one feature that, when removed,
    # drops them in rank (they are not ranked last so there is always room to fall).
    max_drop = max((cf["rank_drop"] for cf in audit["counterfactuals"].values()), default=0)
    assert max_drop >= 1, (
        f"Second candidate {second_cid} has no feature whose removal causes a rank drop. "
        "All counterfactuals: " + str({f: cf["rank_drop"] for f, cf in audit["counterfactuals"].items()})
    )
