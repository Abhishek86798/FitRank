"""
Tests for evidence-grounded explanations (compose_reasoning_with_citations).

Properties verified:
  1. Every citation has a non-empty source_field
  2. For a real sample ML engineer, all citations are verified=True
  3. A clean ML engineer profile produces ungrounded_count=0
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.reasoning import compose_reasoning_with_citations
from src.feature_builder import build_feature_vector

SAMPLE_JSON   = Path("data/sample_candidates.json")
ROLE_MODEL_PATH = Path("role_model.yaml")

# CAND_0000031 — Recommendation Systems Engineer at Swiggy, the top ML candidate
ML_CAND_ID = "CAND_0000031"


@pytest.fixture(scope="module")
def role_model() -> dict:
    with open(ROLE_MODEL_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    return json.loads(SAMPLE_JSON.read_bytes())


@pytest.fixture(scope="module")
def ml_candidate(sample_candidates) -> dict:
    by_id = {c["candidate_id"]: c for c in sample_candidates}
    cand = by_id.get(ML_CAND_ID)
    if cand is None:
        pytest.skip(f"{ML_CAND_ID} not found in sample_candidates.json")
    return cand


def test_all_claims_have_source_field(sample_candidates, role_model):
    """Every citation dict must have a non-empty source_field string."""
    by_id = {c["candidate_id"]: c for c in sample_candidates}
    cand = by_id.get(ML_CAND_ID)
    if cand is None:
        pytest.skip(f"{ML_CAND_ID} not found")
    features = build_feature_vector(cand, role_model, cosine_sim=0.8)
    result = compose_reasoning_with_citations(cand, features, rank=1)
    assert "citations" in result
    for citation in result["citations"]:
        assert "source_field" in citation, f"Missing source_field in citation: {citation}"
        assert isinstance(citation["source_field"], str), (
            f"source_field is not a string: {citation['source_field']!r}"
        )
        assert citation["source_field"].strip() != "", (
            f"source_field is empty for claim: {citation['claim']!r}"
        )


def test_verified_claims_match_source(ml_candidate, role_model):
    """For the sample ML engineer, all citations should be verified=True."""
    features = build_feature_vector(ml_candidate, role_model, cosine_sim=0.8)
    result = compose_reasoning_with_citations(ml_candidate, features, rank=1)
    unverified = [c for c in result["citations"] if not c["verified"]]
    assert unverified == [], (
        f"Expected all citations verified for {ML_CAND_ID}, "
        f"but got {len(unverified)} unverified: {unverified}"
    )


def test_ungrounded_count_zero_on_clean_profile(ml_candidate, role_model):
    """A clean ML engineer profile must produce ungrounded_count=0."""
    features = build_feature_vector(ml_candidate, role_model, cosine_sim=0.8)
    result = compose_reasoning_with_citations(ml_candidate, features, rank=1)
    assert result["ungrounded_count"] == 0, (
        f"Expected 0 ungrounded claims for {ML_CAND_ID}, "
        f"got {result['ungrounded_count']} / {result['total_claims']}"
    )
