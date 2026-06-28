"""
Tests for eval/honeypot_forensics.py

Verifies:
  1. A hand-built impossible profile triggers SKILL_DURATION_IMPOSSIBLE
  2. A clean ML engineer returns 0 contradictions
  3. TENURE_EXCEEDS_COMPANY fires when a role claims more tenure than time elapsed
  4. YOE_MISMATCH fires when stated YOE departs from career sum by > 3 years
  5. SKILL_COUNT_INFLATION fires when 8+ expert skills have 0 career mentions
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.honeypot_forensics import (
    find_contradictions,
    SKILL_DURATION_IMPOSSIBLE,
    TENURE_EXCEEDS_COMPANY,
    YOE_MISMATCH,
    SKILL_COUNT_INFLATION,
)


# ── fixture helpers ────────────────────────────────────────────────────────────

def _make_clean_ml_engineer() -> dict:
    """A realistic clean ML engineering profile that should trigger no contradictions."""
    return {
        "candidate_id": "TEST_CLEAN_001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "current_company": "Acme AI",
            "years_of_experience": 6.0,
            "location": "Bangalore",
            "country": "India",
        },
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "Acme AI",
                "start_date": "2021-01-01",
                "duration_months": 36,
                "is_current": True,
                "description": (
                    "Built embedding-based retrieval system using FAISS. "
                    "Deployed ranking model with NDCG evaluation. "
                    "A/B tested recommendation pipeline."
                ),
            },
            {
                "title": "ML Engineer",
                "company": "Beta Corp",
                "start_date": "2018-06-01",
                "duration_months": 31,
                "is_current": False,
                "end_date": "2021-01-01",
                "description": (
                    "Built semantic search with dense retrieval and Elasticsearch. "
                    "Shipped production recsys."
                ),
            },
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 72},
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 36},
            {"name": "PyTorch", "proficiency": "advanced", "duration_months": 48},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "notice_period_days": 30,
        },
    }


def _make_skill_duration_impossible() -> dict:
    """Profile with an expert skill claiming duration_months = 0."""
    base = _make_clean_ml_engineer()
    base["skills"].append({"name": "Quantum Computing", "proficiency": "expert", "duration_months": 0})
    return base


def _make_tenure_exceeds_company() -> dict:
    """Profile where a role claims more months than time since start_date."""
    base = _make_clean_ml_engineer()
    # Role that started 6 months ago but claims 48 months of tenure
    from datetime import date, timedelta
    six_months_ago = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    base["career_history"].append({
        "title": "Founder",
        "company": "Very New Startup",
        "start_date": six_months_ago,
        "duration_months": 48,  # 4 years at a 6-month-old company
        "is_current": False,
        "end_date": None,
        "description": "Founded and ran startup.",
    })
    return base


def _make_yoe_mismatch() -> dict:
    """Profile where stated YOE is 15 years but career history sums to < 2 years."""
    base = _make_clean_ml_engineer()
    base["profile"]["years_of_experience"] = 15.0
    # Keep only a short career history
    base["career_history"] = [
        {
            "title": "ML Engineer",
            "company": "Startup",
            "start_date": "2024-01-01",
            "duration_months": 12,
            "is_current": True,
            "description": "Built ML models.",
        }
    ]
    return base


def _make_skill_count_inflation() -> dict:
    """Profile with 10 expert skills, none mentioned in career descriptions."""
    base = _make_clean_ml_engineer()
    # Replace career descriptions with something that mentions none of the skills
    base["career_history"] = [
        {
            "title": "ML Engineer",
            "company": "Corp",
            "start_date": "2020-01-01",
            "duration_months": 48,
            "is_current": True,
            "description": "Worked on data pipelines and general software engineering.",
        }
    ]
    # 10 expert skills, none are "data", "pipelines", or "software"
    base["skills"] = [
        {"name": f"ExoticTech{i}", "proficiency": "expert", "duration_months": 24}
        for i in range(10)
    ]
    return base


# ── tests ──────────────────────────────────────────────────────────────────────

def test_clean_ml_engineer_has_no_contradictions():
    cand = _make_clean_ml_engineer()
    result = find_contradictions(cand)
    assert result == [], (
        f"Clean ML engineer should have 0 contradictions, got: {result}"
    )


def test_skill_duration_impossible_fires():
    cand = _make_skill_duration_impossible()
    result = find_contradictions(cand)
    types = [c["type"] for c in result]
    assert SKILL_DURATION_IMPOSSIBLE in types, (
        f"Expected SKILL_DURATION_IMPOSSIBLE in {types}"
    )
    # Evidence must include the skill name and duration
    hit = next(c for c in result if c["type"] == SKILL_DURATION_IMPOSSIBLE)
    assert hit["evidence"]["duration_months"] == 0
    assert hit["evidence"]["skill"] == "Quantum Computing"


def test_tenure_exceeds_company_fires():
    cand = _make_tenure_exceeds_company()
    result = find_contradictions(cand)
    types = [c["type"] for c in result]
    assert TENURE_EXCEEDS_COMPANY in types, (
        f"Expected TENURE_EXCEEDS_COMPANY in {types}"
    )
    hit = next(c for c in result if c["type"] == TENURE_EXCEEDS_COMPANY)
    assert hit["evidence"]["claimed_duration_months"] == 48
    assert hit["evidence"]["max_plausible_months"] < 48


def test_yoe_mismatch_fires():
    cand = _make_yoe_mismatch()
    result = find_contradictions(cand)
    types = [c["type"] for c in result]
    assert YOE_MISMATCH in types, (
        f"Expected YOE_MISMATCH in {types}"
    )
    hit = next(c for c in result if c["type"] == YOE_MISMATCH)
    assert hit["evidence"]["stated_yoe"] == 15.0
    assert hit["evidence"]["gap_years"] > 3.0


def test_skill_count_inflation_fires():
    cand = _make_skill_count_inflation()
    result = find_contradictions(cand)
    types = [c["type"] for c in result]
    assert SKILL_COUNT_INFLATION in types, (
        f"Expected SKILL_COUNT_INFLATION in {types}"
    )
    hit = next(c for c in result if c["type"] == SKILL_COUNT_INFLATION)
    assert hit["evidence"]["expert_skill_count"] == 10
    assert hit["evidence"]["skills_supported_by_career_text"] == 0


def test_contradictions_return_type():
    """find_contradictions always returns a list of dicts with required keys."""
    for make_fn in [
        _make_clean_ml_engineer,
        _make_skill_duration_impossible,
        _make_tenure_exceeds_company,
        _make_yoe_mismatch,
        _make_skill_count_inflation,
    ]:
        result = find_contradictions(make_fn())
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "type" in item
            assert "detail" in item
            assert "evidence" in item
            assert isinstance(item["detail"], str)
            assert len(item["detail"]) > 0


def test_real_sample_clean_candidates_have_no_contradictions():
    """The clean candidates in sample_candidates.json should return 0 contradictions."""
    sample = Path("data/sample_candidates.json")
    if not sample.exists():
        pytest.skip("data/sample_candidates.json not available")
    import json
    candidates = json.loads(sample.read_bytes())
    # CAND_0000031 (recsys engineer) is the known good candidate
    clean_ids = {"CAND_0000031", "CAND_0000010"}
    for cand in candidates:
        if cand["candidate_id"] in clean_ids:
            result = find_contradictions(cand)
            assert result == [], (
                f"{cand['candidate_id']} is a clean candidate but got contradictions: {result}"
            )
