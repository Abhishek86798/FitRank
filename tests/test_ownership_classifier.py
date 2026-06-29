"""
Tests for src/ownership_classifier.py

Seven required cases that encode the core discriminating property:
genuine delivery language vs peripheral observation/testing language.
"""

from __future__ import annotations

import pytest

from src.ownership_classifier import ownership_score

_DOMAIN_KWS = [
    "ranking", "recommendation", "retrieval", "search", "embedding",
    "faiss", "rag", "ltr", "recsys", "nlp", "vector",
]


def test_builder_scores_high():
    """'Built a RAG-based ranking pipeline' → strong ownership."""
    desc = "Built a RAG-based ranking pipeline serving 50M queries per day."
    assert ownership_score(desc, _DOMAIN_KWS) > 0.7, (
        f"Expected > 0.7, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_tester_scores_low():
    """'Tested the frontend UI … Recommendation System API' → peripheral."""
    desc = (
        "I tested the frontend UI for the backend team's Recommendation System API. "
        "Worked with QA to verify edge cases and wrote test reports."
    )
    assert ownership_score(desc, _DOMAIN_KWS) < 0.3, (
        f"Expected < 0.3, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_seminar_attendee_scores_very_low():
    """'Attended a seminar on LLMs and Ranking Systems' → near-zero."""
    desc = "Attended a seminar on LLMs and Ranking Systems organised by the university."
    assert ownership_score(desc, _DOMAIN_KWS) < 0.2, (
        f"Expected < 0.2, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_end_to_end_owner_scores_very_high():
    """'Owned the end-to-end ranking pipeline' → top ownership tier."""
    desc = (
        "Owned the end-to-end ranking pipeline: sourcing → embedding → LTR re-scoring. "
        "Architected the retrieval stack and led the team of 3 engineers that shipped it."
    )
    assert ownership_score(desc, _DOMAIN_KWS) > 0.8, (
        f"Expected > 0.8, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_course_project_familiar_scores_low():
    """'Familiar with FAISS … from a course project' → very low."""
    desc = "Familiar with FAISS and vector databases from a course project on information retrieval."
    assert ownership_score(desc, _DOMAIN_KWS) < 0.25, (
        f"Expected < 0.25, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_designed_and_shipped_ab_scores_high():
    """'Designed and shipped 3 successive ranker variants, ran A/B testing' → high."""
    desc = (
        "Designed and shipped 3 successive ranker variants using LambdaMART, "
        "ran A/B testing on each and improved NDCG@10 by 12%."
    )
    assert ownership_score(desc, _DOMAIN_KWS) > 0.75, (
        f"Expected > 0.75, got {ownership_score(desc, _DOMAIN_KWS)}"
    )


def test_empty_string_returns_zero():
    """Empty input → exactly 0.0."""
    assert ownership_score("", _DOMAIN_KWS) == 0.0
    assert ownership_score("   ", _DOMAIN_KWS) == 0.0
