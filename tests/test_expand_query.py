"""
Tests for src/expand_query.py.

All Claude API calls are mocked — no ANTHROPIC_API_KEY required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.expand_query import expand_query, _parse_profiles


# ── _parse_profiles ───────────────────────────────────────────────────────────

def test_parse_profiles_extracts_n_profiles():
    raw = (
        "1. I have shipped FAISS-based ranking at Swiggy, spending four years on dense retrieval.\n"
        "2. I am an NLP engineer with LambdaMART experience at a mid-stage product company.\n"
        "3. I built hybrid search at a FAANG lab before joining a Series-A startup."
    )
    profiles = _parse_profiles(raw, 3)
    assert len(profiles) == 3
    assert "FAISS" in profiles[0]
    assert "LambdaMART" in profiles[1]
    assert "FAANG" in profiles[2]


def test_parse_profiles_returns_fewer_than_n_gracefully():
    raw = "1. Only one profile here."
    profiles = _parse_profiles(raw, 3)
    assert len(profiles) == 1


def test_parse_profiles_strips_whitespace():
    raw = "1.  Profile with leading spaces.  \n2.  Another profile.  "
    profiles = _parse_profiles(raw, 2)
    assert all(p == p.strip() for p in profiles)


# ── expand_query ──────────────────────────────────────────────────────────────

def _fake_embed(texts: list[str]) -> np.ndarray:
    """Returns unit vectors (one per text) pointing in distinct directions."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(texts), 768)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _mock_claude_response(profiles: list[str]) -> MagicMock:
    """Build a minimal anthropic Message mock."""
    content_block = MagicMock()
    content_block.text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(profiles))
    message = MagicMock()
    message.content = [content_block]
    return message


_SAMPLE_PROFILES = [
    "I have shipped FAISS-based ranking at a product company with 6 years of experience.",
    "I am an NLP engineer with LambdaMART and recsys background at a startup.",
    "I built hybrid dense+BM25 search at scale with strong Python and evaluation tooling.",
    "I have production experience with Weaviate and sentence-transformers at a FAANG lab.",
    "I led the ranking team at an e-commerce platform, owning NDCG-driven A/B testing.",
]


@patch("src.expand_query._call_claude")
def test_expand_query_returns_correct_shape(mock_call):
    mock_call.return_value = _SAMPLE_PROFILES
    vec = expand_query("some jd text", _fake_embed, api_key="sk-test", verbose=False)
    assert vec.shape == (1, 768)
    assert vec.dtype == np.float32


@patch("src.expand_query._call_claude")
def test_expand_query_output_is_unit_norm(mock_call):
    mock_call.return_value = _SAMPLE_PROFILES
    vec = expand_query("some jd text", _fake_embed, api_key="sk-test", verbose=False)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


@patch("src.expand_query._call_claude")
def test_expand_query_averages_multiple_embeddings(mock_call):
    """With n=1 vs n=5 the resulting vectors should differ (different averages)."""
    mock_call.return_value = _SAMPLE_PROFILES[:1]
    vec1 = expand_query("jd", _fake_embed, n_profiles=1, api_key="sk-test", verbose=False)

    mock_call.return_value = _SAMPLE_PROFILES
    vec5 = expand_query("jd", _fake_embed, n_profiles=5, api_key="sk-test", verbose=False)

    # Cosine similarity between the two shouldn't be exactly 1.0
    cos = float(np.dot(vec1.squeeze(), vec5.squeeze()))
    assert cos < 0.9999, "n=1 and n=5 vectors should not be identical"


@patch("src.expand_query._call_claude")
def test_expand_query_passes_n_profiles_to_claude(mock_call):
    mock_call.return_value = _SAMPLE_PROFILES[:3]
    expand_query("jd", _fake_embed, n_profiles=3, api_key="sk-test", verbose=False)
    args, kwargs = mock_call.call_args
    assert args[1] == 3  # second positional arg to _call_claude is n


def test_expand_query_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        expand_query("jd", _fake_embed, api_key=None, verbose=False)


@patch("src.expand_query._call_claude")
def test_expand_query_raises_on_empty_profiles(mock_call):
    mock_call.return_value = []
    with pytest.raises(ValueError, match="zero profiles"):
        expand_query("jd", _fake_embed, n_profiles=3, api_key="sk-test", verbose=False)
