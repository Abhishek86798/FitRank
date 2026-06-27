import json
from pathlib import Path

import pytest

from src.data_loader import build_candidate_text, stream_candidates

SAMPLE_JSON = Path("data/sample_candidates.json")
FULL_JSONL = Path("data/candidates.jsonl")


@pytest.fixture(scope="module")
def sample_candidates():
    return json.loads(SAMPLE_JSON.read_bytes())


def test_sample_candidates_load(sample_candidates):
    assert len(sample_candidates) > 0


def test_build_candidate_text_nonempty(sample_candidates):
    for c in sample_candidates:
        text = build_candidate_text(c)
        assert isinstance(text, str)
        assert len(text) > 0


def test_build_candidate_text_capped(sample_candidates):
    for c in sample_candidates:
        text = build_candidate_text(c)
        assert len(text) <= 3000


def test_stream_candidates_full_jsonl():
    if not FULL_JSONL.exists():
        pytest.skip("candidates.jsonl not present")
    total = 0
    for batch in stream_candidates(FULL_JSONL, batch_size=100):
        assert len(batch) <= 100
        total += len(batch)
    assert total > 0
