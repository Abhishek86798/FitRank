"""
Sandbox constraint tests for the full FitRank pipeline.

Checks:
  1. embeddings.npy exists and has shape (n_candidates, 768)
  2. rank.py end-to-end on full candidates.jsonl runs under 5 min, under 16 GB RAM
  3. Output CSV has exactly 100 rows and passes column validation
  4. BM25 + dense retrieval both return results
  5. precompute streaming path (unit test — does not re-encode, checks ID/shape alignment)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
CANDIDATES_JSONL = DATA / "candidates.jsonl"
SUBMISSION_OUT = ROOT / "submission_full_test.csv"

sys.path.insert(0, str(ROOT))

# ── helpers ───────────────────────────────────────────────────────────────────

def _peak_rss_mb() -> float:
    """Return current process RSS in MB (Windows-safe)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


# ── fixture: skip if full embeddings not yet precomputed ─────────────────────

requires_full_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "embeddings.npy").exists(),
    reason="Full embeddings not precomputed yet — run: python -m src.precompute --candidates data/candidates.jsonl --artifacts-dir artifacts",
)

requires_full_data = pytest.mark.skipif(
    not CANDIDATES_JSONL.exists(),
    reason="data/candidates.jsonl not found",
)


# ── 1. Embedding artifact shape ───────────────────────────────────────────────

@requires_full_artifacts
def test_embeddings_shape():
    emb = np.load(ARTIFACTS / "embeddings.npy")
    ids = np.load(ARTIFACTS / "candidate_ids.npy", allow_pickle=True)
    assert emb.ndim == 2, f"Expected 2D array, got shape {emb.shape}"
    assert emb.shape[1] == 768, f"Expected 768-dim embeddings, got {emb.shape[1]}"
    assert emb.shape[0] == len(ids), "Embeddings and IDs length mismatch"
    assert emb.shape[0] > 0, "Empty embeddings array"
    print(f"\n  embeddings shape: {emb.shape}  dtype: {emb.dtype}")


@requires_full_artifacts
def test_embeddings_normalized():
    emb = np.load(ARTIFACTS / "embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    # fp16 rounding can push norms slightly off; allow 2% tolerance
    bad = np.sum(np.abs(norms - 1.0) > 0.02)
    assert bad == 0, f"{bad} embeddings are not L2-normalised (norms min={norms.min():.4f} max={norms.max():.4f})"


# ── 2. Full pipeline runtime and RAM ─────────────────────────────────────────

@requires_full_artifacts
@requires_full_data
def test_full_pipeline_runtime_and_ram():
    """End-to-end rank.py must finish in <5 min and stay under 16 GB RSS."""
    from src.rank import run

    ram_before = _peak_rss_mb()
    t0 = time.perf_counter()

    run(
        artifacts_dir=ARTIFACTS,
        candidates_path=CANDIDATES_JSONL,
        role_model_path=ROOT / "role_model.yaml",
        output_path=SUBMISSION_OUT,
        top_k=100,
        submission_size=100,
    )

    elapsed = time.perf_counter() - t0
    ram_after = _peak_rss_mb()
    peak_ram = ram_after - ram_before

    print(f"\n  Wall time: {elapsed:.1f}s  Peak RAM delta: {peak_ram:.0f} MB")

    MAX_SECONDS = 300   # 5 minutes
    MAX_RAM_MB  = 16_000  # 16 GB

    assert elapsed < MAX_SECONDS, (
        f"Pipeline took {elapsed:.1f}s — exceeds {MAX_SECONDS}s sandbox limit"
    )
    # RAM check is best-effort (psutil may not be installed)
    if peak_ram > 0:
        assert peak_ram < MAX_RAM_MB, (
            f"RAM delta {peak_ram:.0f} MB exceeds {MAX_RAM_MB} MB sandbox limit"
        )


# ── 3. Output CSV validation ──────────────────────────────────────────────────

@requires_full_artifacts
@requires_full_data
def test_output_csv_schema():
    """submission CSV must have 100 rows with required columns and valid ranks."""
    if not SUBMISSION_OUT.exists():
        pytest.skip("Run test_full_pipeline_runtime_and_ram first to generate output")

    with open(SUBMISSION_OUT, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    actual_cols = set(rows[0].keys())
    assert required_cols <= actual_cols, f"Missing columns: {required_cols - actual_cols}"

    ranks = [int(r["rank"]) for r in rows]
    assert sorted(ranks) == list(range(1, 101)), "Ranks must be 1..100 with no gaps or duplicates"

    for r in rows:
        assert r["candidate_id"].startswith("CAND_"), f"Bad candidate_id: {r['candidate_id']}"
        assert r["reasoning"].strip(), "Empty reasoning string"


# ── 4. BM25 retrieval sanity ──────────────────────────────────────────────────

@requires_full_data
def test_bm25_returns_results():
    from src.data_loader import stream_candidates, build_candidate_text
    from src.retriever import BM25Retriever
    from src.precompute import JD_TEXT

    ids, texts = [], []
    for batch in stream_candidates(CANDIDATES_JSONL):
        for cand in batch:
            ids.append(cand["candidate_id"])
            texts.append(build_candidate_text(cand))
        if len(ids) >= 1000:
            break

    bm25 = BM25Retriever(ids, texts)
    top_ids, scores = bm25.retrieve_top_k(JD_TEXT, k=10)

    assert len(top_ids) == 10
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
        "BM25 scores not sorted descending"
    print(f"\n  BM25 top-1: {top_ids[0]}  score={scores[0]:.4f}")


# ── 5. Dense retrieval sanity ─────────────────────────────────────────────────

@requires_full_artifacts
def test_dense_retrieval_returns_results():
    from src.retriever import retrieve_top_k

    emb = np.load(ARTIFACTS / "embeddings.npy").astype(np.float32)
    ids = np.load(ARTIFACTS / "candidate_ids.npy", allow_pickle=True)
    jd  = np.load(ARTIFACTS / "jd_vector.npy").astype(np.float32)

    top_ids, scores = retrieve_top_k(emb, jd, ids, k=50)

    assert len(top_ids) == 50
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), \
        "Dense scores not sorted descending"
    print(f"\n  Dense top-1: {top_ids[0]}  cosine={scores[0]:.4f}")


# ── 6. ID/shape alignment after precompute ───────────────────────────────────

@requires_full_artifacts
def test_candidate_ids_unique():
    ids = np.load(ARTIFACTS / "candidate_ids.npy", allow_pickle=True)
    assert len(ids) == len(set(ids)), "Duplicate candidate IDs in candidate_ids.npy"
