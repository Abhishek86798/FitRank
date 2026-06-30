# OFFLINE script — runs once on a Colab GPU runtime, NOT in the competition
# sandbox. The sandbox is CPU-only and has no network access, so this script
# cannot run there; it must be executed separately and its output
# (artifacts/ce_scores.json) committed/uploaded alongside the other
# precomputed artifacts (embeddings.npy, candidate_ids.npy, jd_vector.npy).
#
# What it does: cross-encodes the JD against every candidate's text with a
# sentence-transformers CrossEncoder, producing one raw logit per candidate.
# feature_builder.py::_cross_encoder_score applies a sigmoid to this raw
# logit at scoring time — this script must NOT apply sigmoid itself, it
# saves raw logits only.
#
# Output: artifacts/ce_scores.json — {candidate_id: raw_logit, ...}
#
# Why Colab / GPU: cross-encoders score each (query, passage) pair through a
# full transformer forward pass (no caching like bi-encoders), which is too
# slow for 100k candidates on CPU. A T4 GPU does the full corpus in minutes.
#
# Regenerate with (run on Colab, then download ce_scores.json into artifacts/):
#   python scripts/precompute_cross_encoder.py \
#       --candidates data/candidates.jsonl \
#       --artifacts-dir artifacts

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Same JD text used by src/precompute.py for the dense-embedding JD vector —
# kept in sync so the cross-encoder and bi-encoder score against identical
# query text. Sourced verbatim from the competition data package.
from src.precompute import JD_TEXT  # noqa: E402


def _stream_records(candidates_path: Path):
    suffix = candidates_path.suffix.lower()
    if suffix == ".json":
        for cand in json.loads(candidates_path.read_bytes()):
            yield cand
    elif suffix == ".jsonl":
        import orjson
        with open(candidates_path, "rb") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    yield orjson.loads(raw)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def run(candidates_path: Path, artifacts_dir: Path, batch_size: int = 64) -> None:
    from sentence_transformers import CrossEncoder
    from src.data_loader import build_candidate_text

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    model = CrossEncoder(MODEL_NAME)  # auto-selects CUDA if available

    ids: list[str] = []
    texts: list[str] = []
    for cand in _stream_records(candidates_path):
        ids.append(cand["candidate_id"])
        texts.append(build_candidate_text(cand))

    print(f"Scoring {len(ids)} candidates against JD ...", flush=True)
    t0 = time.perf_counter()
    pairs = [(JD_TEXT, text) for text in texts]
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed/60:.1f} min ({len(ids)/elapsed:.0f} cands/s)", flush=True)

    ce_scores = {cid: float(score) for cid, score in zip(ids, raw_scores)}

    out_path = artifacts_dir / "ce_scores.json"
    out_path.write_text(json.dumps(ce_scores), encoding="utf-8")
    print(f"Saved {out_path}  ({len(ce_scores)} entries)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute cross-encoder scores (Colab GPU, offline)."
    )
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run(
        candidates_path=Path(args.candidates),
        artifacts_dir=Path(args.artifacts_dir),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
