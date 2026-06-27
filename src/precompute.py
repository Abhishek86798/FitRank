# OFFLINE script — runs once before the sandboxed ranking step.
# Encodes all candidates and the JD with BGE-base-en-v1.5 and saves
# embeddings.npy, candidate_ids.npy, and jd_vector.npy to artifacts/.
#
# RAM budget (8 GB machine, no GPU):
#   Model load : ~400 MB
#   Per batch  : ~30 MB at ENCODE_BATCH=32
#   memmap file: mapped to disk — never fully in RAM
#   Peak total : <2 GB

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# JD text — sourced from job_description.docx, query-prefix added at encode time.
JD_TEXT = """
Senior AI Engineer — Founding Team. Redrob AI, Series A AI-native talent intelligence platform.
Pune/Noida, India (Hybrid). 5-9 years experience.

Own the intelligence layer: ranking, retrieval, and matching systems.
Ship v2 ranking system using embeddings, hybrid retrieval, LLM-based re-ranking.
Set up evaluation infrastructure: offline benchmarks, A/B testing, recruiter-feedback loops.

Absolute requirements:
Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5).
Production experience with vector databases or hybrid search: Pinecone, Weaviate, Qdrant, Milvus,
OpenSearch, Elasticsearch, FAISS.
Strong Python. Hands-on experience designing evaluation frameworks for ranking systems:
NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost, LambdaMART),
HR-tech experience, distributed systems, open-source AI/ML contributions.

Not a fit: pure consulting career (TCS, Infosys, Wipro, Accenture), non-engineer titles
(marketing manager, HR manager), pure research without production deployment,
computer vision / speech / robotics without NLP/IR background.

Ideal: 6-8 years, 4-5 in applied ML at product companies, shipped end-to-end ranking /
search / recommendation system to real users at scale, strong opinions on retrieval
(hybrid vs dense), evaluation (offline vs online), LLM integration.
Location: Pune, Noida, Delhi NCR, Hyderabad, Mumbai preferred.
Notice period: sub-30 days preferred, up to 30 days buyout available.
""".strip()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ENCODE_BATCH = 64   # 64 texts × 768-dim fp16 ≈ 75 MB working memory; sweet spot for CPU throughput

# i5-12500H: 6P + 4E cores, 16 logical threads.
# Sentence-transformers/PyTorch defaults to spawning all logical threads per batch,
# causing contention when model inference already uses multiple ops internally.
# Capping at physical core count gives ~2-3x throughput improvement on this CPU.
_NUM_THREADS = 6


def _set_threads():
    """Cap PyTorch + NumPy thread counts to avoid inter-op contention."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_THREADS))
    try:
        import torch
        torch.set_num_threads(_NUM_THREADS)
        torch.set_num_interop_threads(2)
    except Exception:
        pass


def _load_model():
    _set_threads()
    from sentence_transformers import SentenceTransformer
    print("Loading BAAI/bge-base-en-v1.5 ...", flush=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return model


def embed_texts(model, texts: list[str], batch_size: int = ENCODE_BATCH, show_progress: bool = True) -> np.ndarray:
    """Encode texts with BGE, L2-normalised. Returns float32 array."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # mandatory — dot product == cosine similarity
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def _count_lines(path: Path) -> int:
    """Fast line count via raw byte scan."""
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def _stream_records(candidates_path: Path):
    """Yield candidate dicts one by one from .json array or .jsonl."""
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


def run(candidates_path: Path, artifacts_dir: Path, prefix: str = "") -> None:
    """
    Encode all candidates from candidates_path and save artifacts.

    Uses np.memmap to write embeddings directly to disk batch-by-batch so
    total RAM stays under 2 GB even on an 8 GB machine.

    prefix — filename prefix, e.g. "sample_" for dev runs, "" for full run.
    """
    from src.data_loader import build_candidate_text

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Count candidates so we can pre-allocate the memmap ────────────────
    suffix = candidates_path.suffix.lower()
    if suffix == ".jsonl":
        print(f"Counting lines in {candidates_path.name} ...", flush=True)
        n_candidates = _count_lines(candidates_path)
        print(f"  {n_candidates} candidates", flush=True)
    else:
        # JSON array: load to count, then re-stream
        _all = json.loads(candidates_path.read_bytes())
        n_candidates = len(_all)
        del _all
        print(f"  {n_candidates} candidates (JSON)", flush=True)

    # ── 2. Pre-allocate memmap on disk — never held fully in RAM ─────────────
    emb_path = artifacts_dir / f"{prefix}embeddings.npy"
    ids_path = artifacts_dir / f"{prefix}candidate_ids.npy"

    # np.save on a memmap writes the .npy header then the data in-place.
    # We create the memmap as a raw buffer; np.save wraps with header at end.
    # Simpler: write via memmap directly, then save IDs normally.
    # Shape: (n_candidates, 768), dtype fp16
    emb_mm = np.lib.format.open_memmap(
        str(emb_path),
        mode="w+",
        dtype=np.float16,
        shape=(n_candidates, 768),
    )
    print(f"Pre-allocated {emb_path.name}  shape={emb_mm.shape}  "
          f"({emb_path.stat().st_size / 1e6:.0f} MB on disk)", flush=True)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    model = _load_model()

    # ── 4. Stream → encode → write to memmap batch by batch ──────────────────
    ids: list[str] = []
    text_buf: list[str] = []
    id_buf: list[str] = []
    write_idx = 0
    n_done = 0
    t0 = time.perf_counter()

    def _flush_batch():
        nonlocal write_idx, n_done
        vecs = embed_texts(model, text_buf, batch_size=len(text_buf), show_progress=False)
        fp16 = vecs.astype(np.float16)
        emb_mm[write_idx : write_idx + len(fp16)] = fp16
        ids.extend(id_buf)
        write_idx += len(fp16)
        n_done += len(fp16)

        if n_done % 2000 == 0 or n_done == n_candidates:
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (n_candidates - n_done) / rate if rate > 0 else 0
            print(
                f"  [{n_done}/{n_candidates}]  "
                f"{elapsed/60:.1f}min elapsed  "
                f"ETA {eta/60:.1f}min  "
                f"({rate:.0f} cands/s)",
                flush=True,
            )

    for cand in _stream_records(candidates_path):
        id_buf.append(cand["candidate_id"])
        text_buf.append(build_candidate_text(cand))
        if len(text_buf) == ENCODE_BATCH:
            _flush_batch()
            text_buf = []
            id_buf = []

    if text_buf:
        _flush_batch()

    del emb_mm  # flush OS write buffer

    # ── 5. Save candidate IDs ─────────────────────────────────────────────────
    np.save(ids_path, np.array(ids, dtype=object))
    print(f"Saved {ids_path.name}  ({len(ids)} IDs)", flush=True)
    print(f"Saved {emb_path.name}  size={emb_path.stat().st_size / 1e6:.0f} MB", flush=True)

    total = time.perf_counter() - t0
    print(f"Encoding done in {total/60:.1f} min  ({n_done/total:.0f} cands/s)", flush=True)

    # ── 6. Build JD query vector (shared across sample and full runs) ───────────
    # Default: persona-based query expansion via Claude (offline, ~5 API calls).
    # Falls back to raw-JD embedding when ANTHROPIC_API_KEY is not set.
    jd_path = artifacts_dir / "jd_vector.npy"
    if not jd_path.exists():
        import os
        from src.expand_query import expand_query

        def _embed_for_expansion(texts: list[str]) -> np.ndarray:
            prefixed = [BGE_QUERY_PREFIX + t for t in texts]
            return embed_texts(model, prefixed, show_progress=False)

        if os.environ.get("ANTHROPIC_API_KEY"):
            jd_vec = expand_query(JD_TEXT, _embed_for_expansion, verbose=True)
        else:
            print(
                "ANTHROPIC_API_KEY not set — falling back to raw-JD embedding "
                "(set the key and delete jd_vector.npy to use persona expansion).",
                flush=True,
            )
            jd_vec = embed_texts(model, [BGE_QUERY_PREFIX + JD_TEXT], show_progress=False)

        np.save(jd_path, jd_vec.astype(np.float16))
        print(f"Saved {jd_path.name}  shape={jd_vec.shape}", flush=True)
    else:
        print(f"JD vector already exists at {jd_path} — skipping.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings.")
    parser.add_argument(
        "--candidates",
        default="data/sample_candidates.json",
        help="Path to candidates file (.json array or .jsonl)",
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--prefix", default="",
        help='Filename prefix for output files (e.g. "sample_" for dev runs)',
    )
    args = parser.parse_args()

    run(
        candidates_path=Path(args.candidates),
        artifacts_dir=Path(args.artifacts_dir),
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
