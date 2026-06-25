# OFFLINE script — runs once before the sandboxed ranking step.
# Encodes all candidates and the JD with BGE-base-en-v1.5 and saves
# embeddings.npy, candidate_ids.npy, and jd_vector.npy to artifacts/.

from __future__ import annotations

import argparse
import json
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


def _load_model():
    from sentence_transformers import SentenceTransformer
    print("Loading BAAI/bge-base-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return model


def embed_texts(model, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Encode texts with BGE, L2-normalised. Returns float32 array."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # mandatory — dot product == cosine similarity
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def run(candidates_path: Path, artifacts_dir: Path, prefix: str = "") -> None:
    """
    Encode all candidates from candidates_path and save artifacts.

    prefix  — filename prefix, e.g. "sample_" for dev runs, "" for full run.
    """
    from src.data_loader import build_candidate_text

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load candidates ────────────────────────────────────────────────────────
    suffix = candidates_path.suffix.lower()
    if suffix == ".json":
        candidates = json.loads(candidates_path.read_bytes())
    elif suffix == ".jsonl":
        import orjson
        candidates = []
        with open(candidates_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(orjson.loads(line))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    print(f"Loaded {len(candidates)} candidates from {candidates_path.name}")

    # ── Build embeddable texts ─────────────────────────────────────────────────
    ids = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]

    # ── Load model ─────────────────────────────────────────────────────────────
    model = _load_model()

    # ── Embed candidates ───────────────────────────────────────────────────────
    print(f"Encoding {len(texts)} candidate texts ...")
    embeddings = embed_texts(model, texts, batch_size=64)          # (n, 768) float32
    embeddings_fp16 = embeddings.astype(np.float16)

    emb_path = artifacts_dir / f"{prefix}embeddings.npy"
    ids_path = artifacts_dir / f"{prefix}candidate_ids.npy"
    np.save(emb_path, embeddings_fp16)
    np.save(ids_path, np.array(ids, dtype=object))
    print(f"Saved {emb_path}  shape={embeddings_fp16.shape}  "
          f"size={emb_path.stat().st_size / 1e6:.1f} MB")

    # ── Embed JD (shared across sample and full runs) ──────────────────────────
    jd_path = artifacts_dir / "jd_vector.npy"
    if not jd_path.exists():
        print("Encoding JD text with query prefix ...")
        jd_vec = embed_texts(model, [BGE_QUERY_PREFIX + JD_TEXT], show_progress=False)
        np.save(jd_path, jd_vec.astype(np.float16))
        print(f"Saved {jd_path}  shape={jd_vec.shape}")
    else:
        print(f"JD vector already exists at {jd_path} — skipping.")


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
