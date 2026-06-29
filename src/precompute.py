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

# JD text — verbatim from job_description.md (Redrob AI, Senior AI Engineer — Founding Team).
# Sourced directly from the competition data package; do not summarise or truncate.
JD_TEXT = """
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years (see "what we mean by this" below)

Let's be honest about this role
We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.
If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.
If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.
We need someone who is simultaneously comfortable with two things that sound contradictory:
1.	Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
2.	Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.
These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

What you'd actually be doing
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.
In practical terms, your first 90 days will probably look like:
•	Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
•	Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
•	Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.
Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

What we mean by "5-9 years"
This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.
That said, here are the disqualifiers we actually apply:
•	If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.
•	If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.
•	If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code.

The skills inventory (please read carefully)
Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.
Things you absolutely need
•	Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
•	Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
•	Strong Python. Yes really, we care about code quality.
•	Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
Things we'd like you to have but won't reject you for
•	LLM fine-tuning experience (LoRA, QLoRA, PEFT)
•	Experience with learning-to-rank models (XGBoost-based or neural)
•	Prior exposure to HR-tech, recruiting tech, or marketplace products
•	Background in distributed systems or large-scale inference optimization
•	Open-source contributions in the AI/ML space
Things we explicitly do NOT want
This is the section most JDs skip but we think it's the most important:
•	Title-chasers. If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.
•	Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.
•	People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.
•	People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.
•	People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.

On location, comp, and logistics
•	Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.
•	Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.

The vibe check
We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.
We work async-first and write a lot. If you find writing painful, you'll find this role painful.
We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.
We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.

How to read between the lines
The "ideal candidate" we're imagining is roughly:
•	6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).
•	Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
•	Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) — and can defend them with reference to systems they actually built.
•	Located in or willing to relocate to Noida or Pune.
•	Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.
We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that — we'd rather see 10 great matches than 1000 maybes.

Final note for the participants of the Redrob hackathon
If you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge:
The "right answer" to this JD is not "find candidates whose skills section contains the most AI keywords." That's a trap we've explicitly built into the dataset.
The right answer involves reasoning about the gap between what the JD says and what the JD means. A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks.
Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.
Good luck.
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
