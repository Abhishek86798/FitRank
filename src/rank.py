# ENTRYPOINT — the single file that runs inside the competition sandbox
# (CPU-only, no network, <5 min).
# Orchestrates: load artifacts → dense retrieve → collect records →
#               build features → score → reason → write submission.csv
#
# Cascade pipeline (v2):
#   Stage 1: hybrid retrieve (dense + BM25 RRF) → pool up to pool_size candidates
#   Stage 2: fast_filter — cheap rule-based gates (skip with --no-cascade)
#   Stage 3: build_feature_vector + LTR score → top submission_size

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


# ── tie-break helper (validator requires: equal scores → candidate_id ascending) ──

def _assign_ranks(scored: list[tuple[str, float]]) -> list[tuple[int, str, float]]:
    """
    Given [(candidate_id, score), ...] sorted best-first, assign ranks 1..N.
    For equal scores, candidate_id ascending breaks ties (validator rule).
    """
    # Group by score to handle ties
    result: list[tuple[int, str, float]] = []
    i = 0
    rank = 1
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][1] == scored[i][1]:
            j += 1
        # Tie group [i, j) — sort by candidate_id ascending
        group = sorted(scored[i:j], key=lambda x: x[0])
        for cid, score in group:
            result.append((rank, cid, score))
            rank += 1
        i = j
    return result


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    artifacts_dir: Path,
    candidates_path: Path,
    role_model_path: Path,
    output_path: Path,
    top_k: int = 300,
    pool_size: int = 500,
    submission_size: int = 100,
    prefix: str = "",
    no_cascade: bool = False,
) -> None:
    import numpy as np
    import yaml

    from src.data_loader import stream_candidates, build_candidate_text
    from src.retriever import retrieve_top_k, BM25Retriever, reciprocal_rank_fusion
    from src.feature_builder import build_feature_vector
    from src.scorer import LTRScorer, score_with_weighted_sum
    from src.reasoning import compose_reasoning, compose_reasoning_with_citations
    from src.fast_filter import fast_filter

    # ── 1. Load role model ────────────────────────────────────────────────────
    print(f"Loading role model from {role_model_path} ...")
    with open(role_model_path, encoding="utf-8") as f:
        role_model: dict = yaml.safe_load(f)

    # ── 2. Load artifacts ─────────────────────────────────────────────────────
    # Auto-detect prefix if not supplied: prefer bare (full run), fall back to sample_
    def _find_artifact(name: str) -> Path:
        bare   = artifacts_dir / name
        sample = artifacts_dir / f"sample_{name}"
        if prefix:
            explicit = artifacts_dir / f"{prefix}{name}"
            if explicit.exists():
                return explicit
        if bare.exists():
            return bare
        if sample.exists():
            print(f"  [warn] {name} not found, using sample_{name}")
            return sample
        raise FileNotFoundError(f"Artifact not found: {name} (tried {bare} and {sample})")

    emb_path  = _find_artifact("embeddings.npy")
    ids_path  = _find_artifact("candidate_ids.npy")
    jd_path   = artifacts_dir / "jd_vector.npy"
    ltr_path  = artifacts_dir / "ltr_model.txt"

    print(f"Loading embeddings from {emb_path} ...")
    embeddings    = np.load(emb_path).astype(np.float32)
    candidate_ids = np.load(ids_path, allow_pickle=True)
    jd_vector     = np.load(jd_path).astype(np.float32)

    print(f"  embeddings shape={embeddings.shape}  ids={len(candidate_ids)}")

    # ── 3. Stage 1: Dense retrieval → top-K candidate IDs ────────────────────
    t0_stage1 = time.perf_counter()
    fetch_k = min(top_k, len(candidate_ids))
    print(f"Stage 1 — dense retrieval: top-{fetch_k} ...")
    dense_ids, dense_scores = retrieve_top_k(embeddings, jd_vector, candidate_ids, k=fetch_k)
    dense_id_set = set(dense_ids)

    # ── 4. Stream candidates — collect dense top-K records + BM25 texts ─────
    print(f"Streaming {candidates_path} to collect dense records ...")
    from src.precompute import JD_TEXT

    top_records: dict[str, dict] = {}   # id → full record
    bm25_ids_list: list[str]     = []   # all IDs in file order, for BM25
    bm25_texts: list[str]        = []   # embeddable text per candidate

    def _iter_candidates(path: Path):
        """Yield individual candidate dicts from .json array or .jsonl."""
        if path.suffix.lower() == ".json":
            import json
            for cand in json.loads(path.read_bytes()):
                yield cand
        else:
            for batch in stream_candidates(path):
                yield from batch

    # Stream once: build BM25 corpus AND collect dense top-K records.
    # BM25 needs all texts, so we cannot break early.
    for cand in _iter_candidates(candidates_path):
        cid  = cand["candidate_id"]
        text = build_candidate_text(cand)
        bm25_ids_list.append(cid)
        bm25_texts.append(text)
        if cid in dense_id_set and cid not in top_records:
            top_records[cid] = cand

    print(f"  Collected {len(top_records)} records for dense top-{fetch_k}")

    # ── 5. BM25 retrieval + RRF fusion → pool up to pool_size ────────────────
    print("Building BM25 index and retrieving ...")
    skill_clusters = role_model.get("skill_clusters", {})
    bm25 = BM25Retriever(bm25_ids_list, bm25_texts, skill_clusters=skill_clusters)
    bm25_ids, _ = bm25.retrieve_top_k(JD_TEXT, k=fetch_k)

    # RRF: merge dense + BM25; pool_size caps the fused set
    print(f"Fusing dense + BM25 via RRF (pool_size={pool_size}) ...")
    merged_ids = reciprocal_rank_fusion([dense_ids, list(bm25_ids)], k_rrf=60)
    merged_ids = merged_ids[:pool_size]

    # Collect any BM25-only candidates not already in top_records
    bm25_id_set = set(bm25_ids[:fetch_k])
    missing = bm25_id_set - dense_id_set
    missing &= set(merged_ids)  # only fetch what's in the pool
    if missing:
        print(f"  Collecting {len(missing)} BM25-only candidates ...")
        for cand in _iter_candidates(candidates_path):
            if cand["candidate_id"] in missing:
                top_records[cand["candidate_id"]] = cand
            if missing <= top_records.keys():
                break

    t1_stage1 = time.perf_counter()
    n_pool = sum(1 for cid in merged_ids if cid in top_records)
    print(f"Stage 1 retrieval: {n_pool} candidates in {t1_stage1 - t0_stage1:.2f}s")

    # ── Stage 2: fast rule-based filter ──────────────────────────────────────
    if no_cascade:
        pool_records = [top_records[cid] for cid in merged_ids if cid in top_records]
        pool_cids    = [cand["candidate_id"] for cand in pool_records]
        print(f"Stage 2 filter: skipped (--no-cascade), {len(pool_records)} candidates pass")
    else:
        t0_stage2 = time.perf_counter()
        pool_records_in = [top_records[cid] for cid in merged_ids if cid in top_records]
        pool_records, drop_log = fast_filter(pool_records_in, role_model)
        pool_cids = [cand["candidate_id"] for cand in pool_records]
        t1_stage2 = time.perf_counter()
        print(
            f"Stage 2 filter: {len(pool_records_in)} -> {len(pool_records)} "
            f"in {t1_stage2 - t0_stage2:.2f}s  drops={drop_log}"
        )

    # ── 6. Stage 3: Score and rank ────────────────────────────────────────────
    t0_stage3 = time.perf_counter()
    print("Stage 3 — scoring candidates ...")
    scorer = LTRScorer(ltr_path, role_model)
    print(f"  Scorer mode: {'LambdaMART' if scorer.is_ltr else 'weighted-sum'}")

    # Build feature vectors for all pool candidates, score in one batch.
    dense_idx = {cid: i for i, cid in enumerate(dense_ids)}

    cids_to_score: list[str] = []
    feature_batch: list[dict] = []
    for cand in pool_records:
        cid = cand["candidate_id"]
        cosine = float(dense_scores[dense_idx[cid]]) if cid in dense_idx else 0.0
        cids_to_score.append(cid)
        feature_batch.append(build_feature_vector(cand, role_model, cosine_sim=cosine))

    batch_scores = scorer.score_batch(feature_batch)
    scored: list[tuple[str, float]] = list(zip(cids_to_score, batch_scores))

    # Sort best-first; pad with remaining pool candidates if fewer than submission_size
    scored.sort(key=lambda x: (-x[1], x[0]))   # score desc, id asc for equal scores

    # Pad to submission_size with remaining pool candidates not yet scored
    scored_ids = {cid for cid, _ in scored}
    pad_cids: list[str] = []
    pad_feats: list[dict] = []
    for cid in pool_cids:
        if len(scored) + len(pad_cids) >= submission_size:
            break
        if cid in scored_ids:
            continue
        cand = top_records.get(cid)
        if cand is None:
            continue
        pad_cids.append(cid)
        pad_feats.append(build_feature_vector(cand, role_model, cosine_sim=0.0))

    if pad_cids:
        pad_scores = scorer.score_batch(pad_feats)
        scored.extend(zip(pad_cids, pad_scores))

    scored.sort(key=lambda x: (-x[1], x[0]))
    scored = scored[:submission_size]

    t1_stage3 = time.perf_counter()
    print(f"Stage 3 LTR: {len(cids_to_score)} scored in {t1_stage3 - t0_stage3:.2f}s")

    # ── 7. Assign ranks (with tie-break) ──────────────────────────────────────
    ranked = _assign_ranks(scored)   # [(rank, cid, score), ...]

    # ── 8. Build reasoning strings + citations ───────────────────────────────
    print("Generating reasoning strings ...")
    import json as _json
    rows: list[dict] = []
    citations_map: dict[str, dict] = {}
    for rank, cid, score in ranked:
        cand     = top_records[cid]
        cosine   = float(dense_scores[dense_idx[cid]]) if cid in dense_idx else 0.0
        features  = build_feature_vector(cand, role_model, cosine_sim=cosine)
        reasoning = compose_reasoning(cand, features, rank)
        cited     = compose_reasoning_with_citations(cand, features, rank)
        citations_map[cid] = {
            "citations":       cited["citations"],
            "ungrounded_count": cited["ungrounded_count"],
            "total_claims":    cited["total_claims"],
        }
        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score,
            "reasoning":    reasoning,
        })

    # Write citations artifact alongside the submission
    citations_path = artifacts_dir / "citations.json"
    citations_path.write_text(_json.dumps(citations_map, indent=2), encoding="utf-8")
    print(f"Wrote citations to {citations_path}")

    # ── 9. Write submission.csv ───────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_path} ({len(rows)} rows) ...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Top-5 preview:")
    for row in rows[:5]:
        print(f"  #{row['rank']:3d}  [{row['score']:.4f}]  {row['candidate_id']}")
        print(f"         {row['reasoning'][:100]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FitRank full pipeline — writes submission.csv")
    parser.add_argument("--artifacts-dir",   default="artifacts",            help="Artifacts directory")
    parser.add_argument("--candidates",      default="data/candidates.jsonl", help="Candidate data file (.jsonl or .json)")
    parser.add_argument("--role-model",      default="role_model.yaml",      help="Role model YAML")
    parser.add_argument("--output",          default="submission.csv",       help="Output submission CSV path")
    parser.add_argument("--top-k",           type=int, default=300,          help="Dense + BM25 retrieval depth (default 300 each)")
    parser.add_argument("--pool-size",       type=int, default=500,          help="Max RRF-fused candidates entering Stage 2/3 (default 500)")
    parser.add_argument("--submission-size", type=int, default=100,          help="Rows in output CSV (must be 100 for validator)")
    parser.add_argument("--prefix",          default="",                     help="Artifact filename prefix (e.g. 'sample_')")
    parser.add_argument("--no-cascade",      action="store_true",            help="Skip Stage 2 fast_filter (ablation mode)")
    args = parser.parse_args()

    run(
        artifacts_dir    = Path(args.artifacts_dir),
        candidates_path  = Path(args.candidates),
        role_model_path  = Path(args.role_model),
        output_path      = Path(args.output),
        top_k            = args.top_k,
        pool_size        = args.pool_size,
        submission_size  = args.submission_size,
        prefix           = args.prefix,
        no_cascade       = args.no_cascade,
    )


if __name__ == "__main__":
    main()
