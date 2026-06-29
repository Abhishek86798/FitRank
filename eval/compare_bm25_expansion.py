"""
Compare BM25 retrieval pool before vs after skill-cluster query expansion.

Usage:
    python eval/compare_bm25_expansion.py

Prints:
  - Which candidates appear only after expansion (newly surfaced)
  - Top-20 IDs before/after
  - NDCG@10 (unchanged, since LTR re-scores the same pool)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import BM25Retriever, expand_query_with_clusters
from src.precompute import JD_TEXT

ROLE_MODEL_PATH = ROOT / "role_model.yaml"
CANDIDATES_PATH = ROOT / "data" / "candidates.jsonl"
SAMPLE_PATH     = ROOT / "data" / "sample_candidates.json"
SUBMISSION_CSV  = ROOT / "team_xxx.csv"

K = 300  # match rank.py default


def _load_role_model() -> dict:
    with open(ROLE_MODEL_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_candidates(path: Path):
    if path.suffix == ".json":
        for c in json.loads(path.read_bytes()):
            yield c
    else:
        try:
            import orjson
            with open(path, "rb") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield orjson.loads(line)
        except ImportError:
            import json as _json
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield _json.loads(line)


def _build_corpus(candidates_path: Path) -> tuple[list[str], list[str]]:
    from src.data_loader import build_candidate_text
    ids, texts = [], []
    for cand in _iter_candidates(candidates_path):
        ids.append(cand["candidate_id"])
        texts.append(build_candidate_text(cand))
    return ids, texts


def main() -> None:
    role_model     = _load_role_model()
    skill_clusters = role_model.get("skill_clusters", {})

    # Choose data file
    data_path = CANDIDATES_PATH if CANDIDATES_PATH.exists() else SAMPLE_PATH
    print(f"Loading candidates from {data_path} …")
    ids, texts = _build_corpus(data_path)
    print(f"  Corpus size: {len(ids)} candidates")

    # Show what the expansion does to the JD query
    expanded = expand_query_with_clusters(JD_TEXT, skill_clusters)
    original_tokens = set(JD_TEXT.lower().split())
    expanded_tokens = set(expanded.lower().split())
    new_tokens = sorted(expanded_tokens - original_tokens)
    print(f"\nQuery expansion: {len(original_tokens)} -> {len(expanded_tokens)} tokens")
    print(f"  New terms added ({len(new_tokens)}): {new_tokens[:20]}")

    # BM25 without expansion
    bm25_plain = BM25Retriever(ids, texts, skill_clusters=None)
    plain_ids, _ = bm25_plain.retrieve_top_k(JD_TEXT, k=K)
    plain_set = set(plain_ids)

    # BM25 with expansion
    bm25_expanded = BM25Retriever(ids, texts, skill_clusters=skill_clusters)
    exp_ids, _ = bm25_expanded.retrieve_top_k(JD_TEXT, k=K)
    exp_set = set(exp_ids)

    # Diff
    newly_surfaced = [cid for cid in exp_ids if cid not in plain_set]
    dropped        = [cid for cid in plain_ids if cid not in exp_set]

    print(f"\nTop-{K} pool changes after expansion:")
    print(f"  Newly surfaced (in expanded, not in plain): {len(newly_surfaced)}")
    print(f"  Dropped       (in plain, not in expanded):  {len(dropped)}")

    if newly_surfaced:
        print(f"\n  New candidates (first 10): {newly_surfaced[:10]}")
    if dropped:
        print(f"  Dropped candidates (first 5): {dropped[:5]}")

    # Top-20 comparison
    print(f"\nTop-20 BM25 IDs — PLAIN vs EXPANDED:")
    print(f"  {'Rank':<5}  {'Plain':<16}  {'Expanded':<16}  {'Same?'}")
    print(f"  {'-'*52}")
    for i in range(20):
        p = plain_ids[i]   if i < len(plain_ids) else "—"
        e = exp_ids[i]     if i < len(exp_ids)   else "—"
        same = "Y" if p == e else "N"
        print(f"  {i+1:<5}  {p:<16}  {e:<16}  {same}")

    # NDCG@10: load current submission as ground truth
    if SUBMISSION_CSV.exists():
        import csv
        with open(SUBMISSION_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        top10_submitted = {r["candidate_id"] for r in rows if int(r["rank"]) <= 10}
        plain_ndcg = _ndcg_at_k(plain_ids, top10_submitted, k=10)
        exp_ndcg   = _ndcg_at_k(exp_ids, top10_submitted, k=10)
        print(f"\nNDCG@10 (BM25 pool vs submitted top-10 as relevance proxy):")
        print(f"  Plain:    {plain_ndcg:.4f}")
        print(f"  Expanded: {exp_ndcg:.4f}")
        delta = exp_ndcg - plain_ndcg
        sign  = "+" if delta >= 0 else ""
        verdict = "no regression" if delta >= -0.001 else "REGRESSION"
        print(f"  Delta:    {sign}{delta:.4f}  [{verdict}]")
    else:
        print("\nteam_xxx.csv not found — skipping NDCG comparison.")


def _ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Simple NDCG@k where relevance is binary (1 if in relevant set, else 0)."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, cid in enumerate(ranked_ids[:k])
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    main()
