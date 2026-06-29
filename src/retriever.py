# Retrieves top-K candidates from precomputed embeddings via dense cosine similarity
# and optional BM25 hybrid (Reciprocal Rank Fusion) for exact keyword coverage.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ── Dense retrieval ───────────────────────────────────────────────────────────

def retrieve_top_k(
    embeddings: np.ndarray,
    jd_vector: np.ndarray,
    candidate_ids: np.ndarray,
    k: int = 50,
) -> tuple[list[str], np.ndarray]:
    """
    Return the top-k candidates by cosine similarity.

    Parameters
    ----------
    embeddings    : (n, 768) float32 — L2-normalised candidate embeddings
    jd_vector     : (1, 768) float32 — L2-normalised JD embedding
    candidate_ids : (n,) array of CAND_XXXXXXX strings, aligned with embeddings
    k             : number of candidates to return

    Returns
    -------
    (ids, scores)
      ids    : list of k candidate_id strings, ranked best-first
      scores : (k,) float32 cosine similarity scores
    """
    # Ensure float32 for matmul speed
    emb = embeddings.astype(np.float32)
    jd  = jd_vector.astype(np.float32)

    # Both are L2-normalised so dot product == cosine similarity
    # Shape: (n,)
    sims = emb @ jd.T.squeeze()

    k = min(k, len(sims))
    # argpartition is O(n) for the top-k, then sort only k elements
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    top_ids    = [str(candidate_ids[i]) for i in top_idx]
    top_scores = sims[top_idx]
    return top_ids, top_scores


# ── BM25 query expansion ──────────────────────────────────────────────────────

def expand_query_with_clusters(query: str, skill_clusters: dict) -> str:
    """
    Expand a BM25 query using skill synonym clusters from role_model.yaml.

    For each cluster, if any member term appears in the query, all sibling terms
    are appended. Multi-word terms (e.g. "learning to rank") are matched as
    substrings and added as space-joined tokens so BM25 scores each word.

    Returns the expanded query string (lowercase, deduplicated tokens).
    """
    query_lower = query.lower()
    extra: list[str] = []
    for _cluster_name, terms in skill_clusters.items():
        cluster_hit = False
        for term in terms:
            if term.lower() in query_lower:
                cluster_hit = True
                break
        if cluster_hit:
            for term in terms:
                if term.lower() not in query_lower:
                    extra.append(term.lower())
    if not extra:
        return query
    return query + " " + " ".join(extra)


# ── BM25 retrieval ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Build once on candidate texts, query at retrieval time.
    Optionally accepts skill_clusters (from role_model.yaml) for query expansion.
    """

    def __init__(
        self,
        candidate_ids: list[str],
        texts: list[str],
        skill_clusters: dict | None = None,
    ):
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._ids  = candidate_ids
        self._skill_clusters = skill_clusters or {}

    def retrieve_top_k(self, query: str, k: int = 50) -> tuple[list[str], np.ndarray]:
        """Return top-k (ids, scores) by BM25, with cluster-based query expansion."""
        if self._skill_clusters:
            query = expand_query_with_clusters(query, self._skill_clusters)
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        k = min(k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self._ids[i] for i in top_idx], scores[top_idx]


# ── Hybrid retrieval (Reciprocal Rank Fusion) ─────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k_rrf: int = 60,
) -> list[str]:
    """
    Merge multiple ranked lists via RRF.
    Returns a single merged ranking (ids only, best-first).

    k_rrf : RRF constant (60 is standard; higher = more lenient toward low ranks)
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)
