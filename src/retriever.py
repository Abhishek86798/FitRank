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


# ── BM25 retrieval ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Build once on candidate texts, query at retrieval time.
    """

    def __init__(self, candidate_ids: list[str], texts: list[str]):
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._ids  = candidate_ids

    def retrieve_top_k(self, query: str, k: int = 50) -> tuple[list[str], np.ndarray]:
        """Return top-k (ids, scores) by BM25."""
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


class HybridRetriever:
    """
    Combines dense (cosine) and BM25 retrieval via Reciprocal Rank Fusion.

    Use when you need both semantic coverage (dense) and exact keyword hits
    (BM25 catches things like "Qdrant", "LambdaMART", "NDCG" that embeddings
    might miss if they're rare in training data).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        jd_vector: np.ndarray,
        candidate_ids: np.ndarray,
        texts: list[str],
        k_rrf: int = 60,
    ):
        self._embeddings    = embeddings
        self._jd_vector     = jd_vector
        self._candidate_ids = candidate_ids
        self._bm25          = BM25Retriever(list(candidate_ids), texts)
        self._k_rrf         = k_rrf

    def retrieve_top_k(self, jd_query: str, k: int = 50) -> list[str]:
        """
        Return top-k candidate IDs via dense+BM25 RRF fusion.

        jd_query : raw JD text string for BM25 (dense uses precomputed jd_vector)
        k        : final number of candidates to return
        """
        fetch = min(k * 3, len(self._candidate_ids))  # over-fetch before merging

        dense_ids, _ = retrieve_top_k(
            self._embeddings, self._jd_vector, self._candidate_ids, k=fetch
        )
        bm25_ids, _  = self._bm25.retrieve_top_k(jd_query, k=fetch)

        merged = reciprocal_rank_fusion([dense_ids, bm25_ids], k_rrf=self._k_rrf)
        return merged[:k]
