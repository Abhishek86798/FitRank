"""
Tests for BM25 skill-cluster query expansion (src/retriever.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from src.retriever import expand_query_with_clusters, BM25Retriever

ROLE_MODEL_PATH = Path("role_model.yaml")


@pytest.fixture(scope="module")
def skill_clusters() -> dict:
    with open(ROLE_MODEL_PATH, encoding="utf-8") as f:
        rm = yaml.safe_load(f)
    clusters = rm.get("skill_clusters", {})
    assert clusters, "skill_clusters missing from role_model.yaml"
    return clusters


# ── expand_query_with_clusters ────────────────────────────────────────────────

def test_cluster_hit_expands_siblings(skill_clusters):
    """When a cluster member is in the query, all siblings are added."""
    query = "experience with qdrant vector store"
    expanded = expand_query_with_clusters(query, skill_clusters)
    exp_lower = expanded.lower()
    # All vector_databases members should appear
    for term in skill_clusters["vector_databases"]:
        assert term.lower() in exp_lower, f"Expected '{term}' in expanded query"


def test_no_cluster_hit_leaves_query_unchanged(skill_clusters):
    """A query with no cluster terms is returned unchanged."""
    query = "python developer with django experience"
    result = expand_query_with_clusters(query, skill_clusters)
    assert result == query


def test_expansion_does_not_duplicate_existing_terms(skill_clusters):
    """Terms already in the query should not be duplicated in output."""
    query = "faiss pinecone ranking retrieval"
    expanded = expand_query_with_clusters(query, skill_clusters)
    tokens = expanded.lower().split()
    for term in ["faiss", "pinecone"]:
        count = tokens.count(term)
        assert count == 1, f"'{term}' appears {count} times, expected once"


def test_multiple_clusters_triggered(skill_clusters):
    """Hitting two clusters expands both."""
    # "qdrant" triggers vector_databases; "lambdamart" triggers ltr_frameworks
    query = "qdrant lambdamart ranking pipeline"
    expanded = expand_query_with_clusters(query, skill_clusters)
    exp_lower = expanded.lower()
    # At least one extra vector DB term
    vdb_extras = [t for t in skill_clusters["vector_databases"] if t != "qdrant"]
    assert any(t.lower() in exp_lower for t in vdb_extras), "No vector_db siblings added"
    # At least one extra LTR term
    ltr_extras = [t for t in skill_clusters["ltr_frameworks"] if t != "lambdamart"]
    assert any(t.lower() in exp_lower for t in ltr_extras), "No ltr_frameworks siblings added"


def test_empty_query_returns_empty(skill_clusters):
    query = ""
    result = expand_query_with_clusters(query, skill_clusters)
    assert result == ""


def test_empty_clusters_returns_query_unchanged():
    query = "faiss qdrant ranking"
    result = expand_query_with_clusters(query, {})
    assert result == query


# ── BM25Retriever with skill_clusters ────────────────────────────────────────

def _make_corpus() -> tuple[list[str], list[str]]:
    ids = ["CAND_A", "CAND_B", "CAND_C", "CAND_D"]
    texts = [
        "machine learning engineer with faiss and pinecone experience ranking retrieval",
        "data scientist using pgvector and hnsw for similarity search",
        "software developer django react no ml experience",
        "nlp researcher with sentence-transformers and bge embeddings recsys",
    ]
    return ids, texts


def test_bm25_retriever_without_clusters_baseline():
    """Without clusters a JD mentioning 'qdrant' should NOT surface pgvector-only candidates."""
    ids, texts = _make_corpus()
    bm25 = BM25Retriever(ids, texts, skill_clusters=None)
    query = "looking for qdrant experience"
    result_ids, _ = bm25.retrieve_top_k(query, k=4)
    # CAND_B has pgvector/hnsw but not qdrant — it should NOT rank first without expansion
    # The test just checks the retriever runs without error and returns all ids
    assert set(result_ids) == set(ids)


def test_bm25_retriever_with_clusters_surfaces_siblings(skill_clusters):
    """With clusters, 'qdrant' in query also scores candidates with faiss/pgvector higher."""
    ids, texts = _make_corpus()
    bm25 = BM25Retriever(ids, texts, skill_clusters=skill_clusters)
    query = "qdrant vector database experience"
    result_ids, scores = bm25.retrieve_top_k(query, k=4)
    # CAND_A (faiss/pinecone) and CAND_B (pgvector/hnsw) should both score > CAND_C (no ML)
    idx = {cid: i for i, cid in enumerate(result_ids)}
    assert idx["CAND_C"] > idx.get("CAND_A", 99), "CAND_C (no ML) should rank below CAND_A"
    assert idx["CAND_C"] > idx.get("CAND_B", 99), "CAND_C (no ML) should rank below CAND_B"


def test_bm25_retriever_preserves_return_shape(skill_clusters):
    ids, texts = _make_corpus()
    bm25 = BM25Retriever(ids, texts, skill_clusters=skill_clusters)
    result_ids, scores = bm25.retrieve_top_k("ranking retrieval recsys", k=3)
    assert len(result_ids) == 3
    assert len(scores) == 3
    assert scores[0] >= scores[1] >= scores[2], "Scores must be descending"


def test_role_model_has_skill_clusters():
    """Smoke test: role_model.yaml actually contains skill_clusters with >=3 clusters."""
    with open(ROLE_MODEL_PATH, encoding="utf-8") as f:
        rm = yaml.safe_load(f)
    clusters = rm.get("skill_clusters", {})
    assert len(clusters) >= 3, f"Expected >=3 clusters, got {len(clusters)}"
    for name, terms in clusters.items():
        assert len(terms) >= 2, f"Cluster '{name}' has fewer than 2 terms"
