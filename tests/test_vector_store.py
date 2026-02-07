"""Tests for the vector store module."""

import numpy as np
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def sample_store():
    """Create a VectorStore with known embeddings."""
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # Ticket A: points in x direction
        [0.0, 1.0, 0.0],  # Ticket B: points in y direction
        [0.7, 0.7, 0.0],  # Ticket C: between x and y
    ], dtype=np.float32)

    ticket_ids = ["TKT-A", "TKT-B", "TKT-C"]
    tickets = [
        {"id": "TKT-A", "title": "Equipment failure", "category": "equipment_failure"},
        {"id": "TKT-B", "title": "Production decline", "category": "production_decline"},
        {"id": "TKT-C", "title": "Mixed issue", "category": "equipment_failure"},
    ]
    return VectorStore(embeddings=embeddings, ticket_ids=ticket_ids, tickets=tickets)


class TestVectorStore:
    def test_search_returns_list(self, sample_store):
        query = np.array([1.0, 0.0, 0.0])
        results = sample_store.search(query, top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_most_similar_first(self, sample_store):
        query = np.array([1.0, 0.0, 0.0])
        results = sample_store.search(query, top_k=3)
        # TKT-A should be most similar (perfect match)
        assert results[0]["id"] == "TKT-A"
        assert results[0]["score"] == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_has_zero_similarity(self, sample_store):
        query = np.array([0.0, 0.0, 1.0])
        results = sample_store.search(query, top_k=3)
        # All embeddings have z=0, so all similarities should be ~0
        for r in results:
            assert abs(r["score"]) < 0.01

    def test_respects_top_k(self, sample_store):
        query = np.array([1.0, 0.0, 0.0])
        results = sample_store.search(query, top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_corpus(self, sample_store):
        query = np.array([1.0, 0.0, 0.0])
        results = sample_store.search(query, top_k=100)
        assert len(results) == 3  # only 3 tickets

    def test_empty_store(self):
        store = VectorStore(embeddings=np.array([]), ticket_ids=[])
        results = store.search(np.array([1.0, 0.0, 0.0]), top_k=5)
        assert results == []

    def test_result_contains_ticket_fields(self, sample_store):
        query = np.array([1.0, 0.0, 0.0])
        results = sample_store.search(query, top_k=1)
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "category" in r
        assert "score" in r

    def test_scores_are_sorted_descending(self, sample_store):
        query = np.array([0.6, 0.8, 0.0])
        results = sample_store.search(query, top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_none_embeddings_returns_empty(self):
        store = VectorStore(embeddings=None, ticket_ids=[])
        results = store.search(np.array([1.0]), top_k=5)
        assert results == []
