"""Tests for the RAG pipeline module."""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.vector_store import VectorStore


@pytest.fixture
def mock_store():
    """VectorStore with known tickets and embeddings."""
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    tickets = [
        {
            "id": "TKT-001",
            "title": "ESP motor tripped",
            "description": "Pump tripped on F-1 C.",
            "category": "equipment_failure",
            "priority": "critical",
            "resolution": "Restarted pump after cooling.",
        },
        {
            "id": "TKT-002",
            "title": "Production dropped 30%",
            "description": "Oil rate fell on F-11.",
            "category": "production_decline",
            "priority": "high",
            "resolution": "Choked back to reduce water coning.",
        },
    ]
    return VectorStore(
        embeddings=embeddings,
        ticket_ids=["TKT-001", "TKT-002"],
        tickets=tickets,
    )


class TestTriageTicket:
    def test_returns_expected_keys(self, mock_store):
        mock_embedding = [1.0, 0.0, 0.0]
        mock_classify = {
            "category": "equipment_failure",
            "priority": "high",
            "confidence": "high",
            "reasoning": "Pump issue",
        }
        mock_response = MagicMock()
        mock_response.text = "Suggested resolution: restart the pump."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with (
            patch("src.rag.embed_text", return_value=mock_embedding),
            patch("src.rag.classify_ticket", return_value=mock_classify),
            patch("src.rag._get_model", return_value=mock_model),
        ):
            from src.rag import triage_ticket

            result = triage_ticket(
                title="Pump failure",
                description="ESP tripped on well F-4",
                store=mock_store,
            )
            assert "classification" in result
            assert "similar_tickets" in result
            assert "resolution_suggestion" in result
            assert "query" in result

    def test_classification_included(self, mock_store):
        mock_classify = {
            "category": "equipment_failure",
            "priority": "critical",
            "confidence": "high",
            "reasoning": "ESP failure detected",
        }
        mock_response = MagicMock()
        mock_response.text = "Check and restart pump."

        with (
            patch("src.rag.embed_text", return_value=[1.0, 0.0, 0.0]),
            patch("src.rag.classify_ticket", return_value=mock_classify),
            patch("src.rag._get_model", return_value=MagicMock(
                generate_content=MagicMock(return_value=mock_response)
            )),
        ):
            from src.rag import triage_ticket

            result = triage_ticket("ESP fail", "Pump down", store=mock_store)
            assert result["classification"]["category"] == "equipment_failure"
            assert result["classification"]["priority"] == "critical"

    def test_similar_tickets_returned(self, mock_store):
        mock_response = MagicMock()
        mock_response.text = "Resolution suggestion."

        with (
            patch("src.rag.embed_text", return_value=[0.9, 0.1, 0.0]),
            patch("src.rag.classify_ticket", return_value={
                "category": "equipment_failure", "priority": "high",
                "confidence": "high", "reasoning": "test"
            }),
            patch("src.rag._get_model", return_value=MagicMock(
                generate_content=MagicMock(return_value=mock_response)
            )),
        ):
            from src.rag import triage_ticket

            result = triage_ticket("Pump issue", "Pump not working", store=mock_store, top_k=2)
            assert len(result["similar_tickets"]) == 2
            assert result["similar_tickets"][0]["id"] in ["TKT-001", "TKT-002"]

    def test_resolution_suggestion_is_string(self, mock_store):
        mock_response = MagicMock()
        mock_response.text = "Based on similar tickets, restart the pump after cooling."

        with (
            patch("src.rag.embed_text", return_value=[1.0, 0.0, 0.0]),
            patch("src.rag.classify_ticket", return_value={
                "category": "equipment_failure", "priority": "high",
                "confidence": "high", "reasoning": "test"
            }),
            patch("src.rag._get_model", return_value=MagicMock(
                generate_content=MagicMock(return_value=mock_response)
            )),
        ):
            from src.rag import triage_ticket

            result = triage_ticket("Test", "Test desc", store=mock_store)
            assert isinstance(result["resolution_suggestion"], str)
            assert len(result["resolution_suggestion"]) > 0

    def test_query_preserved_in_result(self, mock_store):
        mock_response = MagicMock()
        mock_response.text = "Suggestion."

        with (
            patch("src.rag.embed_text", return_value=[1.0, 0.0, 0.0]),
            patch("src.rag.classify_ticket", return_value={
                "category": "equipment_failure", "priority": "high",
                "confidence": "high", "reasoning": "test"
            }),
            patch("src.rag._get_model", return_value=MagicMock(
                generate_content=MagicMock(return_value=mock_response)
            )),
        ):
            from src.rag import triage_ticket

            result = triage_ticket("My Title", "My Description", store=mock_store)
            assert result["query"]["title"] == "My Title"
            assert result["query"]["description"] == "My Description"
