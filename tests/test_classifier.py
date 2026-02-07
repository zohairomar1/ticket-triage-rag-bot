"""Tests for the classifier module."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.classifier import classify_ticket, VALID_CATEGORIES, VALID_PRIORITIES


@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini response with valid JSON."""
    def _make(category="equipment_failure", priority="high", confidence="high", reasoning="Test"):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "reasoning": reasoning,
        })
        return mock_response
    return _make


class TestClassifyTicket:
    def test_returns_expected_keys(self, mock_gemini_response):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_gemini_response()

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("ESP failed", "Pump tripped on well F-11")
            assert "category" in result
            assert "priority" in result
            assert "confidence" in result
            assert "reasoning" in result

    def test_valid_category(self, mock_gemini_response):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_gemini_response(
            category="production_decline"
        )

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("Production drop", "Oil rate fell 30%")
            assert result["category"] in VALID_CATEGORIES

    def test_valid_priority(self, mock_gemini_response):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_gemini_response(
            priority="critical"
        )

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("H2S alarm", "Gas detected at wellpad")
            assert result["priority"] in VALID_PRIORITIES

    def test_handles_invalid_json(self):
        mock_response = MagicMock()
        mock_response.text = "This is not JSON"

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("Test", "Test description")
            assert result["category"] == "unknown"
            assert result["confidence"] == "low"

    def test_handles_markdown_code_fences(self, mock_gemini_response):
        mock_response = MagicMock()
        mock_response.text = '```json\n{"category": "safety_incident", "priority": "critical", "confidence": "high", "reasoning": "H2S is dangerous"}\n```'

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("H2S alarm", "Gas detected")
            assert result["category"] == "safety_incident"
            assert result["priority"] == "critical"

    def test_invalid_category_defaults_to_unknown(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "nonexistent_category",
            "priority": "high",
            "confidence": "medium",
            "reasoning": "test",
        })

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("Test", "Test")
            assert result["category"] == "unknown"

    def test_invalid_priority_defaults_to_medium(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "equipment_failure",
            "priority": "ultra_high",
            "confidence": "medium",
            "reasoning": "test",
        })

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("src.classifier._get_model", return_value=mock_model):
            result = classify_ticket("Test", "Test")
            assert result["priority"] == "medium"
