"""Tests for the classifier module."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.classifier import classify_ticket, VALID_CATEGORIES, VALID_PRIORITIES


@pytest.fixture
def mock_response_text():
    """Create a mock JSON response string."""
    def _make(category="equipment_failure", priority="high", confidence="high", reasoning="Test"):
        return json.dumps({
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "reasoning": reasoning,
        })
    return _make


def _patch_generate(return_value):
    """Patch both _get_client and _generate so no real API calls are made."""
    return (
        patch("src.classifier._get_client", return_value=MagicMock()),
        patch("src.classifier._generate", return_value=return_value),
    )


class TestClassifyTicket:
    def test_returns_expected_keys(self, mock_response_text):
        p1, p2 = _patch_generate(mock_response_text())
        with p1, p2:
            result = classify_ticket("ESP failed", "Pump tripped on well F-11")
            assert "category" in result
            assert "priority" in result
            assert "confidence" in result
            assert "reasoning" in result

    def test_valid_category(self, mock_response_text):
        p1, p2 = _patch_generate(mock_response_text(category="production_decline"))
        with p1, p2:
            result = classify_ticket("Production drop", "Oil rate fell 30%")
            assert result["category"] in VALID_CATEGORIES

    def test_valid_priority(self, mock_response_text):
        p1, p2 = _patch_generate(mock_response_text(priority="critical"))
        with p1, p2:
            result = classify_ticket("H2S alarm", "Gas detected at wellpad")
            assert result["priority"] in VALID_PRIORITIES

    def test_handles_invalid_json_falls_back(self):
        """When LLM returns invalid JSON, keyword fallback kicks in."""
        p1, p2 = _patch_generate("This is not JSON")
        with p1, p2:
            result = classify_ticket("Test", "Test description")
            assert result["category"] in VALID_CATEGORIES
            assert result["confidence"] == "medium"

    def test_handles_markdown_code_fences(self):
        fenced = '```json\n{"category": "safety_incident", "priority": "critical", "confidence": "high", "reasoning": "H2S is dangerous"}\n```'
        p1, p2 = _patch_generate(fenced)
        with p1, p2:
            result = classify_ticket("H2S alarm", "Gas detected")
            assert result["category"] == "safety_incident"
            assert result["priority"] == "critical"

    def test_invalid_category_defaults_to_unknown(self):
        text = json.dumps({
            "category": "nonexistent_category",
            "priority": "high",
            "confidence": "medium",
            "reasoning": "test",
        })
        p1, p2 = _patch_generate(text)
        with p1, p2:
            result = classify_ticket("Test", "Test")
            assert result["category"] == "unknown"

    def test_invalid_priority_defaults_to_medium(self):
        text = json.dumps({
            "category": "equipment_failure",
            "priority": "ultra_high",
            "confidence": "medium",
            "reasoning": "test",
        })
        p1, p2 = _patch_generate(text)
        with p1, p2:
            result = classify_ticket("Test", "Test")
            assert result["priority"] == "medium"
