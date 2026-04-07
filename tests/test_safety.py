"""Tests for the safety classifier (Example 02)."""

import sys
import os
import json
from unittest.mock import MagicMock, patch
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "02_safety_classifier"))

from safety_check import (
    SafetyResult,
    AgentResponse,
    classify_safety,
    SAFETY_TAXONOMY,
    SAFETY_MODEL,
)


class TestSafetyResult:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(SafetyResult)}
        assert field_names == {"is_safe", "categories", "confidence", "latency_ms", "model"}

    def test_creation(self):
        result = SafetyResult(
            is_safe=True, categories=[], confidence=0.95,
            latency_ms=5.0, model="test-model",
        )
        assert result.is_safe is True
        assert result.confidence == 0.95


class TestAgentResponse:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(AgentResponse)}
        assert field_names == {"content", "safety", "was_filtered", "reasoning_latency_ms", "total_latency_ms"}


class TestSafetyTaxonomy:
    def test_taxonomy_not_empty(self):
        assert len(SAFETY_TAXONOMY) >= 5

    def test_core_categories_present(self):
        for cat in ["hate_speech", "harassment", "violence", "sexual_content"]:
            assert cat in SAFETY_TAXONOMY


class TestClassifySafety:
    def _mock_client(self, response_text: str) -> MagicMock:
        client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = response_text
        client.chat.completions.create.return_value = mock_resp
        return client

    def test_parses_valid_json(self):
        resp = '{"is_safe": true, "categories": [], "confidence": 0.98}'
        result = classify_safety(self._mock_client(resp), "safe text")
        assert result.is_safe is True
        assert result.categories == []
        assert result.confidence == 0.98

    def test_parses_unsafe_content(self):
        resp = '{"is_safe": false, "categories": ["violence"], "confidence": 0.91}'
        result = classify_safety(self._mock_client(resp), "bad text")
        assert result.is_safe is False
        assert "violence" in result.categories

    def test_handles_markdown_wrapped_json(self):
        resp = '```json\n{"is_safe": true, "categories": [], "confidence": 0.95}\n```'
        result = classify_safety(self._mock_client(resp), "test")
        assert result.is_safe is True

    def test_handles_malformed_json_gracefully(self):
        result = classify_safety(self._mock_client("not json at all"), "test")
        # Should fail-open
        assert result.is_safe is True
        assert result.confidence == 0.0

    def test_uses_safety_model(self):
        client = self._mock_client('{"is_safe": true, "categories": [], "confidence": 0.9}')
        classify_safety(client, "test")
        call_args = client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == SAFETY_MODEL
