"""Tests for the fallback chain (Example 09)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "09_fallback_chains"))

from fallback_chain import (
    extract_confidence,
    MODEL_CHAIN,
    CONFIDENCE_THRESHOLD,
    ChainStep,
    ChainResult,
    DEMO_QUERIES,
)


class TestExtractConfidence:
    def test_json_at_end(self):
        raw = 'The capital of France is Paris.\n{"confidence": 0.95}'
        answer, confidence = extract_confidence(raw)
        assert answer == "The capital of France is Paris."
        assert confidence == 0.95

    def test_json_on_separate_line(self):
        raw = "Some answer.\n\n{\"confidence\": 0.42}"
        answer, confidence = extract_confidence(raw)
        assert "Some answer" in answer
        assert abs(confidence - 0.42) < 0.01

    def test_no_json_defaults(self):
        raw = "Just a plain answer with no confidence."
        answer, confidence = extract_confidence(raw)
        assert answer == raw.strip()
        assert confidence == 0.5

    def test_malformed_json(self):
        raw = "Answer here.\n{confidence: broken}"
        answer, confidence = extract_confidence(raw)
        assert confidence == 0.5

    def test_confidence_zero(self):
        raw = 'I am unsure.\n{"confidence": 0.0}'
        answer, confidence = extract_confidence(raw)
        assert confidence == 0.0

    def test_confidence_one(self):
        raw = 'Absolutely certain.\n{"confidence": 1.0}'
        answer, confidence = extract_confidence(raw)
        assert confidence == 1.0

    def test_multiline_answer(self):
        raw = (
            "Line 1 of the answer.\n"
            "Line 2 of the answer.\n"
            "Line 3 of the answer.\n"
            '{"confidence": 0.85}'
        )
        answer, confidence = extract_confidence(raw)
        assert "Line 1" in answer
        assert "Line 3" in answer
        assert confidence == 0.85


class TestModelChain:
    def test_at_least_two_tiers(self):
        assert len(MODEL_CHAIN) >= 2

    def test_ordered_cheapest_first(self):
        costs = [t["cost_per_1k_input"] for t in MODEL_CHAIN]
        assert costs == sorted(costs), "Chain should be ordered cheapest first"

    def test_all_tiers_have_required_fields(self):
        for tier in MODEL_CHAIN:
            assert "name" in tier
            assert "label" in tier
            assert "params" in tier
            assert "cost_per_1k_input" in tier
            assert "cost_per_1k_output" in tier
            assert "max_tokens" in tier


class TestConfidenceThreshold:
    def test_reasonable_range(self):
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_default_value(self):
        assert CONFIDENCE_THRESHOLD == 0.7


class TestChainResult:
    def test_default_values(self):
        result = ChainResult(query="test")
        assert result.final_response == ""
        assert result.final_tier == 0
        assert result.total_latency_ms == 0.0
        assert result.total_cost == 0.0
        assert result.steps == []

    def test_accumulation(self):
        result = ChainResult(query="test", total_latency_ms=100, total_cost=0.01)
        result.total_latency_ms += 200
        result.total_cost += 0.02
        assert result.total_latency_ms == 300
        assert abs(result.total_cost - 0.03) < 1e-6


class TestChainStep:
    def test_fields(self):
        step = ChainStep(
            tier=1, model="test", params="8B",
            response="answer", confidence=0.9,
            latency_ms=50, input_tokens=10, output_tokens=20,
            escalated=False,
        )
        assert step.tier == 1
        assert step.confidence == 0.9
        assert not step.escalated


class TestDemoQueries:
    def test_not_empty(self):
        assert len(DEMO_QUERIES) >= 3

    def test_variety(self):
        # At least some short and some long queries
        lengths = [len(q.split()) for q in DEMO_QUERIES]
        assert min(lengths) < 15
        assert max(lengths) > 20
