"""Tests for the dynamic router (Example 08)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "08_dynamic_routing"))

from dynamic_router import (
    score_complexity,
    get_tier_config,
    COMPLEXITY_SIGNALS,
    MODEL_TIERS,
    DEMO_QUERIES,
)


class TestScoreComplexity:
    def test_simple_query_low_score(self):
        result = score_complexity("What is the capital of France?")
        assert result.score < 0.4
        assert result.selected_tier == "lightweight"

    def test_complex_query_high_score(self):
        result = score_complexity(
            "Compare the trade-offs between fine-tuning and RAG, "
            "considering cost, latency, accuracy, and maintenance. "
            "Think step by step."
        )
        assert result.score > 0.5

    def test_multi_step_signals_increase_score(self):
        simple = score_complexity("What is machine learning?")
        complex_ = score_complexity(
            "Analyze the trade-offs between A and B, then compare "
            "the pros and cons step by step."
        )
        assert complex_.score > simple.score

    def test_word_count_tracked(self):
        result = score_complexity("one two three four five")
        assert result.word_count == 5

    def test_question_depth(self):
        result = score_complexity("What? Why? How?")
        assert result.question_depth >= 3

    def test_score_bounded(self):
        # Very short
        r1 = score_complexity("Hi")
        assert 0.0 <= r1.score <= 1.0
        # Very long
        long_q = "Analyze " * 100 + "step by step"
        r2 = score_complexity(long_q)
        assert 0.0 <= r2.score <= 1.0

    def test_signal_hits_populated(self):
        result = score_complexity("Analyze the trade-offs step by step")
        assert "multi_step" in result.signal_hits or "reasoning" in result.signal_hits

    def test_empty_query(self):
        result = score_complexity("")
        assert result.score >= 0.0
        assert result.word_count == 0


class TestGetTierConfig:
    def test_known_tiers(self):
        for tier in MODEL_TIERS:
            config = get_tier_config(tier["tier"])
            assert config["tier"] == tier["tier"]
            assert "model" in config
            assert "params" in config

    def test_unknown_tier_returns_last(self):
        config = get_tier_config("nonexistent")
        assert config["tier"] == MODEL_TIERS[-1]["tier"]


class TestModelTiers:
    def test_ordered_by_complexity(self):
        prev = 0
        for tier in MODEL_TIERS:
            assert tier["max_complexity"] > prev
            prev = tier["max_complexity"]

    def test_all_tiers_have_required_fields(self):
        for tier in MODEL_TIERS:
            assert "tier" in tier
            assert "model" in tier
            assert "params" in tier
            assert "max_complexity" in tier

    def test_last_tier_max_is_one(self):
        assert MODEL_TIERS[-1]["max_complexity"] == 1.0


class TestComplexitySignals:
    def test_all_categories_present(self):
        assert "multi_step" in COMPLEXITY_SIGNALS
        assert "reasoning" in COMPLEXITY_SIGNALS
        assert "simple" in COMPLEXITY_SIGNALS

    def test_patterns_not_empty(self):
        for category, patterns in COMPLEXITY_SIGNALS.items():
            assert len(patterns) > 0, f"{category} has no patterns"


class TestDemoQueries:
    def test_variety_of_complexities(self):
        scores = [score_complexity(q).score for q in DEMO_QUERIES]
        assert min(scores) < 0.3, "Should have at least one simple query"
        assert max(scores) > 0.5, "Should have at least one complex query"
