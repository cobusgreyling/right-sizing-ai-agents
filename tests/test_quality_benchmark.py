"""Tests for the model selection benchmark (Example 07)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "07_model_selection_benchmark"))

from quality_benchmark import (
    QualityScore,
    BenchmarkEntry,
    PRICING,
    BENCHMARK_PROMPTS,
    estimate_cost,
    select_specialized_model,
)


class TestQualityScore:
    def test_overall_average(self):
        qs = QualityScore(relevance=4.0, accuracy=5.0, completeness=3.0)
        assert qs.overall == 4.0

    def test_perfect_score(self):
        qs = QualityScore(relevance=5.0, accuracy=5.0, completeness=5.0)
        assert qs.overall == 5.0

    def test_minimum_score(self):
        qs = QualityScore(relevance=1.0, accuracy=1.0, completeness=1.0)
        assert qs.overall == 1.0

    def test_default_zeros(self):
        qs = QualityScore()
        assert qs.overall == 0.0


class TestBenchmarkEntry:
    def test_default_quality(self):
        entry = BenchmarkEntry(
            prompt="test", approach="monolith",
            model_used="test", params="12B",
        )
        assert entry.quality.overall == 0.0
        assert entry.response == ""
        assert entry.latency_ms == 0.0

    def test_cost_field(self):
        entry = BenchmarkEntry(
            prompt="test", approach="monolith",
            model_used="test", params="12B",
            estimated_cost=0.05,
        )
        assert entry.estimated_cost == 0.05


class TestPricing:
    def test_all_models_have_pricing(self):
        for model, pricing in PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert "params" in pricing

    def test_safety_cheaper_than_reasoning(self):
        from config import REASONING_MODEL, SAFETY_MODEL
        assert PRICING[SAFETY_MODEL]["input"] < PRICING[REASONING_MODEL]["input"]

    def test_embedding_cheapest(self):
        from config import EMBED_MODEL
        min_input = min(p["input"] for p in PRICING.values())
        assert PRICING[EMBED_MODEL]["input"] == min_input


class TestEstimateCost:
    def test_zero_tokens(self):
        from config import REASONING_MODEL
        assert estimate_cost(REASONING_MODEL, 0, 0) == 0.0

    def test_known_cost(self):
        from config import REASONING_MODEL
        cost = estimate_cost(REASONING_MODEL, 1000, 1000)
        expected = 1000 / 1000 * 0.005 + 1000 / 1000 * 0.015
        assert abs(cost - expected) < 1e-6


class TestSelectSpecializedModel:
    def test_safety_routes_to_safety_model(self):
        from config import SAFETY_MODEL
        assert select_specialized_model("safety") == SAFETY_MODEL

    def test_reasoning_routes_to_reasoning_model(self):
        from config import REASONING_MODEL
        assert select_specialized_model("reasoning") == REASONING_MODEL

    def test_unknown_routes_to_reasoning(self):
        from config import REASONING_MODEL
        assert select_specialized_model("other") == REASONING_MODEL


class TestBenchmarkPrompts:
    def test_not_empty(self):
        assert len(BENCHMARK_PROMPTS) >= 3

    def test_all_have_required_keys(self):
        for p in BENCHMARK_PROMPTS:
            assert "prompt" in p
            assert "category" in p
            assert "expected_model" in p

    def test_diverse_categories(self):
        categories = {p["category"] for p in BENCHMARK_PROMPTS}
        assert len(categories) >= 2
