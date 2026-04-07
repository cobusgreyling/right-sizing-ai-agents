"""Tests for the cost benchmark (Example 04)."""

import sys
import os
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "04_cost_comparison"))

from cost_benchmark import (
    BenchmarkResult,
    MONOLITH_MODEL,
    SPECIALIZED_MODELS,
    AGENT_WORKFLOW,
)


class TestBenchmarkResult:
    def test_dataclass_fields(self):
        field_names = {f.name for f in fields(BenchmarkResult)}
        expected = {"approach", "task", "model", "latency_ms", "input_tokens", "output_tokens", "estimated_cost"}
        assert field_names == expected

    def test_cost_calculation(self):
        result = BenchmarkResult(
            approach="test", task="test", model="test",
            latency_ms=100, input_tokens=1000, output_tokens=500,
            estimated_cost=1000 / 1000 * 0.005 + 500 / 1000 * 0.015,
        )
        assert abs(result.estimated_cost - 0.0125) < 1e-6


class TestPricingConfig:
    def test_monolith_pricing_exists(self):
        assert "cost_per_1k_input" in MONOLITH_MODEL
        assert "cost_per_1k_output" in MONOLITH_MODEL
        assert MONOLITH_MODEL["cost_per_1k_input"] > 0

    def test_specialized_pricing_cheaper(self):
        # Safety model should be cheaper than monolith
        assert SPECIALIZED_MODELS["safety"]["cost_per_1k_input"] < MONOLITH_MODEL["cost_per_1k_input"]
        assert SPECIALIZED_MODELS["safety"]["cost_per_1k_output"] < MONOLITH_MODEL["cost_per_1k_output"]

    def test_embedding_cheapest(self):
        emb = SPECIALIZED_MODELS["embedding"]
        for key in ["reasoning", "safety"]:
            assert emb["cost_per_1k_input"] < SPECIALIZED_MODELS[key]["cost_per_1k_input"]

    def test_all_specialized_roles_present(self):
        for role in ["reasoning", "safety", "embedding"]:
            assert role in SPECIALIZED_MODELS


class TestAgentWorkflow:
    def test_workflow_not_empty(self):
        assert len(AGENT_WORKFLOW) >= 3

    def test_all_steps_have_required_keys(self):
        for step in AGENT_WORKFLOW:
            assert "task" in step
            assert "description" in step
            assert "prompt" in step

    def test_workflow_covers_all_task_types(self):
        task_types = {step["task"] for step in AGENT_WORKFLOW}
        assert "reasoning" in task_types
        assert "safety" in task_types
        assert "embedding" in task_types

    def test_specialized_cost_lower_than_monolith(self):
        """Verify the cost model math: specialized should be cheaper overall."""
        mono_cost = 0
        spec_cost = 0
        for step in AGENT_WORKFLOW:
            # Assume 100 input tokens, 200 output tokens per step
            in_tok, out_tok = 100, 200
            if step["task"] == "embedding":
                out_tok = 0

            mono_cost += (
                in_tok / 1000 * MONOLITH_MODEL["cost_per_1k_input"]
                + out_tok / 1000 * MONOLITH_MODEL["cost_per_1k_output"]
            )

            sm = SPECIALIZED_MODELS[step["task"]]
            spec_cost += (
                in_tok / 1000 * sm["cost_per_1k_input"]
                + out_tok / 1000 * sm["cost_per_1k_output"]
            )

        assert spec_cost < mono_cost, f"Specialized ({spec_cost}) should be cheaper than monolith ({mono_cost})"
