"""Tests for the batch throughput benchmark (Example 10)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "10_batch_processing"))

from batch_throughput import (
    BatchItem,
    BatchResult,
    BatchReport,
    classify_batch,
    BATCH_QUERIES,
    MODEL_MAP,
)


class TestBatchItem:
    def test_fields(self):
        item = BatchItem(query="test", category="reasoning", index=0)
        assert item.query == "test"
        assert item.category == "reasoning"
        assert item.index == 0


class TestBatchResult:
    def test_fields(self):
        result = BatchResult(
            index=0, query="test", category="reasoning",
            model="test-model", params="12B",
            latency_ms=100, tokens=50, response="answer",
        )
        assert result.latency_ms == 100
        assert result.tokens == 50


class TestBatchReport:
    def test_throughput_calculation(self):
        report = BatchReport(
            approach="test", total_wall_time_ms=1000,
            total_model_time_ms=2000,
            results=[
                BatchResult(i, "q", "c", "m", "p", 100, 10, "r")
                for i in range(5)
            ],
        )
        # 5 queries in 1 second = 5 qps
        assert report.throughput_qps == 5.0

    def test_throughput_zero_time(self):
        report = BatchReport(approach="test", total_wall_time_ms=0,
                             total_model_time_ms=0)
        assert report.throughput_qps == 0

    def test_avg_latency(self):
        results = [
            BatchResult(i, "q", "c", "m", "p", lat, 10, "r")
            for i, lat in enumerate([100, 200, 300])
        ]
        report = BatchReport(approach="test", total_wall_time_ms=300,
                             total_model_time_ms=600, results=results)
        assert report.avg_latency_ms == 200.0

    def test_avg_latency_empty(self):
        report = BatchReport(approach="test", total_wall_time_ms=0,
                             total_model_time_ms=0)
        assert report.avg_latency_ms == 0


class TestClassifyBatch:
    def test_returns_correct_count(self):
        items = classify_batch(BATCH_QUERIES)
        assert len(items) == len(BATCH_QUERIES)

    def test_indices_sequential(self):
        items = classify_batch(BATCH_QUERIES)
        for i, item in enumerate(items):
            assert item.index == i

    def test_categories_valid(self):
        items = classify_batch(BATCH_QUERIES)
        valid = {"reasoning", "safety", "embedding", "general"}
        for item in items:
            assert item.category in valid

    def test_preserves_query_text(self):
        items = classify_batch(BATCH_QUERIES)
        for i, item in enumerate(items):
            assert item.query == BATCH_QUERIES[i]


class TestModelMap:
    def test_all_categories_mapped(self):
        for cat in ["reasoning", "safety", "embedding", "general"]:
            assert cat in MODEL_MAP

    def test_configs_have_required_keys(self):
        for cat, config in MODEL_MAP.items():
            assert "model" in config
            assert "params" in config
            assert "max_tokens" in config

    def test_safety_has_lower_max_tokens(self):
        assert MODEL_MAP["safety"]["max_tokens"] < MODEL_MAP["reasoning"]["max_tokens"]


class TestBatchQueries:
    def test_not_empty(self):
        assert len(BATCH_QUERIES) >= 6

    def test_diverse_categories(self):
        items = classify_batch(BATCH_QUERIES)
        categories = {item.category for item in items}
        # Should have at least 2 different categories
        assert len(categories) >= 2
