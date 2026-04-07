"""Tests for the specialized agent router (Example 01)."""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "01_specialized_routing"))

from agent_router import classify_intent, INTENT_KEYWORDS, MODELS


class TestClassifyIntent:
    def test_reasoning_keywords(self):
        assert classify_intent("Analyze the trade-offs between models") == "reasoning"
        assert classify_intent("Think step by step about this problem") == "reasoning"
        assert classify_intent("Compare these two approaches") == "reasoning"
        assert classify_intent("Evaluate the design of this system") == "reasoning"

    def test_safety_keywords(self):
        assert classify_intent("Check content safety of this text") == "safety"
        assert classify_intent("Is this safe to publish?") == "safety"
        assert classify_intent("Moderate this user submission") == "safety"
        assert classify_intent("Filter toxic content") == "safety"

    def test_embedding_keywords(self):
        assert classify_intent("Generate a semantic search embedding") == "embedding"
        assert classify_intent("Find similar documents") == "embedding"
        assert classify_intent("Create a vector representation") == "embedding"

    def test_general_fallback(self):
        assert classify_intent("What is the capital of France?") == "general"
        assert classify_intent("Hello, how are you?") == "general"
        assert classify_intent("Tell me a joke") == "general"

    def test_case_insensitive(self):
        assert classify_intent("ANALYZE this problem STEP BY STEP") == "reasoning"
        assert classify_intent("CHECK CONTENT safety") == "safety"

    def test_multiple_keyword_matches_highest_wins(self):
        # Both reasoning and safety keywords, but more reasoning
        query = "Analyze and evaluate the design of our safety checks step by step"
        result = classify_intent(query)
        assert result == "reasoning"


class TestModelRegistry:
    def test_all_roles_have_models(self):
        for role in ["reasoning", "safety", "embedding", "general"]:
            assert role in MODELS
            assert "model" in MODELS[role]
            assert "params" in MODELS[role]

    def test_model_names_are_nvidia(self):
        for role, config in MODELS.items():
            assert config["model"].startswith("nvidia/"), f"{role} model should be NVIDIA"

    def test_intent_keywords_cover_all_specialized_roles(self):
        for role in ["reasoning", "safety", "embedding"]:
            assert role in INTENT_KEYWORDS
            assert len(INTENT_KEYWORDS[role]) > 0
