"""
Shared configuration for Right-Sizing AI Agents.

Centralizes model identifiers, safety taxonomy, and intent classification
so that app.py and the example scripts stay in sync.
"""

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ---------------------------------------------------------------------------
# Model identifiers — single source of truth
# ---------------------------------------------------------------------------
REASONING_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
SAFETY_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

# ---------------------------------------------------------------------------
# Safety taxonomy
# ---------------------------------------------------------------------------
SAFETY_TAXONOMY = [
    "hate_speech", "harassment", "violence", "sexual_content",
    "misinformation", "unauthorized_advice", "self_harm",
    "illegal_activity", "personal_data", "profanity",
]

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------
INTENT_KEYWORDS = {
    "reasoning": [
        "analyze", "compare", "plan", "explain why", "reason",
        "think through", "step by step", "evaluate", "design",
    ],
    "safety": [
        "is this safe", "check content", "moderate", "filter",
        "appropriate", "harmful", "toxic", "classify safety",
    ],
    "embedding": [
        "embed", "vector", "similarity", "search documents",
        "find similar", "retrieve", "semantic search",
    ],
}


def classify_intent(query: str) -> str:
    """Classify user intent to route to the right specialized model."""
    query_lower = query.lower()
    scores = {intent: sum(1 for kw in keywords if kw in query_lower)
              for intent, keywords in INTENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"
