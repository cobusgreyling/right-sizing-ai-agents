"""
Specialized Agent Router — Right-Sizing AI Agents
===================================================
Demonstrates routing tasks to purpose-built NVIDIA Nemotron 3 models
based on intent classification, instead of sending everything to one
massive model.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python agent_router.py
"""

import os
import json
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ---------------------------------------------------------------------------
# Model registry — each model is right-sized for its role
# ---------------------------------------------------------------------------
MODELS = {
    "reasoning": {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "description": "Complex reasoning, planning, and generation",
        "params": "12B active / 120B total",
    },
    "safety": {
        "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "description": "Content safety classification",
        "params": "8B",
    },
    "embedding": {
        "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "description": "Text and image embedding for retrieval",
        "params": "1.7B",
    },
    "general": {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "description": "General-purpose responses",
        "params": "12B active",
    },
}


def get_client() -> OpenAI:
    """Initialize NVIDIA API client via OpenAI-compatible endpoint."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Intent classifier — lightweight routing logic
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
    """
    Classify user intent to route to the right specialized model.

    In production, this could itself be a small classifier model.
    Here we use keyword matching for transparency.
    """
    query_lower = query.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)

    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return "general"
    return best_intent


# ---------------------------------------------------------------------------
# Specialized model calls
# ---------------------------------------------------------------------------
def call_reasoning_model(client: OpenAI, query: str) -> dict:
    """Route to the reasoning specialist with configurable thinking budget."""
    print(f"  -> Routing to REASONING model ({MODELS['reasoning']['params']})")

    response = client.chat.completions.create(
        model=MODELS["reasoning"]["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise reasoning agent. Think step-by-step. "
                    "Be thorough but concise."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return {
        "model": MODELS["reasoning"]["model"],
        "params": MODELS["reasoning"]["params"],
        "intent": "reasoning",
        "response": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


def call_safety_model(client: OpenAI, query: str) -> dict:
    """Route to the lightweight safety classifier."""
    print(f"  -> Routing to SAFETY model ({MODELS['safety']['params']})")

    response = client.chat.completions.create(
        model=MODELS["safety"]["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a content safety classifier. Analyze the following "
                    "content and respond with a JSON object containing: "
                    "'is_safe' (boolean), 'categories' (list of flagged categories "
                    "from: hate, harassment, violence, sexual, misinformation, "
                    "unauthorized_advice), and 'confidence' (0-1 float)."
                ),
            },
            {"role": "user", "content": f"Classify this content:\n\n{query}"},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return {
        "model": MODELS["safety"]["model"],
        "params": MODELS["safety"]["params"],
        "intent": "safety",
        "response": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


def call_embedding_model(client: OpenAI, query: str) -> dict:
    """Route to the embedding specialist for vector operations."""
    print(f"  -> Routing to EMBEDDING model ({MODELS['embedding']['params']})")

    response = client.embeddings.create(
        model=MODELS["embedding"]["model"],
        input=[query],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"},
    )
    embedding = response.data[0].embedding
    return {
        "model": MODELS["embedding"]["model"],
        "params": MODELS["embedding"]["params"],
        "intent": "embedding",
        "response": f"Generated {len(embedding)}-dim embedding vector",
        "embedding_preview": embedding[:5],
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


def call_general_model(client: OpenAI, query: str) -> dict:
    """Fallback to general-purpose model."""
    print(f"  -> Routing to GENERAL model ({MODELS['general']['params']})")

    response = client.chat.completions.create(
        model=MODELS["general"]["model"],
        messages=[{"role": "user", "content": query}],
        temperature=0.6,
        max_tokens=512,
    )
    return {
        "model": MODELS["general"]["model"],
        "params": MODELS["general"]["params"],
        "intent": "general",
        "response": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }


# ---------------------------------------------------------------------------
# Router dispatch
# ---------------------------------------------------------------------------
ROUTE_MAP = {
    "reasoning": call_reasoning_model,
    "safety": call_safety_model,
    "embedding": call_embedding_model,
    "general": call_general_model,
}


def route_query(client: OpenAI, query: str) -> dict:
    """Classify intent and dispatch to the right specialized model."""
    intent = classify_intent(query)
    print(f"\n{'='*60}")
    print(f"Query: {query[:80]}...")
    print(f"Classified intent: {intent.upper()}")

    handler = ROUTE_MAP[intent]
    result = handler(client, query)

    print(f"Tokens used: {result.get('tokens_used', 'N/A')}")
    print(f"{'='*60}")
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # Should route to REASONING
    "Analyze the trade-offs between using a single large language model versus "
    "a specialized model stack for production AI agents. Think step by step.",

    # Should route to SAFETY
    "Check content safety: classify whether the following text contains harmful "
    "or inappropriate material: 'The weather today is sunny and pleasant.'",

    # Should route to EMBEDDING
    "Generate a semantic search embedding for: 'NVIDIA Nemotron 3 architecture "
    "and deployment guide'",

    # Should route to GENERAL (no strong intent signal)
    "What is the capital of France?",
]


def main():
    print("=" * 60)
    print("  RIGHT-SIZING AI AGENTS: Specialized Model Router")
    print("  Demonstrating intent-based routing to purpose-built models")
    print("=" * 60)

    client = get_client()
    results = []

    for query in DEMO_QUERIES:
        try:
            result = route_query(client, query)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"error": str(e), "query": query[:50]})

    # Summary
    print("\n" + "=" * 60)
    print("  ROUTING SUMMARY")
    print("=" * 60)
    print(f"{'Intent':<12} {'Model Params':<25} {'Tokens':<10}")
    print("-" * 47)
    for r in results:
        if "error" not in r:
            print(
                f"{r['intent']:<12} {r['params']:<25} "
                f"{r.get('tokens_used', 'N/A'):<10}"
            )

    print("\nKey insight: Each task used a model RIGHT-SIZED for the job.")
    print("No single 400B+ model needed. Total cost: fraction of monolith.\n")


if __name__ == "__main__":
    main()
