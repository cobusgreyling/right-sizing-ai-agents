"""
Dynamic Routing — Adaptive Model Selection Based on Query Complexity
====================================================================
Routes queries to different model tiers based on estimated complexity.
Simple factual queries go to smaller models; multi-step reasoning
queries escalate to larger ones. Complexity is scored locally using
heuristics (token count, question depth, keyword signals) — no LLM
call required for the routing decision itself.

Author: Cobus Greyling

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python dynamic_router.py
"""

import os
import sys
import re
import time
from dataclasses import dataclass
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    SAFETY_MODEL,
)


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------
COMPLEXITY_SIGNALS = {
    "multi_step": [
        "step by step", "first.*then", "compare.*and", "analyze.*and",
        "evaluate.*considering", "trade-offs", "pros and cons",
    ],
    "reasoning": [
        "why", "how does", "explain the relationship", "what would happen if",
        "design a", "architect", "debug", "optimize",
    ],
    "simple": [
        "what is", "who is", "when did", "where is", "define",
        "list", "name the", "how many",
    ],
}

# Model tiers ordered by capability (and cost)
MODEL_TIERS = [
    {
        "tier": "lightweight",
        "model": SAFETY_MODEL,
        "params": "8B",
        "max_complexity": 0.3,
        "description": "Simple factual queries, classifications",
    },
    {
        "tier": "standard",
        "model": REASONING_MODEL,
        "params": "12B active",
        "max_complexity": 0.7,
        "description": "Moderate reasoning, explanations",
    },
    {
        "tier": "heavyweight",
        "model": REASONING_MODEL,
        "params": "12B active (extended thinking)",
        "max_complexity": 1.0,
        "description": "Complex multi-step reasoning, analysis",
    },
]


@dataclass
class ComplexityAssessment:
    """Result of local complexity analysis."""
    score: float          # 0.0 (trivial) to 1.0 (very complex)
    word_count: int
    question_depth: int   # number of sub-questions / clauses
    signal_hits: dict     # which complexity signals matched
    selected_tier: str


def score_complexity(query: str) -> ComplexityAssessment:
    """
    Estimate query complexity using local heuristics.
    No LLM call needed — this runs in microseconds.
    """
    query_lower = query.lower()
    words = query.split()
    word_count = len(words)

    # Count signal matches
    signal_hits = {}
    for category, patterns in COMPLEXITY_SIGNALS.items():
        hits = [p for p in patterns if re.search(p, query_lower)]
        if hits:
            signal_hits[category] = hits

    # Question depth: count question marks, conjunctions, commas
    question_depth = (
        query.count("?")
        + query_lower.count(" and ")
        + query_lower.count(" or ")
        + min(query.count(","), 3)
    )

    # Base score from word count (longer = likely more complex)
    length_score = min(word_count / 60, 1.0)

    # Signal-based adjustments
    signal_score = 0.0
    if "multi_step" in signal_hits:
        signal_score += 0.4
    if "reasoning" in signal_hits:
        signal_score += 0.3
    if "simple" in signal_hits:
        signal_score -= 0.3

    # Depth adjustment
    depth_score = min(question_depth / 5, 0.3)

    # Combine
    raw_score = length_score * 0.3 + signal_score + depth_score
    score = max(0.0, min(1.0, raw_score))

    # Select tier
    selected_tier = MODEL_TIERS[-1]["tier"]
    for tier in MODEL_TIERS:
        if score <= tier["max_complexity"]:
            selected_tier = tier["tier"]
            break

    return ComplexityAssessment(
        score=round(score, 3),
        word_count=word_count,
        question_depth=question_depth,
        signal_hits=signal_hits,
        selected_tier=selected_tier,
    )


def get_tier_config(tier_name: str) -> dict:
    """Look up model configuration for a tier."""
    for tier in MODEL_TIERS:
        if tier["tier"] == tier_name:
            return tier
    return MODEL_TIERS[-1]


# ---------------------------------------------------------------------------
# Dynamic routing
# ---------------------------------------------------------------------------
def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def route_and_call(client: OpenAI, query: str) -> dict:
    """Score complexity, select model tier, and call the appropriate model."""
    assessment = score_complexity(query)
    tier = get_tier_config(assessment.selected_tier)

    print(f"\n{'='*70}")
    print(f"  Query:      {query[:65]}...")
    print(f"  Complexity: {assessment.score:.2f} "
          f"({assessment.word_count} words, depth={assessment.question_depth})")
    print(f"  Signals:    {assessment.signal_hits or 'none'}")
    print(f"  Tier:       {tier['tier'].upper()} -> {tier['params']}")
    print(f"{'='*70}")

    # Adjust generation params by tier
    if tier["tier"] == "lightweight":
        temperature, max_tokens = 0.1, 256
    elif tier["tier"] == "standard":
        temperature, max_tokens = 0.3, 512
    else:
        temperature, max_tokens = 0.2, 1024

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=tier["model"],
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = (time.perf_counter() - start) * 1000

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return {
        "query": query,
        "complexity": assessment.score,
        "tier": tier["tier"],
        "model": tier["model"],
        "params": tier["params"],
        "latency_ms": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response": response.choices[0].message.content,
    }


# ---------------------------------------------------------------------------
# Demo queries — spanning the complexity spectrum
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # Simple (should route to lightweight)
    "What is the capital of Japan?",

    # Simple (should route to lightweight)
    "Define 'retrieval-augmented generation' in one sentence.",

    # Moderate (should route to standard)
    "How does a transformer attention mechanism work? Explain the key "
    "components including queries, keys, and values.",

    # Complex (should route to heavyweight)
    "Compare the trade-offs between fine-tuning a foundation model and "
    "using retrieval-augmented generation for a customer support chatbot. "
    "Consider cost, latency, accuracy, and maintenance. Think step by step.",

    # Complex (should route to heavyweight)
    "Design a production architecture for a multi-agent AI system that "
    "handles customer onboarding. Analyze the trade-offs between using "
    "specialized models vs. a single large model, considering scalability, "
    "cost, and reliability. What would happen if one model tier goes down?",
]


def main():
    print("=" * 70)
    print("  DYNAMIC ROUTING: Adaptive Model Selection by Complexity")
    print("  Simple queries -> small model | Complex queries -> large model")
    print("=" * 70)

    client = get_client()
    results = []

    for query in DEMO_QUERIES:
        try:
            result = route_and_call(client, query)
            results.append(result)
            print(f"  Response: {result['response'][:100]}...")
            print(f"  Latency: {result['latency_ms']:.0f}ms | "
                  f"Tokens: {result['input_tokens']}+{result['output_tokens']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("  ROUTING SUMMARY")
    print("=" * 70)
    print(f"\n  {'Query (truncated)':<40} {'Complexity':>10} {'Tier':<14} {'Latency':>8}")
    print("  " + "-" * 72)

    for r in results:
        print(
            f"  {r['query'][:38]:<40} {r['complexity']:>10.2f} "
            f"{r['tier']:<14} {r['latency_ms']:>7.0f}ms"
        )

    tier_counts = {}
    for r in results:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

    print(f"\n  Distribution: {tier_counts}")
    print("  Dynamic routing saves cost by using smaller models for simpler tasks")
    print("  without sacrificing quality on complex queries.\n")


if __name__ == "__main__":
    main()
