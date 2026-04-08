"""
Fallback Chains — Graceful Degradation Across Model Tiers
==========================================================
Demonstrates a "try-small-first" strategy: send the query to the smallest
capable model, evaluate its confidence, and escalate to a larger model
only when needed.  This minimizes cost for easy queries while preserving
quality for hard ones.

Author: Cobus Greyling

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python fallback_chain.py
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    SAFETY_MODEL,
)


# ---------------------------------------------------------------------------
# Model chain (smallest to largest)
# ---------------------------------------------------------------------------
MODEL_CHAIN = [
    {
        "name": SAFETY_MODEL,
        "label": "Tier 1 — Lightweight (8B)",
        "params": "8B",
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.002,
        "max_tokens": 256,
    },
    {
        "name": REASONING_MODEL,
        "label": "Tier 2 — Reasoning (12B active)",
        "params": "12B active",
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
        "max_tokens": 512,
    },
]

# Confidence threshold — if the model self-reports below this, escalate
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ChainStep:
    """One attempt in the fallback chain."""
    tier: int
    model: str
    params: str
    response: str
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    escalated: bool


@dataclass
class ChainResult:
    """Complete result from the fallback chain."""
    query: str
    steps: list[ChainStep] = field(default_factory=list)
    final_response: str = ""
    final_tier: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------
CONFIDENCE_PROMPT_SUFFIX = (
    "\n\nAfter your answer, on a new line write ONLY a JSON object: "
    '{"confidence": <float 0.0-1.0>} indicating how confident you are '
    "in your answer's correctness and completeness."
)


def extract_confidence(raw_response: str) -> tuple[str, float]:
    """
    Split model output into the actual answer and a confidence score.
    Returns (answer, confidence).
    """
    lines = raw_response.strip().split("\n")

    # Try to find a JSON confidence line near the end
    for i in range(len(lines) - 1, max(len(lines) - 4, -1), -1):
        line = lines[i].strip()
        if line.startswith("{") and "confidence" in line:
            try:
                parsed = json.loads(line)
                confidence = float(parsed.get("confidence", 0.5))
                answer = "\n".join(lines[:i]).strip()
                return answer, confidence
            except (json.JSONDecodeError, ValueError):
                continue

    # Fallback: check if any line contains confidence JSON
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if "confidence" in stripped and stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                confidence = float(parsed.get("confidence", 0.5))
                answer = "\n".join(lines[:i]).strip()
                return answer, confidence
            except (json.JSONDecodeError, ValueError):
                continue

    return raw_response.strip(), 0.5


# ---------------------------------------------------------------------------
# Chain execution
# ---------------------------------------------------------------------------
def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def run_chain(client: OpenAI, query: str,
              threshold: float = CONFIDENCE_THRESHOLD) -> ChainResult:
    """
    Try each model in the chain from smallest to largest.
    Stop as soon as confidence exceeds the threshold.
    """
    result = ChainResult(query=query)

    for tier_idx, tier in enumerate(MODEL_CHAIN):
        print(f"    [{tier['label']}] Calling...")

        augmented_prompt = query + CONFIDENCE_PROMPT_SUFFIX

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=tier["name"],
            messages=[{"role": "user", "content": augmented_prompt}],
            temperature=0.3,
            max_tokens=tier["max_tokens"],
        )
        latency = (time.perf_counter() - start) * 1000

        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        raw_text = response.choices[0].message.content

        answer, confidence = extract_confidence(raw_text)

        cost = (
            in_tok / 1000 * tier["cost_per_1k_input"]
            + out_tok / 1000 * tier["cost_per_1k_output"]
        )

        is_last = tier_idx == len(MODEL_CHAIN) - 1
        should_escalate = confidence < threshold and not is_last

        step = ChainStep(
            tier=tier_idx + 1,
            model=tier["name"],
            params=tier["params"],
            response=answer,
            confidence=confidence,
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            escalated=should_escalate,
        )
        result.steps.append(step)
        result.total_latency_ms += latency
        result.total_cost += cost

        if should_escalate:
            print(f"    Confidence {confidence:.2f} < {threshold} — ESCALATING")
        else:
            result.final_response = answer
            result.final_tier = tier_idx + 1
            if not is_last:
                print(f"    Confidence {confidence:.2f} >= {threshold} — ACCEPTED")
            else:
                print(f"    Final tier reached — confidence {confidence:.2f}")
            break

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    # Easy — should resolve at Tier 1
    "What is the capital of France?",

    # Easy — should resolve at Tier 1
    "What does HTML stand for?",

    # Medium — may escalate
    "Explain the difference between TCP and UDP, including when to use each.",

    # Hard — likely escalates to Tier 2
    "Design a caching strategy for a distributed microservices architecture "
    "that handles 10K requests per second. Consider cache invalidation, "
    "consistency trade-offs, and failure modes. Think step by step.",

    # Hard — likely escalates to Tier 2
    "Compare and analyze the trade-offs between event-driven architecture "
    "and request-response patterns for a real-time trading platform. "
    "Evaluate latency, throughput, fault tolerance, and debugging difficulty.",
]


def main():
    print("=" * 70)
    print("  FALLBACK CHAINS: Try Small First, Escalate When Needed")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Chain: {' -> '.join(t['params'] for t in MODEL_CHAIN)}")
    print("=" * 70)

    client = get_client()
    all_results = []

    for query in DEMO_QUERIES:
        print(f"\n  Query: {query[:60]}...")
        result = run_chain(client, query)
        all_results.append(result)
        print(f"  -> Resolved at Tier {result.final_tier} | "
              f"Cost: ${result.total_cost:.5f} | "
              f"Latency: {result.total_latency_ms:.0f}ms")

    # Summary
    print("\n" + "=" * 70)
    print("  FALLBACK CHAIN SUMMARY")
    print("=" * 70)

    print(f"\n  {'Query (truncated)':<45} {'Tier':>5} {'Steps':>6} "
          f"{'Latency':>8} {'Cost':>10}")
    print("  " + "-" * 75)

    for r in all_results:
        print(
            f"  {r.query[:43]:<45} {r.final_tier:>5} "
            f"{len(r.steps):>6} "
            f"{r.total_latency_ms:>7.0f}ms "
            f"${r.total_cost:>9.5f}"
        )

    tier1_count = sum(1 for r in all_results if r.final_tier == 1)
    total_cost = sum(r.total_cost for r in all_results)
    # Hypothetical cost if everything went to the largest model
    print(f"\n  Resolved at Tier 1 (small model): {tier1_count}/{len(all_results)}")
    print(f"  Total cost: ${total_cost:.5f}")
    print()
    print("  KEY INSIGHT: Simple queries resolve at the cheapest tier.")
    print("  The fallback chain only spends on the large model when it needs to.")
    print("  In production, 60-80% of queries are simple — this strategy can")
    print("  cut costs dramatically without sacrificing quality on hard queries.\n")


if __name__ == "__main__":
    main()
