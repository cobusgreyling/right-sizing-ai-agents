"""
Cost & Latency Comparison: Monolith vs. Specialized Stack
==========================================================
Benchmarks the real-world cost and latency differences between
routing all tasks through a single large model vs. using NVIDIA's
specialized Nemotron 3 stack.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python cost_benchmark.py
"""

import os
import sys
import time
from dataclasses import dataclass
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    SAFETY_MODEL,
    EMBED_MODEL,
)

# ---------------------------------------------------------------------------
# Model configurations with estimated costs
# ---------------------------------------------------------------------------
# Pricing is illustrative based on typical API pricing tiers
MONOLITH_MODEL = {
    "name": REASONING_MODEL,
    "label": "Monolith (Large Model for Everything)",
    "cost_per_1k_input": 0.005,    # $/1K input tokens
    "cost_per_1k_output": 0.015,   # $/1K output tokens
}

SPECIALIZED_MODELS = {
    "reasoning": {
        "name": REASONING_MODEL,
        "label": "Nemotron Super (Reasoning)",
        "cost_per_1k_input": 0.005,
        "cost_per_1k_output": 0.015,
    },
    "safety": {
        "name": SAFETY_MODEL,
        "label": "Nemotron Nano (Safety)",
        "cost_per_1k_input": 0.001,
        "cost_per_1k_output": 0.002,
    },
    "embedding": {
        "name": EMBED_MODEL,
        "label": "Embed VL (Retrieval)",
        "cost_per_1k_input": 0.0003,
        "cost_per_1k_output": 0.0,
    },
}


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    approach: str
    task: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Simulated agent workflow tasks
# ---------------------------------------------------------------------------
AGENT_WORKFLOW = [
    {
        "task": "reasoning",
        "description": "Plan multi-step response",
        "prompt": (
            "A user is planning a trip to Tokyo. They want recommendations for "
            "3 days including cultural sites, food, and transportation. Create a "
            "structured day-by-day itinerary with reasoning for each choice."
        ),
    },
    {
        "task": "safety",
        "description": "Check input safety",
        "prompt": (
            "Classify whether this user request is safe to process: "
            "'I want to plan a trip to Tokyo and learn about Japanese culture.'"
        ),
    },
    {
        "task": "embedding",
        "description": "Embed query for retrieval",
        "prompt": "Tokyo travel recommendations cultural sites food transportation",
    },
    {
        "task": "safety",
        "description": "Check output safety",
        "prompt": (
            "Classify whether this AI response is safe to return: "
            "'Day 1: Visit Senso-ji Temple in Asakusa, try street food at "
            "Nakamise-dori, take the metro to Shibuya for evening dining.'"
        ),
    },
    {
        "task": "reasoning",
        "description": "Generate final answer",
        "prompt": (
            "Based on the following context, generate a friendly and detailed "
            "travel recommendation for Tokyo:\n\n"
            "Context: Top-rated cultural sites include Senso-ji, Meiji Shrine, "
            "and the Imperial Palace. Best food areas: Tsukiji Outer Market, "
            "Shinjuku Memory Lane, Shibuya. Transit: Get a Suica card for "
            "metro access."
        ),
    },
]


def run_monolith_workflow(client: OpenAI) -> list[BenchmarkResult]:
    """Run the entire workflow through a single large model."""
    results = []
    model_config = MONOLITH_MODEL

    for step in AGENT_WORKFLOW:
        start = time.perf_counter()

        if step["task"] == "embedding":
            response = client.embeddings.create(
                model=SPECIALIZED_MODELS["embedding"]["name"],
                input=[step["prompt"]],
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            elapsed = (time.perf_counter() - start) * 1000
            input_tokens = response.usage.total_tokens if response.usage else 20
            output_tokens = 0
            cost_config = SPECIALIZED_MODELS["embedding"]
        else:
            # Monolith: use the big model for everything
            response = client.chat.completions.create(
                model=model_config["name"],
                messages=[{"role": "user", "content": step["prompt"]}],
                temperature=0.3,
                max_tokens=512,
            )
            elapsed = (time.perf_counter() - start) * 1000
            input_tokens = response.usage.prompt_tokens if response.usage else 100
            output_tokens = response.usage.completion_tokens if response.usage else 200
            cost_config = model_config

        cost = (
            input_tokens / 1000 * cost_config["cost_per_1k_input"]
            + output_tokens / 1000 * cost_config["cost_per_1k_output"]
        )

        results.append(BenchmarkResult(
            approach="monolith",
            task=step["description"],
            model=cost_config.get("label", cost_config["name"]),
            latency_ms=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
        ))

    return results


def run_specialized_workflow(client: OpenAI) -> list[BenchmarkResult]:
    """Run each task through its purpose-built specialist model."""
    results = []

    for step in AGENT_WORKFLOW:
        task_type = step["task"]
        model_config = SPECIALIZED_MODELS[task_type]
        start = time.perf_counter()

        if task_type == "embedding":
            response = client.embeddings.create(
                model=model_config["name"],
                input=[step["prompt"]],
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"},
            )
            elapsed = (time.perf_counter() - start) * 1000
            input_tokens = response.usage.total_tokens if response.usage else 20
            output_tokens = 0
        else:
            response = client.chat.completions.create(
                model=model_config["name"],
                messages=[{"role": "user", "content": step["prompt"]}],
                temperature=0.3,
                max_tokens=512 if task_type == "reasoning" else 128,
            )
            elapsed = (time.perf_counter() - start) * 1000
            input_tokens = response.usage.prompt_tokens if response.usage else 100
            output_tokens = response.usage.completion_tokens if response.usage else 200

        cost = (
            input_tokens / 1000 * model_config["cost_per_1k_input"]
            + output_tokens / 1000 * model_config["cost_per_1k_output"]
        )

        results.append(BenchmarkResult(
            approach="specialized",
            task=step["description"],
            model=model_config["label"],
            latency_ms=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results(monolith: list[BenchmarkResult], specialized: list[BenchmarkResult]):
    """Print a comparison table of both approaches."""
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: Monolith vs. Specialized Stack")
    print("=" * 80)

    # Per-step comparison
    print(f"\n{'Task':<25} {'Monolith':<20} {'Specialized':<20} {'Savings':<15}")
    print("-" * 80)

    total_mono_cost = 0
    total_spec_cost = 0
    total_mono_latency = 0
    total_spec_latency = 0

    for m, s in zip(monolith, specialized):
        saving = ((m.estimated_cost - s.estimated_cost) / max(m.estimated_cost, 0.0001)) * 100
        print(
            f"{m.task:<25} "
            f"${m.estimated_cost:.5f} {m.latency_ms:>6.0f}ms  "
            f"${s.estimated_cost:.5f} {s.latency_ms:>6.0f}ms  "
            f"{saving:>5.1f}% cheaper"
        )
        total_mono_cost += m.estimated_cost
        total_spec_cost += s.estimated_cost
        total_mono_latency += m.latency_ms
        total_spec_latency += s.latency_ms

    print("-" * 80)
    total_saving = ((total_mono_cost - total_spec_cost) / max(total_mono_cost, 0.0001)) * 100
    latency_saving = ((total_mono_latency - total_spec_latency) / max(total_mono_latency, 1)) * 100
    print(
        f"{'TOTAL':<25} "
        f"${total_mono_cost:.5f} {total_mono_latency:>6.0f}ms  "
        f"${total_spec_cost:.5f} {total_spec_latency:>6.0f}ms  "
        f"{total_saving:>5.1f}% cheaper"
    )

    # Scale projections
    print("\n" + "=" * 80)
    print("  SCALE PROJECTIONS (per day)")
    print("=" * 80)
    print(f"\n{'Daily Interactions':<25} {'Monolith Cost':<18} {'Specialized Cost':<18} {'Daily Savings':<15}")
    print("-" * 76)

    for daily_interactions in [1_000, 10_000, 100_000, 1_000_000]:
        mono_daily = total_mono_cost * daily_interactions
        spec_daily = total_spec_cost * daily_interactions
        daily_saving = mono_daily - spec_daily
        print(
            f"{daily_interactions:>20,}   "
            f"${mono_daily:>12,.2f}   "
            f"${spec_daily:>12,.2f}   "
            f"${daily_saving:>10,.2f}"
        )

    print("\n" + "=" * 80)
    print("  KEY TAKEAWAYS")
    print("=" * 80)
    print(f"  Cost reduction:    {total_saving:.1f}% per agent workflow")
    print(f"  Latency change:    {latency_saving:+.1f}%")
    print(f"  Annual savings at 100K/day: ${(total_mono_cost - total_spec_cost) * 100_000 * 365:,.0f}")
    print()
    print("  The specialized stack uses purpose-built models for each task:")
    print("    • Reasoning:  12B active params (not 400B+)")
    print("    • Safety:     8B params (dedicated classifier)")
    print("    • Embeddings: 1.7B params (retrieval specialist)")
    print("  Each model is RIGHT-SIZED — no wasted compute.\n")


def main():
    print("=" * 80)
    print("  COST & LATENCY BENCHMARK")
    print("  Monolith (Single Large Model) vs. Specialized Nemotron Stack")
    print("=" * 80)
    print("\n  Running agent workflow through both approaches...\n")

    client = get_client()

    print("  [1/2] Running MONOLITH workflow (all tasks → large model)...")
    monolith_results = run_monolith_workflow(client)

    print("  [2/2] Running SPECIALIZED workflow (each task → right-sized model)...")
    specialized_results = run_specialized_workflow(client)

    print_results(monolith_results, specialized_results)


if __name__ == "__main__":
    main()
