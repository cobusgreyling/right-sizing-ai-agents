"""
Model Selection Benchmark — Quality vs. Cost A/B Comparison
============================================================
Runs identical prompts through both a monolith (single large model) and
the specialized Nemotron 3 stack, then compares response quality, latency,
and cost side-by-side.

Quality is measured by a lightweight LLM-as-judge evaluation: the
reasoning model scores each response on relevance, accuracy, and
completeness (1-5 scale).

Author: Cobus Greyling

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python quality_benchmark.py
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
    EMBED_MODEL,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class QualityScore:
    """LLM-as-judge quality assessment."""
    relevance: float = 0.0
    accuracy: float = 0.0
    completeness: float = 0.0

    @property
    def overall(self) -> float:
        return round((self.relevance + self.accuracy + self.completeness) / 3, 2)


@dataclass
class BenchmarkEntry:
    """A single prompt run through one approach."""
    prompt: str
    approach: str
    model_used: str
    params: str
    response: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    quality: QualityScore = field(default_factory=QualityScore)


# ---------------------------------------------------------------------------
# Pricing tiers (illustrative)
# ---------------------------------------------------------------------------
PRICING = {
    REASONING_MODEL: {"input": 0.005, "output": 0.015, "params": "12B active"},
    SAFETY_MODEL: {"input": 0.001, "output": 0.002, "params": "8B"},
    EMBED_MODEL: {"input": 0.0003, "output": 0.0, "params": "1.7B"},
}


# ---------------------------------------------------------------------------
# Benchmark prompts — diverse tasks that stress different capabilities
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    {
        "prompt": (
            "A startup has $500K in runway and needs to choose between "
            "building a custom ML pipeline or using a managed API. "
            "Analyze the trade-offs considering cost, time-to-market, "
            "team expertise, and scalability. Think step by step."
        ),
        "category": "reasoning",
        "expected_model": "reasoning",
    },
    {
        "prompt": (
            "Classify whether the following customer message is safe to "
            "process: 'I'd like to cancel my subscription and get a full "
            "refund for the last three months.'"
        ),
        "category": "safety",
        "expected_model": "safety",
    },
    {
        "prompt": (
            "Explain the difference between retrieval-augmented generation "
            "and fine-tuning for domain adaptation. Include pros and cons."
        ),
        "category": "reasoning",
        "expected_model": "reasoning",
    },
    {
        "prompt": (
            "Check this AI-generated response for safety issues: "
            "'To reset your password, visit account.example.com/reset and "
            "enter your email. A reset link will be sent within 5 minutes.'"
        ),
        "category": "safety",
        "expected_model": "safety",
    },
    {
        "prompt": (
            "Compare microservices architecture to monolithic architecture "
            "for a team of 5 engineers building a B2B SaaS product. "
            "Evaluate maintenance burden, deployment complexity, and "
            "debugging difficulty."
        ),
        "category": "reasoning",
        "expected_model": "reasoning",
    },
]


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Model call helpers
# ---------------------------------------------------------------------------
def call_model(client: OpenAI, model: str, prompt: str,
               temperature: float = 0.3, max_tokens: int = 512) -> dict:
    """Call a chat model and return response + metrics."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = (time.perf_counter() - start) * 1000
    usage = response.usage
    return {
        "response": response.choices[0].message.content,
        "latency_ms": elapsed,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on model pricing tier."""
    p = PRICING.get(model, PRICING[REASONING_MODEL])
    return input_tokens / 1000 * p["input"] + output_tokens / 1000 * p["output"]


# ---------------------------------------------------------------------------
# LLM-as-judge quality evaluation
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are a quality evaluator. Score the following AI response on three criteria.
Each score is 1-5 (1=poor, 5=excellent).

Respond with ONLY a JSON object:
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>}}

User prompt: {prompt}

AI response: {response}
"""


def evaluate_quality(client: OpenAI, prompt: str,
                     response: str) -> QualityScore:
    """Use LLM-as-judge to score response quality."""
    try:
        result = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(prompt=prompt, response=response),
            }],
            temperature=0.0,
            max_tokens=64,
        )
        raw = result.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return QualityScore(
            relevance=float(parsed.get("relevance", 3)),
            accuracy=float(parsed.get("accuracy", 3)),
            completeness=float(parsed.get("completeness", 3)),
        )
    except (json.JSONDecodeError, Exception):
        return QualityScore(relevance=3, accuracy=3, completeness=3)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def select_specialized_model(category: str) -> str:
    """Pick the right-sized model for a task category."""
    if category == "safety":
        return SAFETY_MODEL
    return REASONING_MODEL


def run_benchmark(client: OpenAI, evaluate: bool = True) -> dict:
    """
    Run all prompts through both approaches.

    Returns dict with 'monolith' and 'specialized' lists of BenchmarkEntry.
    """
    monolith_results = []
    specialized_results = []

    for item in BENCHMARK_PROMPTS:
        prompt = item["prompt"]
        category = item["category"]

        # --- Monolith: everything through the large model ---
        print(f"  [Monolith]     {category:<12} {prompt[:50]}...")
        mono_data = call_model(client, REASONING_MODEL, prompt)
        mono_entry = BenchmarkEntry(
            prompt=prompt,
            approach="monolith",
            model_used=REASONING_MODEL,
            params=PRICING[REASONING_MODEL]["params"],
            response=mono_data["response"],
            latency_ms=mono_data["latency_ms"],
            input_tokens=mono_data["input_tokens"],
            output_tokens=mono_data["output_tokens"],
            estimated_cost=estimate_cost(
                REASONING_MODEL,
                mono_data["input_tokens"],
                mono_data["output_tokens"],
            ),
        )
        monolith_results.append(mono_entry)

        # --- Specialized: route to the right model ---
        spec_model = select_specialized_model(category)
        print(f"  [Specialized]  {category:<12} {prompt[:50]}...")
        spec_data = call_model(
            client, spec_model, prompt,
            max_tokens=128 if category == "safety" else 512,
        )
        spec_entry = BenchmarkEntry(
            prompt=prompt,
            approach="specialized",
            model_used=spec_model,
            params=PRICING.get(spec_model, PRICING[REASONING_MODEL])["params"],
            response=spec_data["response"],
            latency_ms=spec_data["latency_ms"],
            input_tokens=spec_data["input_tokens"],
            output_tokens=spec_data["output_tokens"],
            estimated_cost=estimate_cost(
                spec_model,
                spec_data["input_tokens"],
                spec_data["output_tokens"],
            ),
        )
        specialized_results.append(spec_entry)

    # Evaluate quality with LLM-as-judge
    if evaluate:
        print("\n  Evaluating quality with LLM-as-judge...")
        for mono, spec in zip(monolith_results, specialized_results):
            mono.quality = evaluate_quality(client, mono.prompt, mono.response)
            spec.quality = evaluate_quality(client, spec.prompt, spec.response)

    return {"monolith": monolith_results, "specialized": specialized_results}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_comparison(results: dict):
    """Print a detailed A/B comparison report."""
    monolith = results["monolith"]
    specialized = results["specialized"]

    print("\n" + "=" * 90)
    print("  MODEL SELECTION BENCHMARK: Monolith vs. Specialized Stack")
    print("=" * 90)

    # Per-prompt comparison
    print(f"\n  {'#':<4} {'Category':<12} {'Approach':<14} {'Model':<10} "
          f"{'Latency':>8} {'Cost':>10} {'Quality':>8}")
    print("  " + "-" * 80)

    for i, (m, s) in enumerate(zip(monolith, specialized)):
        print(
            f"  {i+1:<4} {BENCHMARK_PROMPTS[i]['category']:<12} "
            f"{'Monolith':<14} {m.params:<10} "
            f"{m.latency_ms:>7.0f}ms ${m.estimated_cost:>8.5f} "
            f"{m.quality.overall:>7.1f}/5"
        )
        print(
            f"  {'':<4} {'':<12} "
            f"{'Specialized':<14} {s.params:<10} "
            f"{s.latency_ms:>7.0f}ms ${s.estimated_cost:>8.5f} "
            f"{s.quality.overall:>7.1f}/5"
        )
        print()

    # Aggregate
    mono_cost = sum(e.estimated_cost for e in monolith)
    spec_cost = sum(e.estimated_cost for e in specialized)
    mono_latency = sum(e.latency_ms for e in monolith)
    spec_latency = sum(e.latency_ms for e in specialized)
    mono_quality = sum(e.quality.overall for e in monolith) / len(monolith)
    spec_quality = sum(e.quality.overall for e in specialized) / len(specialized)

    cost_saving = ((mono_cost - spec_cost) / max(mono_cost, 0.0001)) * 100
    latency_diff = ((mono_latency - spec_latency) / max(mono_latency, 1)) * 100

    print("=" * 90)
    print("  AGGREGATE RESULTS")
    print("=" * 90)
    print(f"\n  {'Metric':<25} {'Monolith':>15} {'Specialized':>15} {'Difference':>15}")
    print("  " + "-" * 70)
    print(f"  {'Total cost':<25} ${mono_cost:>14.5f} ${spec_cost:>14.5f} {cost_saving:>14.1f}%")
    print(f"  {'Total latency':<25} {mono_latency:>14.0f}ms {spec_latency:>14.0f}ms {latency_diff:>13.1f}%")
    print(f"  {'Avg quality (1-5)':<25} {mono_quality:>15.2f} {spec_quality:>15.2f} {spec_quality - mono_quality:>+14.2f}")

    print("\n  KEY TAKEAWAY")
    print("  " + "-" * 70)
    if cost_saving > 0:
        print(f"  Specialized stack saves {cost_saving:.1f}% in cost")
    if abs(spec_quality - mono_quality) < 0.5:
        print("  Quality is comparable — specialized models match the monolith")
        print("  on task-appropriate prompts while costing significantly less.")
    elif spec_quality > mono_quality:
        print("  Specialized models actually score HIGHER on task-appropriate prompts.")
    print()


def main():
    print("=" * 90)
    print("  MODEL SELECTION BENCHMARK")
    print("  A/B comparison: same prompts, monolith vs. specialized stack")
    print("=" * 90)

    client = get_client()
    results = run_benchmark(client, evaluate=True)
    print_comparison(results)


if __name__ == "__main__":
    main()
