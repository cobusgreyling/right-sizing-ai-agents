"""
Batch Processing — Throughput Gains with Parallel Specialized Models
====================================================================
Demonstrates the throughput advantage of the specialized model stack
when processing many queries concurrently.  Instead of queuing
everything through a single large model, queries are classified by
type and dispatched in parallel to purpose-built models.

Author: Cobus Greyling

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python batch_throughput.py
"""

import os
import sys
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    SAFETY_MODEL,
    EMBED_MODEL,
    classify_intent,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BatchItem:
    """A single query in the batch."""
    query: str
    category: str
    index: int


@dataclass
class BatchResult:
    """Result of processing one batch item."""
    index: int
    query: str
    category: str
    model: str
    params: str
    latency_ms: float
    tokens: int
    response: str


@dataclass
class BatchReport:
    """Aggregate report for a batch run."""
    approach: str
    total_wall_time_ms: float
    total_model_time_ms: float
    results: list[BatchResult] = field(default_factory=list)

    @property
    def throughput_qps(self) -> float:
        """Queries per second based on wall clock time."""
        if self.total_wall_time_ms <= 0:
            return 0
        return len(self.results) / (self.total_wall_time_ms / 1000)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0
        return sum(r.latency_ms for r in self.results) / len(self.results)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "reasoning": {"model": REASONING_MODEL, "params": "12B active", "max_tokens": 256},
    "safety": {"model": SAFETY_MODEL, "params": "8B", "max_tokens": 128},
    "embedding": {"model": REASONING_MODEL, "params": "12B active", "max_tokens": 256},
    "general": {"model": REASONING_MODEL, "params": "12B active", "max_tokens": 256},
}


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Simulated batch of diverse queries
# ---------------------------------------------------------------------------
BATCH_QUERIES = [
    # Reasoning tasks
    "Explain why transformers replaced RNNs for most NLP tasks.",
    "Compare supervised learning and reinforcement learning for robotics.",
    "Analyze the trade-offs between SQL and NoSQL databases for analytics.",

    # Safety checks
    "Check content safety: 'Join our community cooking class this Saturday!'",
    "Classify safety: 'The new product launch was a massive success.'",
    "Is this safe to publish? 'Tips for growing tomatoes in small gardens.'",

    # General / simple
    "What is the boiling point of water?",
    "Name three programming languages used in data science.",
    "What does API stand for?",

    # More reasoning
    "How does gradient descent optimize a neural network? Step by step.",
    "Evaluate whether serverless is better than containers for event processing.",
    "Design a retry strategy for a distributed message queue.",
]


def classify_batch(queries: list[str]) -> list[BatchItem]:
    """Classify all queries in the batch."""
    items = []
    for i, query in enumerate(queries):
        category = classify_intent(query)
        items.append(BatchItem(query=query, category=category, index=i))
    return items


# ---------------------------------------------------------------------------
# Sequential (monolith) processing
# ---------------------------------------------------------------------------
def process_monolith_single(client: OpenAI, item: BatchItem) -> BatchResult:
    """Process one query through the monolith model."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": item.query}],
        temperature=0.3,
        max_tokens=256,
    )
    latency = (time.perf_counter() - start) * 1000
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

    return BatchResult(
        index=item.index,
        query=item.query,
        category=item.category,
        model=REASONING_MODEL,
        params="12B active",
        latency_ms=latency,
        tokens=tokens,
        response=response.choices[0].message.content[:80],
    )


def run_monolith_sequential(client: OpenAI,
                            items: list[BatchItem]) -> BatchReport:
    """Process all items sequentially through the monolith model."""
    report = BatchReport(approach="monolith_sequential", total_wall_time_ms=0,
                         total_model_time_ms=0)
    wall_start = time.perf_counter()

    for item in items:
        result = process_monolith_single(client, item)
        report.results.append(result)
        report.total_model_time_ms += result.latency_ms

    report.total_wall_time_ms = (time.perf_counter() - wall_start) * 1000
    return report


# ---------------------------------------------------------------------------
# Parallel specialized processing
# ---------------------------------------------------------------------------
def process_specialized_single(client: OpenAI, item: BatchItem) -> BatchResult:
    """Process one query through the right-sized specialized model."""
    config = MODEL_MAP.get(item.category, MODEL_MAP["general"])

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": item.query}],
        temperature=0.3,
        max_tokens=config["max_tokens"],
    )
    latency = (time.perf_counter() - start) * 1000
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

    return BatchResult(
        index=item.index,
        query=item.query,
        category=item.category,
        model=config["model"],
        params=config["params"],
        latency_ms=latency,
        tokens=tokens,
        response=response.choices[0].message.content[:80],
    )


def run_specialized_parallel(client: OpenAI, items: list[BatchItem],
                             max_workers: int = 4) -> BatchReport:
    """Process items in parallel, routing each to its specialized model."""
    report = BatchReport(approach="specialized_parallel", total_wall_time_ms=0,
                         total_model_time_ms=0)
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_specialized_single, client, item): item
            for item in items
        }
        for future in as_completed(futures):
            result = future.result()
            report.results.append(result)
            report.total_model_time_ms += result.latency_ms

    # Sort results back to original order
    report.results.sort(key=lambda r: r.index)
    report.total_wall_time_ms = (time.perf_counter() - wall_start) * 1000
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(mono: BatchReport, parallel: BatchReport):
    """Print comparison of sequential monolith vs. parallel specialized."""
    print("\n" + "=" * 80)
    print("  BATCH PROCESSING BENCHMARK")
    print(f"  {len(mono.results)} queries — Sequential Monolith vs. Parallel Specialized")
    print("=" * 80)

    # Per-query detail
    print(f"\n  {'#':<4} {'Category':<12} {'Mono Model':<12} {'Mono ms':>8} "
          f"{'Spec Model':<12} {'Spec ms':>8}")
    print("  " + "-" * 65)

    for m, s in zip(mono.results, parallel.results):
        m_params = "12B active"
        s_params = s.params
        print(
            f"  {m.index:<4} {m.category:<12} {m_params:<12} {m.latency_ms:>7.0f}ms "
            f"{s_params:<12} {s.latency_ms:>7.0f}ms"
        )

    # Aggregate comparison
    print(f"\n{'='*80}")
    print("  THROUGHPUT COMPARISON")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<30} {'Monolith':>18} {'Specialized':>18}")
    print("  " + "-" * 66)

    print(f"  {'Wall clock time':<30} {mono.total_wall_time_ms:>17.0f}ms "
          f"{parallel.total_wall_time_ms:>17.0f}ms")
    print(f"  {'Sum of model latencies':<30} {mono.total_model_time_ms:>17.0f}ms "
          f"{parallel.total_model_time_ms:>17.0f}ms")
    print(f"  {'Avg latency per query':<30} {mono.avg_latency_ms:>17.0f}ms "
          f"{parallel.avg_latency_ms:>17.0f}ms")
    print(f"  {'Throughput (queries/sec)':<30} {mono.throughput_qps:>18.1f} "
          f"{parallel.throughput_qps:>18.1f}")

    speedup = mono.total_wall_time_ms / max(parallel.total_wall_time_ms, 1)
    print(f"\n  Speedup: {speedup:.1f}x wall-clock time improvement")

    # Category breakdown
    print(f"\n  CATEGORY BREAKDOWN (parallel specialized)")
    print("  " + "-" * 50)
    categories = {}
    for r in parallel.results:
        cat = categories.setdefault(r.category, {"count": 0, "total_ms": 0})
        cat["count"] += 1
        cat["total_ms"] += r.latency_ms

    for cat, stats in sorted(categories.items()):
        avg = stats["total_ms"] / stats["count"]
        print(f"  {cat:<12} {stats['count']} queries  avg {avg:.0f}ms")

    print()
    print("  KEY INSIGHT: Parallel dispatch to specialized models achieves")
    print("  higher throughput because:")
    print("    1. Smaller models respond faster per query")
    print("    2. Different model endpoints can serve requests concurrently")
    print("    3. No single bottleneck — load is distributed across model tiers")
    print()


def main():
    print("=" * 80)
    print("  BATCH THROUGHPUT BENCHMARK")
    print("  Sequential Monolith vs. Parallel Specialized Stack")
    print("=" * 80)

    client = get_client()

    # Classify all queries
    items = classify_batch(BATCH_QUERIES)
    print(f"\n  Batch: {len(items)} queries")
    for item in items:
        print(f"    [{item.category:<10}] {item.query[:55]}...")

    # Run sequential monolith
    print(f"\n  [1/2] Running SEQUENTIAL monolith ({len(items)} queries, one at a time)...")
    mono_report = run_monolith_sequential(client, items)

    # Run parallel specialized
    print(f"  [2/2] Running PARALLEL specialized ({len(items)} queries, 4 workers)...")
    parallel_report = run_specialized_parallel(client, items, max_workers=4)

    print_report(mono_report, parallel_report)


if __name__ == "__main__":
    main()
