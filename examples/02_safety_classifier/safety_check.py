"""
Safety-as-a-Service — Dedicated Content Safety Classification
=============================================================
Demonstrates using a lightweight, specialized safety model as an
independent guardrail rather than prompting a large reasoning model.

The key insight: a 4B-parameter safety specialist runs in ~5ms and
catches what prompt-based safety in a 400B model often misses.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python safety_check.py
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
    SAFETY_TAXONOMY,
)


@dataclass
class SafetyResult:
    """Result from the safety classification pipeline."""
    is_safe: bool
    categories: list[str]
    confidence: float
    latency_ms: float
    model: str


@dataclass
class AgentResponse:
    """Complete agent response with safety metadata."""
    content: str
    safety: SafetyResult
    was_filtered: bool
    reasoning_latency_ms: float
    total_latency_ms: float


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def classify_safety(client: OpenAI, content: str) -> SafetyResult:
    """
    Run content through the dedicated safety classifier.

    This is the 'safety-as-a-service' pattern: instead of asking a large
    reasoning model "is this safe?", we use a purpose-built classifier
    that is faster, cheaper, and more accurate at this specific task.
    """
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a content safety classifier. Analyze the input and "
                    "respond with ONLY a JSON object (no markdown, no explanation):\n"
                    '{"is_safe": true/false, "categories": ["category1"], '
                    '"confidence": 0.95}\n\n'
                    f"Valid categories: {', '.join(SAFETY_TAXONOMY)}\n"
                    "If content is safe, return empty categories list."
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    raw = response.choices[0].message.content.strip()

    # Parse the safety classification
    try:
        # Handle potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json
        parsed = json.loads(raw)
        return SafetyResult(
            is_safe=parsed.get("is_safe", True),
            categories=parsed.get("categories", []),
            confidence=parsed.get("confidence", 0.0),
            latency_ms=elapsed_ms,
            model=SAFETY_MODEL,
        )
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, default to safe (fail-open)
        # In production, you might want fail-closed instead
        return SafetyResult(
            is_safe=True,
            categories=[],
            confidence=0.0,
            latency_ms=elapsed_ms,
            model=SAFETY_MODEL,
        )


def generate_response(client: OpenAI, query: str) -> tuple[str, float]:
    """Generate a response using the reasoning model."""
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond concisely.",
            },
            {"role": "user", "content": query},
        ],
        temperature=0.6,
        max_tokens=512,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    return response.choices[0].message.content, elapsed_ms


def safe_agent_respond(client: OpenAI, query: str) -> AgentResponse:
    """
    Complete agent pipeline with dedicated safety guardrails.

    Pipeline:
    1. Check INPUT safety  (lightweight 4B classifier)
    2. Generate response   (reasoning model)
    3. Check OUTPUT safety (same lightweight classifier)

    The safety checks add minimal latency because they use a
    purpose-built small model, not the reasoning heavyweight.
    """
    total_start = time.perf_counter()

    # Step 1: Input safety check
    print("  [1/3] Checking input safety...")
    input_safety = classify_safety(client, query)
    print(f"        Safe={input_safety.is_safe} ({input_safety.latency_ms:.0f}ms)")

    if not input_safety.is_safe:
        total_ms = (time.perf_counter() - total_start) * 1000
        return AgentResponse(
            content="I can't process this request as it was flagged for: "
                    + ", ".join(input_safety.categories),
            safety=input_safety,
            was_filtered=True,
            reasoning_latency_ms=0,
            total_latency_ms=total_ms,
        )

    # Step 2: Generate response with reasoning model
    print("  [2/3] Generating response...")
    content, reasoning_ms = generate_response(client, query)
    print(f"        Generated ({reasoning_ms:.0f}ms)")

    # Step 3: Output safety check
    print("  [3/3] Checking output safety...")
    output_safety = classify_safety(client, content)
    print(f"        Safe={output_safety.is_safe} ({output_safety.latency_ms:.0f}ms)")

    was_filtered = False
    if not output_safety.is_safe:
        content = (
            "The generated response was filtered for safety. "
            f"Flagged categories: {', '.join(output_safety.categories)}"
        )
        was_filtered = True

    total_ms = (time.perf_counter() - total_start) * 1000
    return AgentResponse(
        content=content,
        safety=output_safety,
        was_filtered=was_filtered,
        reasoning_latency_ms=reasoning_ms,
        total_latency_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    "Explain the benefits of renewable energy sources for developing nations.",
    "What are the key principles of the NVIDIA Nemotron 3 architecture?",
    "How do specialized AI models reduce production costs compared to monolithic models?",
]


def main():
    print("=" * 60)
    print("  SAFETY-AS-A-SERVICE: Dedicated Safety Classification")
    print("  Using lightweight 4B model as inline guardrail")
    print("=" * 60)

    client = get_client()

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n--- Query {i}: {query[:60]}...")
        result = safe_agent_respond(client, query)

        print(f"\n  Response: {result.content[:150]}...")
        print(f"  Filtered: {result.was_filtered}")
        print(f"  Safety confidence: {result.safety.confidence}")
        print(f"  Reasoning latency: {result.reasoning_latency_ms:.0f}ms")
        print(f"  Total latency:     {result.total_latency_ms:.0f}ms")
        safety_overhead = result.total_latency_ms - result.reasoning_latency_ms
        print(f"  Safety overhead:   {safety_overhead:.0f}ms "
              f"({safety_overhead / max(result.total_latency_ms, 1) * 100:.1f}% of total)")

    print("\n" + "=" * 60)
    print("  KEY INSIGHT")
    print("=" * 60)
    print("  The dedicated safety model adds minimal latency overhead")
    print("  compared to embedding safety checks in the reasoning prompt.")
    print("  It's faster, cheaper, and more reliable as a separate service.")
    print()


if __name__ == "__main__":
    main()
