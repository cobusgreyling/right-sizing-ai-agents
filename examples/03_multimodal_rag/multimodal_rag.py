"""
Multimodal RAG with Specialized Embed + Rerank Pipeline
========================================================
Demonstrates the two-stage retrieval pattern using purpose-built
NVIDIA models: lightweight embedding for fast recall, followed by
cross-encoder reranking for precision, and finally reasoning for
answer generation.

Each stage uses the MINIMUM model necessary for its task.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key-here"
    python multimodal_rag.py
"""

import os
import sys
import time
import numpy as np
import requests
from dataclasses import dataclass, field
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    EMBED_MODEL,
    RERANK_MODEL,
)


@dataclass
class Document:
    """A document in our knowledge base."""
    id: str
    title: str
    content: str
    category: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""
    document: Document
    similarity_score: float
    stage: str  # "embed" or "rerank"


# ---------------------------------------------------------------------------
# Sample knowledge base — NVIDIA Nemotron 3 documentation
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    Document(
        id="doc_1",
        title="Nemotron 3 Super Architecture",
        content=(
            "Nemotron 3 Super is a hybrid Mamba-Transformer mixture-of-experts "
            "model with 120B total parameters but only 12B active per inference "
            "pass. It features a 1M-token context window, configurable thinking "
            "budget for bounded chain-of-thought reasoning, and multi-token "
            "prediction for faster generation. Trained across 10+ reinforcement "
            "learning environments."
        ),
        category="architecture",
    ),
    Document(
        id="doc_2",
        title="Content Safety Classification",
        content=(
            "Nemotron 3 Content Safety is a 4B-parameter multimodal classifier "
            "built on Gemma-3-4B. It detects unsafe content across text and "
            "images using a 23-category taxonomy covering hate, harassment, "
            "violence, sexual content, and unauthorized advice. Supports "
            "12 languages with zero-shot generalization."
        ),
        category="safety",
    ),
    Document(
        id="doc_3",
        title="VoiceChat End-to-End Speech",
        content=(
            "Nemotron 3 VoiceChat is a 12B-parameter end-to-end speech model "
            "with sub-300ms latency processing 80ms audio chunks. It uses a "
            "unified streaming LLM architecture eliminating cascaded "
            "ASR-LLM-TTS pipelines, enabling full-duplex interruptible "
            "conversations."
        ),
        category="voice",
    ),
    Document(
        id="doc_4",
        title="Llama Nemotron Embed VL",
        content=(
            "A 1.7B-parameter dense embedding model that encodes page images "
            "and text into single-dimensional vectors. Supports Matryoshka "
            "embeddings for flexible dimensionality. Built on NVIDIA Eagle "
            "vision-language model with Llama 3.2 1B backbone and SigLip2 "
            "400M vision encoder."
        ),
        category="retrieval",
    ),
    Document(
        id="doc_5",
        title="Production Deployment Guide",
        content=(
            "For production deployment, the Nemotron 3 stack emphasizes "
            "efficiency as a requirement: real agents make dozens or hundreds "
            "of model calls per task, necessitating right-sized, "
            "throughput-optimized models. NVFP4 precision on Blackwell GPUs "
            "achieves 5x higher throughput than previous generation."
        ),
        category="deployment",
    ),
    Document(
        id="doc_6",
        title="NeMo Agent Toolkit",
        content=(
            "An open-source framework for end-to-end profiling and optimization "
            "of agentic systems. Compatible with LangChain, AutoGen, and AWS "
            "Strands without code changes. Provides visibility into latency "
            "bottlenecks, token costs, and orchestration overhead."
        ),
        category="tooling",
    ),
    Document(
        id="doc_7",
        title="Model Licensing and Openness",
        content=(
            "All Nemotron 3 models are released under permissive NVIDIA open "
            "model licenses, enabling teams to fine-tune behaviors and deploy "
            "securely on-premises or in cloud environments. Models available "
            "on Hugging Face with preview access through build.nvidia.com."
        ),
        category="licensing",
    ),
    Document(
        id="doc_8",
        title="Benchmark Performance",
        content=(
            "Nemotron 3 Super ranks among top open-weight models under 250B "
            "on the Artificial Analysis Intelligence Index, landing in the "
            "'most attractive' efficiency quadrant combining intelligence "
            "with throughput. Content Safety achieves ~84% accuracy on "
            "multimodal multilingual benchmarks."
        ),
        category="benchmarks",
    ),
]


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set NVIDIA_API_KEY environment variable. "
            "Get one free at https://build.nvidia.com"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


# ---------------------------------------------------------------------------
# Stage 1: Embed (1.7B model — fast vector similarity)
# ---------------------------------------------------------------------------
def embed_documents(client: OpenAI, documents: list[Document]) -> list[Document]:
    """Embed all documents using the lightweight embedding model."""
    print(f"\n  [EMBED] Encoding {len(documents)} documents with {EMBED_MODEL}")
    start = time.perf_counter()

    texts = [f"{doc.title}: {doc.content}" for doc in documents]

    # Batch embed for efficiency
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    )

    for i, doc in enumerate(documents):
        doc.embedding = response.data[i].embedding

    elapsed = (time.perf_counter() - start) * 1000
    dim = len(documents[0].embedding)
    print(f"  [EMBED] Done: {dim}-dim vectors in {elapsed:.0f}ms")
    return documents


def embed_query(client: OpenAI, query: str) -> list[float]:
    """Embed a query using the same embedding model."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"},
    )
    return response.data[0].embedding


def retrieve_by_embedding(
    client: OpenAI,
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Stage 1: Fast vector similarity retrieval."""
    print(f"\n  [STAGE 1] Vector similarity search (top-{top_k})")
    start = time.perf_counter()

    query_embedding = embed_query(client, query)

    # Score all documents by cosine similarity
    scored = []
    for doc in documents:
        if doc.embedding:
            score = cosine_similarity(query_embedding, doc.embedding)
            scored.append(RetrievalResult(
                document=doc, similarity_score=score, stage="embed"
            ))

    # Sort by similarity and take top-k
    scored.sort(key=lambda r: r.similarity_score, reverse=True)
    results = scored[:top_k]

    elapsed = (time.perf_counter() - start) * 1000
    print(f"  [STAGE 1] Retrieved {len(results)} candidates in {elapsed:.0f}ms")
    for r in results:
        print(f"            {r.similarity_score:.4f} — {r.document.title}")

    return results


# ---------------------------------------------------------------------------
# Stage 2: Rerank (dedicated 1.7B cross-encoder via NVIDIA Rerank API)
# ---------------------------------------------------------------------------
def rerank_results(
    client: OpenAI,
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """
    Stage 2: Cross-encoder reranking for precision.

    Uses the NVIDIA Rerank API with a dedicated 1.7B cross-encoder model,
    which scores query-document relevance far more accurately than
    vector similarity alone.
    """
    print(f"\n  [STAGE 2] Cross-encoder reranking with {RERANK_MODEL} (top-{top_k})")
    start = time.perf_counter()

    # Build passages for the rerank API
    passages = [
        {"text": f"{r.document.title}: {r.document.content}"}
        for r in candidates
    ]

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    try:
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/ranking",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": {"text": query},
                "passages": passages,
            },
        )
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        # Sort by logit score descending
        rankings.sort(key=lambda r: r["logit"], reverse=True)

        reranked = []
        for rank in rankings[:top_k]:
            idx = rank["index"]
            if 0 <= idx < len(candidates):
                result = candidates[idx]
                result.stage = "rerank"
                result.similarity_score = rank["logit"]
                reranked.append(result)
    except Exception as e:
        print(f"  [STAGE 2] Rerank API error: {e}. Falling back to embedding order.")
        reranked = candidates[:top_k]

    elapsed = (time.perf_counter() - start) * 1000

    print(f"  [STAGE 2] Reranked to {len(reranked)} results in {elapsed:.0f}ms")
    for r in reranked:
        print(f"            {r.similarity_score:.4f} — {r.document.title}")

    return reranked


# ---------------------------------------------------------------------------
# Stage 3: Generate answer (reasoning model with retrieved context)
# ---------------------------------------------------------------------------
def generate_answer(
    client: OpenAI,
    query: str,
    context_docs: list[RetrievalResult],
) -> str:
    """Stage 3: Generate answer using reasoning model with retrieved context."""
    print(f"\n  [STAGE 3] Generating answer with {REASONING_MODEL}")
    start = time.perf_counter()

    context = "\n\n".join(
        f"### {r.document.title}\n{r.document.content}"
        for r in context_docs
    )

    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise and cite which documents you used."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.3,
        max_tokens=512,
    )

    elapsed = (time.perf_counter() - start) * 1000
    answer = response.choices[0].message.content
    print(f"  [STAGE 3] Generated in {elapsed:.0f}ms")
    return answer


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------
def rag_pipeline(client: OpenAI, query: str, documents: list[Document]) -> str:
    """
    Complete 3-stage RAG pipeline with specialized models.

    Stage 1: Embed VL (1.7B)  → Fast recall, top-5 candidates
    Stage 2: Rerank VL (1.7B) → Precision reranking, top-3
    Stage 3: Super (12B)      → Reasoning and answer generation
    """
    print(f"\n{'='*60}")
    print(f"  QUERY: {query}")
    print(f"{'='*60}")

    pipeline_start = time.perf_counter()

    # Stage 1: Vector similarity (1.7B model)
    candidates = retrieve_by_embedding(client, query, documents, top_k=5)

    # Stage 2: Cross-encoder reranking
    reranked = rerank_results(client, query, candidates, top_k=3)

    # Stage 3: Answer generation (12B active model)
    answer = generate_answer(client, query, reranked)

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    print(f"\n  Total pipeline latency: {total_ms:.0f}ms")
    print(f"  Documents in knowledge base: {len(documents)}")
    print(f"  Stage 1 candidates: 5 | Stage 2 reranked: 3")

    return answer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    "How does Nemotron 3 Super achieve efficiency despite having 120B parameters?",
    "What safety capabilities does the Nemotron stack provide?",
    "How can I deploy Nemotron models in production?",
]


def main():
    print("=" * 60)
    print("  MULTIMODAL RAG: Specialized Embed + Rerank Pipeline")
    print("  Each stage uses the MINIMUM model for its task")
    print("=" * 60)

    client = get_client()

    # Pre-embed the knowledge base
    documents = embed_documents(client, KNOWLEDGE_BASE)

    # Run queries through the full pipeline
    for query in DEMO_QUERIES:
        answer = rag_pipeline(client, query, documents)
        print(f"\n  ANSWER: {answer[:300]}")
        print(f"\n{'─'*60}")

    print("\n" + "=" * 60)
    print("  KEY INSIGHT")
    print("=" * 60)
    print("  Three-stage pipeline: 1.7B → 1.7B → 12B")
    print("  Embedding (1.7B) and reranking (1.7B cross-encoder) are real")
    print("  NVIDIA API calls — no simulation. Only the final answer")
    print("  generation uses the reasoning model (12B active).")
    print("  Total cost: ~10% of routing everything through a 400B model.")
    print()


if __name__ == "__main__":
    main()
