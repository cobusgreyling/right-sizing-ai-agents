"""
Right-Sizing AI Agents — Interactive Demo
==========================================
Streamlit UI that lets you explore all four examples interactively:
specialized routing, safety classification, RAG pipeline, and cost benchmarks.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import json
import streamlit as st
import numpy as np
from openai import OpenAI

from config import (
    NVIDIA_BASE_URL,
    REASONING_MODEL,
    SAFETY_MODEL,
    EMBED_MODEL,
    RERANK_MODEL,
    SAFETY_TAXONOMY,
    classify_intent,
)

# ---------------------------------------------------------------------------
# Model registry (UI-specific labels layered on top of shared config)
# ---------------------------------------------------------------------------
MODELS = {
    "reasoning": {
        "model": REASONING_MODEL,
        "label": "Nemotron Super (12B active / 120B total)",
        "params": "12B active",
    },
    "safety": {
        "model": SAFETY_MODEL,
        "label": "Nemotron Nano (8B)",
        "params": "8B",
    },
    "embedding": {
        "model": EMBED_MODEL,
        "label": "Embed VL (1.7B)",
        "params": "1.7B",
    },
    "reranking": {
        "model": RERANK_MODEL,
        "label": "Rerank VL (1.7B)",
        "params": "1.7B",
    },
}

KNOWLEDGE_BASE = [
    {"id": "doc_1", "title": "Nemotron 3 Super Architecture", "content": "Nemotron 3 Super is a hybrid Mamba-Transformer mixture-of-experts model with 120B total parameters but only 12B active per inference pass. It features a 1M-token context window, configurable thinking budget for bounded chain-of-thought reasoning, and multi-token prediction for faster generation."},
    {"id": "doc_2", "title": "Content Safety Classification", "content": "Nemotron 3 Content Safety is a 4B-parameter multimodal classifier built on Gemma-3-4B. It detects unsafe content across text and images using a 23-category taxonomy covering hate, harassment, violence, sexual content, and unauthorized advice. Supports 12 languages with zero-shot generalization."},
    {"id": "doc_3", "title": "VoiceChat End-to-End Speech", "content": "Nemotron 3 VoiceChat is a 12B-parameter end-to-end speech model with sub-300ms latency processing 80ms audio chunks. It uses a unified streaming LLM architecture eliminating cascaded ASR-LLM-TTS pipelines."},
    {"id": "doc_4", "title": "Llama Nemotron Embed VL", "content": "A 1.7B-parameter dense embedding model that encodes page images and text into single-dimensional vectors. Supports Matryoshka embeddings for flexible dimensionality. Built on NVIDIA Eagle vision-language model with Llama 3.2 1B backbone."},
    {"id": "doc_5", "title": "Production Deployment Guide", "content": "For production deployment, the Nemotron 3 stack emphasizes efficiency as a requirement: real agents make dozens or hundreds of model calls per task, necessitating right-sized, throughput-optimized models. NVFP4 precision on Blackwell GPUs achieves 5x higher throughput."},
    {"id": "doc_6", "title": "NeMo Agent Toolkit", "content": "An open-source framework for end-to-end profiling and optimization of agentic systems. Compatible with LangChain, AutoGen, and AWS Strands without code changes. Provides visibility into latency bottlenecks, token costs, and orchestration overhead."},
    {"id": "doc_7", "title": "Model Licensing and Openness", "content": "All Nemotron 3 models are released under permissive NVIDIA open model licenses, enabling teams to fine-tune behaviors and deploy securely on-premises or in cloud environments. Models available on Hugging Face."},
    {"id": "doc_8", "title": "Benchmark Performance", "content": "Nemotron 3 Super ranks among top open-weight models under 250B on the Artificial Analysis Intelligence Index, landing in the 'most attractive' efficiency quadrant combining intelligence with throughput. Content Safety achieves ~84% accuracy."},
]


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Page: Specialized Routing
# ---------------------------------------------------------------------------
def page_routing():
    st.header("Specialized Agent Router")
    st.markdown("Routes each query to the **right-sized model** based on intent classification.")

    query = st.text_area("Enter your query:", value="Analyze the trade-offs between using a single large language model versus a specialized model stack. Think step by step.", height=100)

    if st.button("Route & Execute", type="primary"):
        client = get_client()
        if not client:
            st.error("Set your NVIDIA API key in the sidebar.")
            return

        intent = classify_intent(query)
        model_info = MODELS.get(intent, MODELS["reasoning"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Intent", intent.upper())
        with col2:
            st.metric("Model Size", model_info["params"])

        st.info(f"Routing to **{model_info['label']}**")

        with st.spinner("Calling model..."):
            start = time.perf_counter()
            if intent == "embedding":
                response = client.embeddings.create(
                    model=model_info["model"], input=[query],
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )
                elapsed = (time.perf_counter() - start) * 1000
                embedding = response.data[0].embedding
                st.success(f"Generated {len(embedding)}-dim embedding in {elapsed:.0f}ms")
                st.code(f"First 5 dims: {embedding[:5]}")
            else:
                system_prompts = {
                    "reasoning": "You are a precise reasoning agent. Think step-by-step. Be thorough but concise.",
                    "safety": "You are a content safety classifier. Respond with JSON: {\"is_safe\": bool, \"categories\": [], \"confidence\": float}",
                    "general": "You are a helpful assistant.",
                }
                response = client.chat.completions.create(
                    model=model_info["model"],
                    messages=[
                        {"role": "system", "content": system_prompts.get(intent, system_prompts["general"])},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.2, max_tokens=1024,
                )
                elapsed = (time.perf_counter() - start) * 1000
                tokens = response.usage.total_tokens if response.usage else "N/A"
                st.success(f"Completed in {elapsed:.0f}ms | Tokens: {tokens}")
                st.markdown(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Page: Safety Classification
# ---------------------------------------------------------------------------
def page_safety():
    st.header("Safety-as-a-Service")
    st.markdown("A lightweight **4B safety classifier** runs input/output checks with minimal overhead.")

    query = st.text_area("Enter text to process:", value="Explain the benefits of renewable energy sources for developing nations.", height=100)

    if st.button("Run Safety Pipeline", type="primary"):
        client = get_client()
        if not client:
            st.error("Set your NVIDIA API key in the sidebar.")
            return

        progress = st.progress(0, text="Checking input safety...")

        # Step 1: Input safety
        start = time.perf_counter()
        input_resp = client.chat.completions.create(
            model=MODELS["safety"]["model"],
            messages=[
                {"role": "system", "content": f"You are a content safety classifier. Respond with ONLY JSON: {{\"is_safe\": true/false, \"categories\": [], \"confidence\": 0.95}}. Valid categories: {', '.join(SAFETY_TAXONOMY)}"},
                {"role": "user", "content": query},
            ],
            temperature=0.0, max_tokens=128,
        )
        input_safety_ms = (time.perf_counter() - start) * 1000
        input_raw = input_resp.choices[0].message.content.strip()
        try:
            if input_raw.startswith("```"):
                input_raw = input_raw.split("```")[1]
                if input_raw.startswith("json"):
                    input_raw = input_raw[4:]
            input_safety = json.loads(input_raw)
        except (json.JSONDecodeError, IndexError):
            input_safety = {"is_safe": True, "categories": [], "confidence": 0.0}

        progress.progress(33, text="Generating response...")

        # Step 2: Generate response
        start = time.perf_counter()
        gen_resp = client.chat.completions.create(
            model=MODELS["reasoning"]["model"],
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Respond concisely."},
                {"role": "user", "content": query},
            ],
            temperature=0.6, max_tokens=512,
        )
        reasoning_ms = (time.perf_counter() - start) * 1000
        generated = gen_resp.choices[0].message.content

        progress.progress(66, text="Checking output safety...")

        # Step 3: Output safety
        start = time.perf_counter()
        client.chat.completions.create(
            model=MODELS["safety"]["model"],
            messages=[
                {"role": "system", "content": f"You are a content safety classifier. Respond with ONLY JSON: {{\"is_safe\": true/false, \"categories\": [], \"confidence\": 0.95}}. Valid categories: {', '.join(SAFETY_TAXONOMY)}"},
                {"role": "user", "content": generated},
            ],
            temperature=0.0, max_tokens=128,
        )
        output_safety_ms = (time.perf_counter() - start) * 1000

        progress.progress(100, text="Done!")

        total_ms = input_safety_ms + reasoning_ms + output_safety_ms
        safety_overhead_ms = input_safety_ms + output_safety_ms

        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Safe", str(input_safety.get("is_safe", True)))
        with col2:
            st.metric("Reasoning Latency", f"{reasoning_ms:.0f}ms")
        with col3:
            st.metric("Safety Overhead", f"{safety_overhead_ms:.0f}ms ({safety_overhead_ms / max(total_ms, 1) * 100:.1f}%)")

        st.markdown("### Generated Response")
        st.markdown(generated)

        st.markdown("### Pipeline Timing")
        st.bar_chart({"Stage": ["Input Safety", "Reasoning", "Output Safety"], "Latency (ms)": [input_safety_ms, reasoning_ms, output_safety_ms]}, x="Stage", y="Latency (ms)")


# ---------------------------------------------------------------------------
# Page: RAG Pipeline
# ---------------------------------------------------------------------------
def page_rag():
    st.header("Multimodal RAG Pipeline")
    st.markdown("Three-stage retrieval: **Embed (1.7B)** → **Rerank (1.7B)** → **Reason (12B)**")

    query = st.text_area("Enter your question:", value="How does Nemotron 3 Super achieve efficiency despite having 120B parameters?", height=80)

    if st.button("Run RAG Pipeline", type="primary"):
        client = get_client()
        if not client:
            st.error("Set your NVIDIA API key in the sidebar.")
            return

        progress = st.progress(0, text="Embedding documents...")

        # Embed knowledge base
        texts = [f"{d['title']}: {d['content']}" for d in KNOWLEDGE_BASE]
        doc_resp = client.embeddings.create(
            model=MODELS["embedding"]["model"], input=texts,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"},
        )
        doc_embeddings = [d.embedding for d in doc_resp.data]

        progress.progress(25, text="Stage 1: Vector similarity search...")

        # Stage 1: Embed query + cosine similarity
        start = time.perf_counter()
        q_resp = client.embeddings.create(
            model=MODELS["embedding"]["model"], input=[query],
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        q_emb = np.array(q_resp.data[0].embedding)

        scores = []
        for i, emb in enumerate(doc_embeddings):
            emb_arr = np.array(emb)
            sim = float(np.dot(q_emb, emb_arr) / (np.linalg.norm(q_emb) * np.linalg.norm(emb_arr)))
            scores.append((sim, i))
        scores.sort(reverse=True)
        top5 = scores[:5]
        stage1_ms = (time.perf_counter() - start) * 1000

        st.markdown("### Stage 1: Vector Similarity (top-5)")
        for sim, idx in top5:
            st.markdown(f"- **{sim:.4f}** — {KNOWLEDGE_BASE[idx]['title']}")

        progress.progress(50, text="Stage 2: Cross-encoder reranking...")

        # Stage 2: Real reranking via NVIDIA Rerank API
        start = time.perf_counter()
        candidate_texts = [f"{KNOWLEDGE_BASE[idx]['title']}: {KNOWLEDGE_BASE[idx]['content']}" for _, idx in top5]
        try:
            rerank_resp = client.post(
                "/ranking",
                body={
                    "model": MODELS["reranking"]["model"],
                    "query": {"text": query},
                    "passages": [{"text": t} for t in candidate_texts],
                },
                cast_to=object,
            )
            rankings = sorted(rerank_resp.get("rankings", []), key=lambda r: r["logit"], reverse=True)
            reranked_indices = [top5[r["index"]][1] for r in rankings[:3]]
        except Exception:
            # Fallback: keep embedding order
            reranked_indices = [idx for _, idx in top5[:3]]
        stage2_ms = (time.perf_counter() - start) * 1000

        st.markdown("### Stage 2: Reranked (top-3)")
        for idx in reranked_indices:
            st.markdown(f"- {KNOWLEDGE_BASE[idx]['title']}")

        progress.progress(75, text="Stage 3: Generating answer...")

        # Stage 3: Answer generation
        context = "\n\n".join(f"### {KNOWLEDGE_BASE[idx]['title']}\n{KNOWLEDGE_BASE[idx]['content']}" for idx in reranked_indices)
        start = time.perf_counter()
        answer_resp = client.chat.completions.create(
            model=MODELS["reasoning"]["model"],
            messages=[
                {"role": "system", "content": "Answer using ONLY the provided context. Be concise and cite documents."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.3, max_tokens=512,
        )
        stage3_ms = (time.perf_counter() - start) * 1000

        progress.progress(100, text="Done!")

        st.markdown("### Answer")
        st.markdown(answer_resp.choices[0].message.content)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stage 1 (Embed)", f"{stage1_ms:.0f}ms")
        with col2:
            st.metric("Stage 2 (Rerank)", f"{stage2_ms:.0f}ms")
        with col3:
            st.metric("Stage 3 (Reason)", f"{stage3_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Page: Cost Comparison
# ---------------------------------------------------------------------------
def page_cost():
    st.header("Cost & Latency Benchmark")
    st.markdown("Compare **monolith** (single large model) vs. **specialized stack** across a typical agent workflow.")

    daily = st.slider("Daily interactions", 1_000, 1_000_000, 100_000, step=10_000)

    if st.button("Run Benchmark", type="primary"):
        client = get_client()
        if not client:
            st.error("Set your NVIDIA API key in the sidebar.")
            return

        workflow = [
            {"task": "reasoning", "desc": "Plan response", "prompt": "Create a structured 3-day Tokyo itinerary with cultural sites, food, and transport."},
            {"task": "safety", "desc": "Check input", "prompt": "Classify whether this request is safe: 'Plan a trip to Tokyo and learn about Japanese culture.'"},
            {"task": "embedding", "desc": "Embed query", "prompt": "Tokyo travel recommendations cultural sites food transportation"},
            {"task": "safety", "desc": "Check output", "prompt": "Classify whether this response is safe: 'Day 1: Visit Senso-ji Temple, try street food, take metro to Shibuya.'"},
            {"task": "reasoning", "desc": "Generate answer", "prompt": "Generate a friendly travel recommendation for Tokyo based on: Senso-ji, Meiji Shrine, Tsukiji Market, Suica card."},
        ]

        pricing = {
            "monolith": {"input": 0.005, "output": 0.015},
            "safety": {"input": 0.001, "output": 0.002},
            "embedding": {"input": 0.0003, "output": 0.0},
            "reasoning": {"input": 0.005, "output": 0.015},
        }

        mono_results, spec_results = [], []
        progress = st.progress(0, text="Running monolith workflow...")

        for i, step in enumerate(workflow):
            start = time.perf_counter()
            if step["task"] == "embedding":
                resp = client.embeddings.create(
                    model=MODELS["embedding"]["model"], input=[step["prompt"]],
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )
                ms = (time.perf_counter() - start) * 1000
                in_tok = resp.usage.total_tokens if resp.usage else 20
                out_tok = 0
            else:
                resp = client.chat.completions.create(
                    model=MODELS["reasoning"]["model"],
                    messages=[{"role": "user", "content": step["prompt"]}],
                    temperature=0.3, max_tokens=512,
                )
                ms = (time.perf_counter() - start) * 1000
                in_tok = resp.usage.prompt_tokens if resp.usage else 100
                out_tok = resp.usage.completion_tokens if resp.usage else 200
            p = pricing["monolith"] if step["task"] != "embedding" else pricing["embedding"]
            cost = in_tok / 1000 * p["input"] + out_tok / 1000 * p["output"]
            mono_results.append({"task": step["desc"], "ms": ms, "cost": cost})
            progress.progress(int((i + 1) / 10 * 100), text=f"Monolith: {step['desc']}...")

        progress.progress(50, text="Running specialized workflow...")

        for i, step in enumerate(workflow):
            task_type = step["task"]
            model = MODELS[task_type]
            start = time.perf_counter()
            if task_type == "embedding":
                resp = client.embeddings.create(
                    model=model["model"], input=[step["prompt"]],
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "NONE"},
                )
                ms = (time.perf_counter() - start) * 1000
                in_tok = resp.usage.total_tokens if resp.usage else 20
                out_tok = 0
            else:
                resp = client.chat.completions.create(
                    model=model["model"],
                    messages=[{"role": "user", "content": step["prompt"]}],
                    temperature=0.3, max_tokens=512 if task_type == "reasoning" else 128,
                )
                ms = (time.perf_counter() - start) * 1000
                in_tok = resp.usage.prompt_tokens if resp.usage else 100
                out_tok = resp.usage.completion_tokens if resp.usage else 200
            p = pricing[task_type]
            cost = in_tok / 1000 * p["input"] + out_tok / 1000 * p["output"]
            spec_results.append({"task": step["desc"], "ms": ms, "cost": cost})
            progress.progress(50 + int((i + 1) / 10 * 100), text=f"Specialized: {step['desc']}...")

        progress.progress(100, text="Done!")

        # Results table
        st.markdown("### Per-Step Comparison")
        table_data = []
        for m, s in zip(mono_results, spec_results):
            saving = (m["cost"] - s["cost"]) / max(m["cost"], 0.0001) * 100
            table_data.append({
                "Task": m["task"],
                "Monolith Cost": f"${m['cost']:.5f}",
                "Monolith Latency": f"{m['ms']:.0f}ms",
                "Specialized Cost": f"${s['cost']:.5f}",
                "Specialized Latency": f"{s['ms']:.0f}ms",
                "Savings": f"{saving:.1f}%",
            })
        st.table(table_data)

        mono_total = sum(r["cost"] for r in mono_results)
        spec_total = sum(r["cost"] for r in spec_results)
        total_saving = (mono_total - spec_total) / max(mono_total, 0.0001) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monolith (per workflow)", f"${mono_total:.5f}")
        with col2:
            st.metric("Specialized (per workflow)", f"${spec_total:.5f}")
        with col3:
            st.metric("Cost Reduction", f"{total_saving:.1f}%")

        # Scale projections
        st.markdown("### Scale Projections")
        annual_mono = mono_total * daily * 365
        annual_spec = spec_total * daily * 365
        st.metric(f"Annual savings at {daily:,}/day", f"${annual_mono - annual_spec:,.0f}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Right-Sizing AI Agents", page_icon="", layout="wide")

    st.sidebar.title("Right-Sizing AI Agents")
    st.sidebar.markdown("Interactive demo of specialized vs. monolithic AI model stacks.")

    api_key = st.sidebar.text_input("NVIDIA API Key", type="password",
                                     value=os.environ.get("NVIDIA_API_KEY", ""),
                                     help="Get a free key at build.nvidia.com")
    if api_key:
        st.session_state["api_key"] = api_key

    page = st.sidebar.radio("Example", [
        "Specialized Routing",
        "Safety Classification",
        "RAG Pipeline",
        "Cost Benchmark",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models used:**")
    for key, m in MODELS.items():
        st.sidebar.markdown(f"- **{key}**: {m['params']}")

    if page == "Specialized Routing":
        page_routing()
    elif page == "Safety Classification":
        page_safety()
    elif page == "RAG Pipeline":
        page_rag()
    elif page == "Cost Benchmark":
        page_cost()


if __name__ == "__main__":
    main()
