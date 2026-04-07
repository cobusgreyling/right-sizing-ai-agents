"""Tests for the multimodal RAG pipeline (Example 03)."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "03_multimodal_rag"))

from multimodal_rag import (
    cosine_similarity,
    Document,
    RetrievalResult,
    KNOWLEDGE_BASE,
    EMBED_MODEL,
    RERANK_MODEL,
    REASONING_MODEL,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_opposite_vectors(self):
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(cosine_similarity(a, b) - expected) < 1e-6

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        a = rng.random(768).tolist()
        b = rng.random(768).tolist()
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0


class TestDocument:
    def test_default_embedding(self):
        doc = Document(id="1", title="Test", content="Content", category="test")
        assert doc.embedding == []

    def test_with_embedding(self):
        emb = [0.1, 0.2, 0.3]
        doc = Document(id="1", title="Test", content="Content", category="test", embedding=emb)
        assert doc.embedding == emb


class TestKnowledgeBase:
    def test_not_empty(self):
        assert len(KNOWLEDGE_BASE) >= 5

    def test_all_docs_have_required_fields(self):
        for doc in KNOWLEDGE_BASE:
            assert doc.id
            assert doc.title
            assert doc.content
            assert doc.category

    def test_unique_ids(self):
        ids = [doc.id for doc in KNOWLEDGE_BASE]
        assert len(ids) == len(set(ids))

    def test_categories_are_diverse(self):
        categories = {doc.category for doc in KNOWLEDGE_BASE}
        assert len(categories) >= 4


class TestModelConfig:
    def test_embed_model_is_lightweight(self):
        assert "1b" in EMBED_MODEL.lower() or "embed" in EMBED_MODEL.lower()

    def test_rerank_model_is_lightweight(self):
        assert "rerank" in RERANK_MODEL.lower()

    def test_reasoning_model_is_present(self):
        assert "nemotron" in REASONING_MODEL.lower() or "super" in REASONING_MODEL.lower()


class TestRetrievalResult:
    def test_creation(self):
        doc = Document(id="1", title="Test", content="Content", category="test")
        result = RetrievalResult(document=doc, similarity_score=0.95, stage="embed")
        assert result.similarity_score == 0.95
        assert result.stage == "embed"
