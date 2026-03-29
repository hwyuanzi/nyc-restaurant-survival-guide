import torch
import pytest
from retrieval.vector_search import SemanticSearchModel


@pytest.fixture(scope="module")
def model():
    """Load the NLP model once for all tests in this module."""
    return SemanticSearchModel(model_name='sentence-transformers/all-MiniLM-L6-v2')


def test_embed_texts_shape(model):
    """Test that embedding produces correct tensor dimensions."""
    texts = ["hello world", "test sentence"]
    embeddings = model.embed_texts(texts)
    assert embeddings.shape == (2, 384), "Embedding should be (n_texts, 384)"


def test_embeddings_are_normalized(model):
    """Test that output vectors are L2-normalized (unit vectors)."""
    embeddings = model.embed_texts(["any text"])
    norm = torch.norm(embeddings, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-4), "Vectors should be L2-normalized"


def test_search_returns_correct_top_k(model):
    """Test that search returns the requested number of results."""
    corpus = ["Italian pasta", "Mexican tacos", "Japanese sushi", "French bread"]
    corpus_emb = model.embed_texts(corpus)
    indices, scores = model.search("Italian food", corpus_emb, top_k=2)
    assert len(indices) == 2
    assert len(scores) == 2


def test_search_semantic_relevance(model):
    """Test that semantically similar queries rank higher."""
    corpus = [
        "A romantic Italian restaurant with pasta and wine",
        "A loud cheap street food cart",
        "A fancy French fine dining experience",
    ]
    corpus_emb = model.embed_texts(corpus)

    indices, scores = model.search("cozy Italian dinner", corpus_emb, top_k=1)
    assert indices[0] == 0, "Italian restaurant should rank #1 for 'cozy Italian dinner'"

    indices, scores = model.search("cheap fast food", corpus_emb, top_k=1)
    assert indices[0] == 1, "Street food cart should rank #1 for 'cheap fast food'"
