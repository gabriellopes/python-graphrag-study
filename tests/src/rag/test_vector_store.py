import os
import pytest
from unittest.mock import patch
from llama_index.core.base.embeddings.base import BaseEmbedding
from app.src.rag.vector_store import KnowledgeVectorStore

@pytest.fixture
def dummy_google_yaml(tmp_path):
    config_file = tmp_path / "rag.yaml"
    config_file.write_text("""
embeddings:
  provider: "google"
  model_name: "gemini-embedding-001"
""")
    return str(config_file)


# Using spec=BaseEmbedding ensures the mock inherits the required base class type
@patch("app.src.rag.vector_store.GoogleGenAIEmbedding", spec=BaseEmbedding)
def test_vector_store_initialization(mock_embed, dummy_google_yaml, monkeypatch):
    """Verifies vector store initialization without hitting network calls or failing type checks."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_testing")
    
    with patch("llama_index.core.VectorStoreIndex.from_documents"):
        store = KnowledgeVectorStore(config_path=dummy_google_yaml)
        assert store.config["embeddings"]["provider"] == "google"
        assert store.config["embeddings"]["model_name"] == "gemini-embedding-001"
        mock_embed.assert_called_once()


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    reason="Requires valid GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
)
def test_add_and_retrieve_solution(dummy_google_yaml):
    """Live integration test against Google Embeddings API."""
    store = KnowledgeVectorStore(config_path=dummy_google_yaml)
    
    prob_id = "http://example.org/kg-study#ProblemBounds"
    code_snippet = "def check_bounds(val: int):\n    return 0 <= val <= 100"

    store.add_verified_solution(problem_id=prob_id, code=code_snippet)
    results = store.search_solutions(query="check integer bounds", similarity_top_k=1)

    assert len(results) > 0
    assert "def check_bounds" in results[0]