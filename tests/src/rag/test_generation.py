# tests/src/rag/test_generation.py
import pytest
from app.src.rag.generation import GeminiGenerator
from app.src.rag.embeddings import EmbeddingModelLoader


def test_gemini_generator_class():
    """Verify app.src.rag.generation.GeminiGenerator using gemini-3.5-flash-lite."""
    generator = GeminiGenerator(model_name="gemini-2.5-flash-lite")
    
    prompt = "Synthesize a 1-line Python function signature for checking if an integer is within bounds."
    response_text = generator.generate(prompt)

    assert response_text is not None
    assert len(response_text) > 0
    print(f"\n[GeminiGenerator Output]:\n{response_text}")


def test_embedding_model_loader_class():
    """Verify app.src.rag.embeddings.EmbeddingModelLoader output vector dimension."""
    embedder = EmbeddingModelLoader(model_name="gemini-embedding-001")
    
    sample_text = "def check_bounds(val: int) -> bool:"
    vector = embedder.get_embedding(sample_text)

    assert isinstance(vector, list)
    assert len(vector) > 0
    print(f"\n[Embedding Vector Dim]: {len(vector)} dimensions")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])