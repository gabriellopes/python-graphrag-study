# app/src/rag/embeddings.py
import os
import logging
from typing import List
from dotenv import load_dotenv
from google import genai

load_dotenv()
logger = logging.getLogger("EmbeddingModelLoader")


class EmbeddingModelLoader:
    """Wrapper for Google Gemini embeddings using text-embedding-004 / gemini-embedding-001."""

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.client = genai.Client(api_key=self.api_key)

    def get_embedding(self, text: str) -> List[float]:
        """Returns dense float vector embedding for a single text string."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return response.embeddings[0].values