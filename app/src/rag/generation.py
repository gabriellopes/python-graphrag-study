# app/src/rag/generation.py
import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()
logger = logging.getLogger("GeminiGenerator")


class GeminiGenerator:
    """Wrapper for Gemini API generation calls using gemini-3.5-flash-lite."""

    def __init__(self, model_name: str = "gemini-3.5-flash-lite", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, max_retries: int = 3, backoff_factor: float = 2.0) -> str:
        """Generates content with exponential backoff retry logic."""
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response.text:
                    return response.text.strip()
                raise ValueError("Received empty response from Gemini API.")
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    logger.error(f"Failed generation after {max_retries} attempts: {e}")
                    raise e
                sleep_time = backoff_factor ** attempt
                logger.warning(f"Retrying generation in {sleep_time}s (Attempt {attempt}/{max_retries})...")
                time.sleep(sleep_time)
        return ""