import os
import yaml
from typing import List
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

class KnowledgeVectorStore:
    def __init__(self, config_path: str = "app/config/rag.yaml"):
        self.config = self._load_config(config_path)
        self._setup_embed_model()
        
        self.index = VectorStoreIndex.from_documents(
            [], 
            embed_model=Settings.embed_model
        )

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_embed_model(self):
        embed_cfg = self.config.get("embeddings", {})
        provider = embed_cfg.get("provider", "google")
        
        # Default to gemini-embedding-001
        model_name = embed_cfg.get("model_name", "gemini-embedding-001")

        # Strip any 'models/' prefix to avoid path duplication in google-genai
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")

        if provider == "google":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            
            Settings.embed_model = GoogleGenAIEmbedding(
                model_name=model_name,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def add_verified_solution(self, problem_id: str, code: str):
        doc = Document(
            text=code,
            metadata={"problem_id": problem_id, "type": "verified_solution"}
        )
        self.index.insert(doc)

    def search_solutions(self, query: str, similarity_top_k: int = 2) -> List[str]:
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        results = retriever.retrieve(query)
        return [res.node.get_content() for res in results]