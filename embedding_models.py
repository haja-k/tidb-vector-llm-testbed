"""
Embedding model loader module.
Supports remote API models (OpenAI-compatible).
"""

from typing import List
import requests
from langchain_core.embeddings import Embeddings
from config import Config


class CustomOpenAIEmbeddings(Embeddings):
    """
    Custom embedding class for self-hosted OpenAI-compatible APIs.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": text,
                "model": self.model
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": texts,
                "model": self.model
            }
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class EmbeddingModelLoader:
    """
    Factory class for loading remote API embedding models.
    """
    
    @staticmethod
    def load_model():
        """
        Load remote API embedding model based on configuration.
        
        Returns:
            Embedding model instance compatible with LangChain
        """
        print("Loading remote API embedding model...")
        return EmbeddingModelLoader._load_remote()
    
    @staticmethod
    def _load_remote():
        """
        Load remote API embedding model (OpenAI-compatible).
        
        Returns:
            CustomOpenAIEmbeddings instance
        """
        if not Config.REMOTE_EMBEDDING_API_KEY:
            raise ValueError("REMOTE_EMBEDDING_API_KEY not found in configuration")
        
        embeddings = CustomOpenAIEmbeddings(
            api_key=Config.REMOTE_EMBEDDING_API_KEY,
            base_url=Config.REMOTE_EMBEDDING_BASE_URL,
            model=Config.REMOTE_EMBEDDING_MODEL
        )
        
        print(f"Remote API embeddings loaded successfully: {Config.REMOTE_EMBEDDING_MODEL}")
        print(f"  - API URL: {Config.REMOTE_EMBEDDING_BASE_URL}")
        return embeddings
    
    @staticmethod
    def get_embedding_dimension() -> int:
        """
        Get the dimension of embeddings for the remote model.
        
        Returns:
            Dimension of the embedding vectors
        """
        return Config.VECTOR_DIMENSION
