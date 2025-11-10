"""
Embedding model loader module.
Supports remote API models (OpenAI-compatible).
"""

from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from config import Config


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
            OpenAIEmbeddings instance
        """
        if not Config.REMOTE_EMBEDDING_API_KEY:
            raise ValueError("REMOTE_EMBEDDING_API_KEY not found in configuration")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=Config.REMOTE_EMBEDDING_API_KEY,
            openai_api_base=Config.REMOTE_EMBEDDING_BASE_URL,
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
