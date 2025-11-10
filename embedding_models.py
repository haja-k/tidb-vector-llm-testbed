"""
Embedding model loader module.
Supports OpenAI and HuggingFace embedding models.
"""

from typing import List
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from config import Config


class EmbeddingModelLoader:
    """
    Factory class for loading different embedding models.
    """
    
    @staticmethod
    def load_model(model_type: str = None):
        """
        Load embedding model based on configuration.
        
        Args:
            model_type: Type of model to load ('openai' or 'huggingface')
                       If None, uses Config.EMBEDDING_MODEL
        
        Returns:
            Embedding model instance compatible with LangChain
        """
        if model_type is None:
            model_type = Config.EMBEDDING_MODEL
        
        print(f"Loading {model_type} embedding model...")
        
        if model_type.lower() == 'openai':
            return EmbeddingModelLoader._load_openai()
        elif model_type.lower() == 'huggingface':
            return EmbeddingModelLoader._load_huggingface()
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
    
    @staticmethod
    def _load_openai():
        """
        Load OpenAI embedding model.
        
        Returns:
            OpenAIEmbeddings instance
        """
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in configuration")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=Config.OPENAI_API_KEY,
            model="text-embedding-ada-002"  # Latest OpenAI embedding model
        )
        
        print("OpenAI embeddings loaded successfully (dimension: 1536)")
        return embeddings
    
    @staticmethod
    def _load_huggingface():
        """
        Load HuggingFace embedding model.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        model_name = Config.HUGGINGFACE_MODEL
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"HuggingFace embeddings loaded successfully: {model_name}")
        return embeddings
    
    @staticmethod
    def get_embedding_dimension(model_type: str = None) -> int:
        """
        Get the dimension of embeddings for a given model type.
        
        Args:
            model_type: Type of model ('openai' or 'huggingface')
        
        Returns:
            Dimension of the embedding vectors
        """
        if model_type is None:
            model_type = Config.EMBEDDING_MODEL
        
        if model_type.lower() == 'openai':
            return 1536  # OpenAI text-embedding-ada-002 dimension
        elif model_type.lower() == 'huggingface':
            # For all-MiniLM-L6-v2 it's 384, but may vary by model
            return Config.VECTOR_DIMENSION
        else:
            return Config.VECTOR_DIMENSION
