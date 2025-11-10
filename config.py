"""
Configuration module for TiDB Vector LLM Testbed.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for TiDB and embedding settings."""
    
    # TiDB Connection Settings
    TIDB_HOST = os.getenv('TIDB_HOST', 'localhost')
    TIDB_PORT = int(os.getenv('TIDB_PORT', '4000'))
    TIDB_USER = os.getenv('TIDB_USER', 'root')
    TIDB_PASSWORD = os.getenv('TIDB_PASSWORD', '')
    TIDB_DATABASE = os.getenv('TIDB_DATABASE', 'vector_testbed')
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai')  # 'openai', 'huggingface', or 'ollama'
    HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Ollama Settings (for locally hosted models)
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'qwen:latest')
    OLLAMA_LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'llama3:latest')
    
    # Vector Index Configuration
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', '1536'))  # Default for OpenAI
    TABLE_NAME = os.getenv('TABLE_NAME', 'documents_vector')
    
    @classmethod
    def get_tidb_connection_string(cls):
        """Generate TiDB connection string for SQLAlchemy."""
        return f"mysql+pymysql://{cls.TIDB_USER}:{cls.TIDB_PASSWORD}@{cls.TIDB_HOST}:{cls.TIDB_PORT}/{cls.TIDB_DATABASE}"
    
    @classmethod
    def validate(cls):
        """Validate required configuration settings."""
        if cls.EMBEDDING_MODEL == 'openai' and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        
        if cls.EMBEDDING_MODEL == 'ollama' and not cls.OLLAMA_BASE_URL:
            raise ValueError("OLLAMA_BASE_URL is required when using Ollama embeddings")
        
        if not cls.TIDB_HOST:
            raise ValueError("TIDB_HOST is required")
        
        return True
