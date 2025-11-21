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
    
    # Remote API Settings (for Qwen and other models)
    REMOTE_EMBEDDING_BASE_URL = os.getenv('REMOTE_EMBEDDING_BASE_URL', '')
    REMOTE_EMBEDDING_API_KEY = os.getenv('REMOTE_EMBEDDING_API_KEY', '')
    REMOTE_EMBEDDING_MODEL = os.getenv('REMOTE_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B')
    
    REMOTE_LLM_BASE_URL = os.getenv('REMOTE_LLM_BASE_URL', '')
    REMOTE_LLM_API_KEY = os.getenv('REMOTE_LLM_API_KEY', '')
    REMOTE_LLM_MODEL = os.getenv('REMOTE_LLM_MODEL', 'Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic')
    
    # Vector Index Configuration
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', '4096'))
    TABLE_NAME = os.getenv('TABLE_NAME', 'documents_vector')
    
    @classmethod
    def get_tidb_connection_string(cls):
        """Generate TiDB connection string for SQLAlchemy."""
        return f"mysql+pymysql://{cls.TIDB_USER}:{cls.TIDB_PASSWORD}@{cls.TIDB_HOST}:{cls.TIDB_PORT}/{cls.TIDB_DATABASE}"
    
    @classmethod
    def validate(cls):
        """Validate required configuration settings."""
        if not cls.REMOTE_EMBEDDING_BASE_URL:
            raise ValueError("REMOTE_EMBEDDING_BASE_URL is required")
        if not cls.REMOTE_EMBEDDING_API_KEY:
            raise ValueError("REMOTE_EMBEDDING_API_KEY is required")
        if not cls.TIDB_HOST:
            raise ValueError("TIDB_HOST is required")
        return True
