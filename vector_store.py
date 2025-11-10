"""
Vector store module for TiDB integration with LangChain.
Handles document ingestion, embedding, and storage.
"""

from typing import List, Dict, Any
from langchain_community.vectorstores import TiDBVectorStore
from langchain.schema import Document
from db_connection import TiDBConnection
from embedding_models import EmbeddingModelLoader
from config import Config
import json


class TiDBVectorStoreManager:
    """
    Manages TiDB vector store operations including ingestion and retrieval.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize TiDB Vector Store Manager.
        
        Args:
            embedding_model: Pre-loaded embedding model. If None, loads from config.
        """
        self.db_connection = TiDBConnection()
        self.engine = None
        self.vector_store = None
        
        # Load or use provided embedding model
        if embedding_model is None:
            self.embeddings = EmbeddingModelLoader.load_model()
        else:
            self.embeddings = embedding_model
    
    def initialize(self, drop_existing=False):
        """
        Initialize database connection and vector table.
        
        Args:
            drop_existing: If True, drop and recreate the vector table
        """
        print("Initializing TiDB connection and vector table...")
        
        # Connect to TiDB
        self.engine = self.db_connection.connect()
        
        # Create vector table
        self.db_connection.create_vector_table(drop_existing=drop_existing)
        
        # Initialize LangChain TiDB vector store
        self.vector_store = TiDBVectorStore(
            embedding=self.embeddings,
            connection_string=Config.get_tidb_connection_string(),
            table_name=Config.TABLE_NAME,
            distance_strategy="cosine"
        )
        
        print("TiDB Vector Store initialized successfully.")
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ingest documents into TiDB vector store.
        Automatically generates embeddings for document content.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
        
        Returns:
            Number of documents ingested
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        print(f"\nIngesting {len(documents)} documents...")
        
        # Convert to LangChain Document format
        langchain_docs = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            # Convert metadata dict to JSON string for storage
            metadata_str = json.dumps(metadata) if isinstance(metadata, dict) else str(metadata)
            
            langchain_docs.append(
                Document(
                    page_content=doc['content'],
                    metadata={'metadata_json': metadata_str, **metadata}
                )
            )
        
        # Add documents to vector store (embeddings generated automatically)
        try:
            self.vector_store.add_documents(langchain_docs)
            print(f"Successfully ingested {len(documents)} documents with embeddings.")
            return len(documents)
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            raise
    
    def get_retriever(self, search_type="similarity", k=5, **kwargs):
        """
        Get LangChain retriever interface for vector search.
        
        Args:
            search_type: Type of search ('similarity' or 'mmr')
            k: Number of documents to retrieve
            **kwargs: Additional parameters for the retriever
        
        Returns:
            LangChain Retriever instance
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        search_kwargs = {"k": k}
        search_kwargs.update(kwargs)
        
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        print(f"Retriever created: {search_type} search with k={k}")
        return retriever
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query string
            k: Number of results to return
        
        Returns:
            List of Document objects with similarity scores
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def close(self):
        """Close database connections."""
        if self.db_connection:
            self.db_connection.close()
