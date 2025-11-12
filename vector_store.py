"""
Vector store module for TiDB integration with LangChain.
Handles document ingestion, embedding, and storage.
"""

from typing import List, Dict, Any, Union
from langchain_community.vectorstores import TiDBVectorStore
from langchain_core.documents import Document
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
    
    def ingest_documents(self, documents: List[Union[Document, dict]]) -> int:
        """
        Ingest documents into the vector store.
        Checks for existing documents based on source metadata to avoid duplicates.
        
        Args:
            documents: List of Document objects or dictionaries with 'page_content' and 'metadata'
        
        Returns:
            Number of documents ingested (excluding duplicates)
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        # Convert dict to LangChain Document if needed
        langchain_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                langchain_docs.append(
                    Document(
                        page_content=doc.get("page_content", ""),
                        metadata=doc.get("metadata", {})
                    )
                )
            else:
                langchain_docs.append(doc)
        
        # Check for existing documents to avoid duplicates
        from sqlalchemy import select
        from db_connection import DocumentVector
        
        existing_sources = set()
        with self.db_connection.Session() as session:
            # Get all existing document sources
            stmt = select(DocumentVector.meta)
            results = session.execute(stmt).all()
            
            for result in results:
                if result[0]:  # meta column
                    import json
                    try:
                        metadata = json.loads(result[0])
                        if "source" in metadata:
                            existing_sources.add(metadata["source"])
                    except (json.JSONDecodeError, TypeError):
                        continue
        
        # Filter out documents that already exist
        new_docs = []
        skipped_count = 0
        for doc in langchain_docs:
            source = doc.metadata.get("source", "")
            if source and source in existing_sources:
                skipped_count += 1
            else:
                new_docs.append(doc)
        
        # Add only new documents to vector store
        if new_docs:
            self.vector_store.add_documents(new_docs)
            print(f"Ingested {len(new_docs)} new documents into TiDB Vector Store")
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} duplicate documents")
        
        if not new_docs and skipped_count > 0:
            print("All documents already exist in the database")
        
        return len(new_docs)
    
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
