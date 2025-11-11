"""
Sample dataset for testing and benchmarking.
Loads markdown documents from the scspedia folder and chunks them for vectorization.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document

def load_scspedia_documents(scspedia_path: str = "scspedia") -> List[Document]:
    """
    Load markdown documents from the scspedia folder.
    
    Args:
        scspedia_path: Path to the scspedia folder (relative to project root)
    
    Returns:
        List of LangChain Document objects
    """
    if not os.path.exists(scspedia_path):
        raise FileNotFoundError(f"Scspedia folder not found: {scspedia_path}")
    
    # Load all markdown files from the directory
    loader = DirectoryLoader(
        scspedia_path,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    # Add metadata about the source file
    for doc in documents:
        filename = os.path.basename(doc.metadata['source'])
        doc.metadata.update({
            'filename': filename,
            'type': 'markdown',
            'source_folder': 'scspedia'
        })
    
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better embedding and retrieval.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked Document objects
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        chunked_documents.extend(chunks)
    
    return chunked_documents


def get_documents(scspedia_path: str = "scspedia", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Load and chunk documents from scspedia folder for ingestion into vector store.
    
    Args:
        scspedia_path: Path to the scspedia folder
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with 'content' and 'metadata' keys
    """
    # Load documents
    raw_documents = load_scspedia_documents(scspedia_path)
    
    # Chunk documents
    chunked_documents = chunk_documents(raw_documents, chunk_size, chunk_overlap)
    
    # Convert to the format expected by vector_store.py
    documents = []
    for idx, doc in enumerate(chunked_documents):
        documents.append({
            "content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "chunk_id": idx,
                "chunk_size": len(doc.page_content)
            }
        })
    
    return documents


def get_markdown_documents(scspedia_path: str = "scspedia") -> List[Dict[str, Any]]:
    """
    Load full markdown documents without chunking (for smaller documents).
    
    Args:
        scspedia_path: Path to the scspedia folder
    
    Returns:
        List of dictionaries with 'content' and 'metadata' keys
    """
    # Load documents
    raw_documents = load_scspedia_documents(scspedia_path)
    
    # Convert to the format expected by vector_store.py
    documents = []
    for idx, doc in enumerate(raw_documents):
        documents.append({
            "content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "document_id": idx,
                "document_size": len(doc.page_content)
            }
        })
    
    return documents


def get_test_queries() -> List[str]:
    """
    Get test queries based on scspedia content for evaluation.
    
    Returns:
        List of test query strings
    """
    return [
        "What is the Cabinet of Malaysia and Sarawak?",
        "What are the key provisions of the Federal Constitution?",
        "What is the General Order of Sarawak?",
        "What are the highlights of PCDS 2023?",
        "What are the economic sectors in PCDS 2030?",
        "What are the poverty alleviation measures in PCDS 2030?",
        "What is the main report of PCDS 2030?",
        "Who is the Premier of Sarawak?",
        "What is Sarawak Delta Geopark?",
        "What is the Sarawak Digital Economy Blueprint 2030?",
        "What are the facts and figures of Sarawak 2024?",
        "What are the 6 Shared Values of SCS?",
        "What is the Constitution of the State of Sarawak?"
    ]


def load_scspedia_documents(scspedia_path: str = "scspedia") -> List[Document]:
    """
    Load markdown documents from the scspedia folder.
    
    Args:
        scspedia_path: Path to the scspedia folder (relative to project root)
    
    Returns:
        List of LangChain Document objects
    """
    if not os.path.exists(scspedia_path):
        raise FileNotFoundError(f"Scspedia folder not found: {scspedia_path}")
    
    # Load all markdown files from the directory
    loader = DirectoryLoader(
        scspedia_path,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    # Add metadata about the source file
    for doc in documents:
        filename = os.path.basename(doc.metadata['source'])
        doc.metadata.update({
            'filename': filename,
            'type': 'markdown',
            'source_folder': 'scspedia'
        })
    
    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better embedding and retrieval.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked Document objects
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        chunked_documents.extend(chunks)
    
    return chunked_documents


def get_documents(scspedia_path: str = "scspedia", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Load and chunk documents from scspedia folder for ingestion into vector store.
    
    Args:
        scspedia_path: Path to the scspedia folder
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with 'content' and 'metadata' keys
    """
    # Load documents
    raw_documents = load_scspedia_documents(scspedia_path)
    
    # Chunk documents
    chunked_documents = chunk_documents(raw_documents, chunk_size, chunk_overlap)
    
    # Convert to the format expected by vector_store.py
    documents = []
    for idx, doc in enumerate(chunked_documents):
        documents.append({
            "content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "chunk_id": idx,
                "chunk_size": len(doc.page_content)
            }
        })
    
    return documents


def get_markdown_documents(scspedia_path: str = "scspedia") -> List[Dict[str, Any]]:
    """
    Load full markdown documents without chunking (for smaller documents).
    
    Args:
        scspedia_path: Path to the scspedia folder
    
    Returns:
        List of dictionaries with 'content' and 'metadata' keys
    """
    # Load documents
    raw_documents = load_scspedia_documents(scspedia_path)
    
    # Convert to the format expected by vector_store.py
    documents = []
    for idx, doc in enumerate(raw_documents):
        documents.append({
            "content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "document_id": idx,
                "document_size": len(doc.page_content)
            }
        })
    
    return documents


def get_test_queries() -> List[str]:
    """
    Get test queries based on scspedia content for evaluation.
    
    Returns:
        List of test query strings
    """
    return [
        "What is the Cabinet of Malaysia and Sarawak?",
        "What are the key provisions of the Federal Constitution?",
        "What is the General Order of Sarawak?",
        "What are the highlights of PCDS 2023?",
        "What are the economic sectors in PCDS 2030?",
        "What are the poverty alleviation measures in PCDS 2030?",
        "What is the main report of PCDS 2030?",
        "Who is the Premier of Sarawak?",
        "What is Sarawak Delta Geopark?",
        "What is the Sarawak Digital Economy Blueprint 2030?",
        "What are the facts and figures of Sarawak 2024?",
        "What are the 6 Shared Values of SCS?",
        "What is the Constitution of the State of Sarawak?"
    ]
