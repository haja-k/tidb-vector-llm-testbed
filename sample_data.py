"""
Sample dataset for testing and benchmarking.
Loads markdown documents from the scspedia folder and chunks them for vectorization.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document

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
        "Could you please explain how the mining sector is expected to contribute to inclusivity, employment opportunities, and the development of rural communities in Sarawak?",
        "Can you explain how the handling daily operation of the facility is planned for tourism facilities development?",
        "Can u tell me how state-owned universties help with UP-DLP Sarawak?",
        "Apakah tindakan Jabatan Keselamatan dan Kesihatan Pekerjaan terhadap syarikat yang terlibat dalam bahaya utama di SIP?",
        "Boleh jelaskan berapa projek yang telah dilaksanakan oleh Sr Aman Development Agency (SADA) dan jumlah peruntukan yang diterima untuk tahun 2025?",
        "Can you tell me what are the plans for Totally Protected Areas facilities upgrade?",
        "Could you please explain the main strategies and initiatives outlined for the tourism sector in Sarawak?",
        "Can you explain how the Rajah Brooke dynasty influenced the cultural and historical development of the Sarawak Delta Geopark?",
        "Can you explain the repayment terms for an advance to purchase a new vehicle based on the Sarawak General Order 1996?",
        "Can you provide detailed information on how the projects under the tourism sector ensure timely completion with minimal delays?",
        "Datuk Seri Alexander Nanta Linggi tu dia kerja apa dalam Kabinet Malaysia sekarang?",
        "Can you explain in detail the strategies and expected outcomes of the Sarawak Heritage Ordinance administration?",
        "What are the key details and benefits of the Sarawak tourism promotion incentives?",
        "Can you provide details on the initiative for Securing Business Events for Miri?",
        "How is Sarawak planning to enhance food production for export?",
        "What are the key targets and expected economic impacts of the Business Events 2021 to 2025 initiatives in Sarawak?",
        "Could you please explain how the manufacturing sector's initiatives will benefit rural communities in Sarawak?",
        "Can you tell me what facilities will be developed at Limbang Mangrove National Park?",
        "How can I obtain permission for use of content from the PCDS 2030 Highlights 2023 document?",
        "Who is the State Secretary of Sarawak based on Cabinet Members of Malaysia and Sarawak Government?"
    ]
