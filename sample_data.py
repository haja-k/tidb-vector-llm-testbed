"""
Sample FAQ dataset for testing and benchmarking.
Contains common questions and answers about TiDB and vector databases.
"""

SAMPLE_FAQ_DATA = [
    {
        "question": "What is TiDB?",
        "answer": "TiDB is an open-source, distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads. It is MySQL compatible and provides horizontal scalability, strong consistency, and high availability.",
        "category": "General"
    },
    {
        "question": "Does TiDB support vector search?",
        "answer": "Yes, TiDB supports vector search capabilities, allowing you to store and query high-dimensional vector embeddings for similarity search use cases like semantic search, recommendation systems, and RAG applications.",
        "category": "Features"
    },
    {
        "question": "How do I create a vector index in TiDB?",
        "answer": "You can create a vector index in TiDB by using the VECTOR data type for your column and then creating an index with VEC_COSINE_DISTANCE or other vector distance functions. For example: ALTER TABLE documents ADD VECTOR INDEX idx((VEC_COSINE_DISTANCE(embedding)));",
        "category": "Technical"
    },
    {
        "question": "What embedding models work with TiDB vector search?",
        "answer": "TiDB vector search works with any embedding model that produces numerical vectors. Popular choices include OpenAI embeddings (1536 dimensions), sentence-transformers models like all-MiniLM-L6-v2 (384 dimensions), and other models from HuggingFace.",
        "category": "Technical"
    },
    {
        "question": "What is the maximum vector dimension supported by TiDB?",
        "answer": "TiDB supports vector dimensions up to 16,000. This is sufficient for most common embedding models including OpenAI (1536), BERT variants (768), and larger models.",
        "category": "Specifications"
    },
    {
        "question": "How does TiDB compare to dedicated vector databases?",
        "answer": "TiDB combines traditional relational database capabilities with vector search, allowing you to perform hybrid queries that filter on structured data and rank by vector similarity. This eliminates the need for separate databases and simplifies architecture.",
        "category": "Comparison"
    },
    {
        "question": "What distance metrics does TiDB support for vector search?",
        "answer": "TiDB supports multiple distance metrics including cosine distance (VEC_COSINE_DISTANCE), L2 distance (VEC_L2_DISTANCE), and inner product (VEC_NEGATIVE_INNER_PRODUCT) for vector similarity calculations.",
        "category": "Technical"
    },
    {
        "question": "Can I use TiDB for RAG applications?",
        "answer": "Yes, TiDB is well-suited for Retrieval-Augmented Generation (RAG) applications. You can store document embeddings and use vector similarity search to retrieve relevant context for LLM prompts, all while maintaining transactional consistency.",
        "category": "Use Cases"
    },
    {
        "question": "How do I optimize vector search performance in TiDB?",
        "answer": "To optimize vector search performance in TiDB: 1) Create appropriate vector indexes, 2) Use batch operations for insertions, 3) Choose the right distance metric for your use case, 4) Consider dimension reduction if using very high-dimensional vectors, and 5) Leverage TiDB's distributed architecture for scaling.",
        "category": "Performance"
    },
    {
        "question": "Does TiDB support real-time vector search?",
        "answer": "Yes, TiDB supports real-time vector search with low latency. Thanks to its HTAP architecture, you can perform both transactional updates and analytical vector searches on the same dataset without delays or synchronization issues.",
        "category": "Performance"
    },
    {
        "question": "What programming languages can I use with TiDB vector search?",
        "answer": "TiDB is MySQL-compatible, so you can use any MySQL driver or ORM. Popular options include Python (PyMySQL, SQLAlchemy), Java (JDBC), Node.js (mysql2), Go (go-sql-driver), and frameworks like LangChain for LLM applications.",
        "category": "Integration"
    },
    {
        "question": "How do I migrate from Pinecone or Weaviate to TiDB?",
        "answer": "To migrate from dedicated vector databases to TiDB: 1) Export your vectors and metadata, 2) Create a table with VECTOR columns in TiDB, 3) Insert the data using batch operations, 4) Create vector indexes, and 5) Update your application code to use TiDB's SQL interface with vector functions.",
        "category": "Migration"
    },
    {
        "question": "Can TiDB handle hybrid queries with vectors and SQL?",
        "answer": "Yes, TiDB excels at hybrid queries. You can combine traditional SQL WHERE clauses for filtering with ORDER BY vector distance functions for similarity ranking, all in a single query. This enables powerful filtering and ranking in one operation.",
        "category": "Features"
    },
    {
        "question": "Is TiDB vector search production-ready?",
        "answer": "Yes, TiDB's vector search capabilities are production-ready and battle-tested. Many companies use TiDB for production RAG applications, recommendation systems, and semantic search at scale.",
        "category": "General"
    },
    {
        "question": "What are the best practices for storing embeddings in TiDB?",
        "answer": "Best practices for storing embeddings in TiDB: 1) Normalize your vectors before storage, 2) Use appropriate precision (FLOAT vs DOUBLE), 3) Store metadata alongside vectors for filtering, 4) Create indexes after bulk inserts, and 5) Monitor query performance and adjust indexes as needed.",
        "category": "Best Practices"
    }
]


def get_documents():
    """
    Convert FAQ data to document format for ingestion.
    
    Returns:
        List of dictionaries with 'content' and 'metadata' keys
    """
    documents = []
    for idx, faq in enumerate(SAMPLE_FAQ_DATA):
        documents.append({
            "content": f"Q: {faq['question']}\nA: {faq['answer']}",
            "metadata": {
                "id": idx,
                "question": faq['question'],
                "category": faq['category'],
                "type": "faq"
            }
        })
    return documents


def get_test_queries():
    """
    Get test queries for evaluation.
    
    Returns:
        List of test query strings
    """
    return [
        "How does TiDB support vector embeddings?",
        "What is the best way to use TiDB for RAG?",
        "Can I combine SQL filters with vector search?",
        "How do I create vector indexes?",
        "What embedding models are compatible with TiDB?",
        "How does TiDB compare to Pinecone?",
        "What are the performance characteristics of TiDB vector search?",
        "Can I use TiDB for recommendation systems?"
    ]
