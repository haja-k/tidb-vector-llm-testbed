# TiDB Vector LLM Testbed

Experimental framework for evaluating TiDB's vector search capabilities with LangChain-based LLM retrieval workflows. Includes setup scripts, indexing pipelines, and retrieval benchmarks to test hybrid query performance and relevance scoring on TiDB's vector database engine.

## Overview

This testbed provides a complete solution for benchmarking LLM retrieval over TiDB's vector database using LangChain. It demonstrates:

- **TiDB Connection**: Seamless connection to TiDB clusters
- **Embedding Models**: Support for OpenAI and HuggingFace embeddings
- **Vector Storage**: Efficient vector indexing and storage in TiDB
- **Document Ingestion**: Automated embedding generation and storage
- **LangChain Integration**: Native LangChain Retriever interface
- **Evaluation Metrics**: Comprehensive precision, recall, NDCG, and latency measurements

## Features

- ✅ Modular, clean code with comprehensive comments
- ✅ Support for multiple embedding models (OpenAI, HuggingFace)
- ✅ Automatic vector index creation and management
- ✅ Sample FAQ dataset included for immediate testing
- ✅ LangChain-compatible retriever interface
- ✅ Comprehensive evaluation metrics (Precision@K, Recall@K, F1, NDCG, MRR)
- ✅ Latency benchmarking
- ✅ Easy configuration via environment variables

## Project Structure

```
tidb-vector-llm-testbed/
├── benchmark.py           # Main benchmark orchestration script
├── config.py              # Configuration management
├── db_connection.py       # TiDB connection and table management
├── embedding_models.py    # Embedding model loaders (OpenAI, HuggingFace)
├── vector_store.py        # TiDB vector store integration with LangChain
├── evaluation.py          # Evaluation metrics and reporting
├── sample_data.py         # Sample FAQ dataset for testing
├── requirements.txt       # Python dependencies
├── .env.example           # Example environment configuration
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Access to a TiDB cluster (TiDB Cloud or self-hosted)
- OpenAI API key (if using OpenAI embeddings) or HuggingFace models

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/haja-k/tidb-vector-llm-testbed.git
   cd tidb-vector-llm-testbed
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your TiDB connection details and API keys
   ```

4. **Set up TiDB:**
   - Ensure your TiDB cluster is running and accessible
   - Create a database for the testbed (or use an existing one)
   - Update the `.env` file with your connection details

## Configuration

Edit the `.env` file with your settings:

```bash
# TiDB Connection
TIDB_HOST=your-tidb-host.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=vector_testbed

# Embedding Model Configuration
EMBEDDING_MODEL=ollama  # Options: openai, huggingface, ollama

# OpenAI Settings (only if using OpenAI)
OPENAI_API_KEY=your-openai-api-key

# HuggingFace Settings (only if using HuggingFace)
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Ollama Settings (for locally hosted models - no API key required!)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen:latest
OLLAMA_LLM_MODEL=llama3:latest

# Vector Configuration
VECTOR_DIMENSION=1536  # 1536 for OpenAI, 384 for all-MiniLM-L6-v2, varies for Ollama
TABLE_NAME=documents_vector
```

## Usage

### Running the Complete Benchmark

Run the full benchmark pipeline (all 6 steps):

```bash
python benchmark.py
```

This will execute:
1. ✓ Connect to TiDB cluster
2. ✓ Load embedding model
3. ✓ Create vector index table
4. ✓ Ingest and embed sample documents
5. ✓ Query through LangChain Retriever
6. ✓ Evaluate precision/recall and relevance

### Command-Line Options

```bash
# Drop existing table and recreate
python benchmark.py --drop-existing

# Skip document ingestion (use existing data)
python benchmark.py --skip-ingest

# Use markdown format instead of FAQ format
python benchmark.py --markdown

# Combine options
python benchmark.py --drop-existing --markdown
```

### Using Individual Modules

You can also use individual components in your own scripts:

#### Connect to TiDB
```python
from db_connection import TiDBConnection
from config import Config

db = TiDBConnection()
engine = db.connect()
db.create_vector_table()
```

#### Load Embedding Model
```python
from embedding_models import EmbeddingModelLoader

# Load OpenAI embeddings
embeddings = EmbeddingModelLoader.load_model('openai')

# Or load HuggingFace embeddings
embeddings = EmbeddingModelLoader.load_model('huggingface')
```

#### Ingest Documents
```python
from vector_store import TiDBVectorStoreManager
from sample_data import get_documents

manager = TiDBVectorStoreManager()
manager.initialize()

documents = get_documents()
manager.ingest_documents(documents)
```

#### Query with Retriever
```python
from vector_store import TiDBVectorStoreManager

manager = TiDBVectorStoreManager()
manager.initialize()

retriever = manager.get_retriever(k=5)
results = retriever.get_relevant_documents("What is TiDB vector search?")

for doc in results:
    print(doc.page_content)
```

#### Evaluate Performance
```python
from evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator()

# Evaluate query
metrics = evaluator.evaluate_query(
    query="What is TiDB?",
    retrieved_docs=results,
    relevant_ids=[0, 1, 2],
    k_values=[1, 3, 5]
)

# Measure latency
latency_metrics = evaluator.evaluate_retrieval_latency(
    retriever, 
    "test query", 
    num_runs=5
)
```

## Evaluation Metrics

The benchmark evaluates retrieval performance using:

- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that are retrieved
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Latency**: Mean, median, min, max, and standard deviation

## Sample Output

```
================================================================================
TiDB Vector LLM Testbed - Benchmark Suite
================================================================================

STEP 1: Connecting to TiDB Cluster
================================================================================
✓ Configuration validated
  - Host: gateway01.us-east-1.prod.aws.tidbcloud.com:4000
  - Database: vector_testbed
  - Embedding Model: openai

STEP 2: Loading Embedding Model
================================================================================
Loading openai embedding model...
OpenAI embeddings loaded successfully (dimension: 1536)
✓ Embedding model loaded successfully

STEP 3: Creating Vector Index Table
================================================================================
Connecting to TiDB at gateway01.us-east-1.prod.aws.tidbcloud.com:4000...
Successfully connected to TiDB version: 8.1.0-TiDB-v8.1.0
Creating vector table documents_vector...
Creating vector index for similarity search...
Vector index created successfully.
✓ Vector index table created: documents_vector

STEP 4: Ingesting and Embedding Documents
================================================================================
Loaded 15 sample documents (FAQ dataset)
Ingesting 15 documents...
Successfully ingested 15 documents with embeddings.
✓ Successfully ingested 15 documents with embeddings

STEP 5: Querying Through LangChain Retriever
================================================================================
Retriever created: similarity search with k=5
Running 8 test queries...
✓ Completed 8 queries

STEP 6: Evaluating Retrieval Performance
================================================================================
Mean latency: 45.23 ms
Median latency: 43.10 ms

RETRIEVAL EVALUATION REPORT
================================================================================
Total Queries Evaluated: 8

Average Metrics:
K = 1:
  Precision@1: 1.0000
  Recall@1:    0.3333
  F1@1:        0.5000
  NDCG@1:      1.0000

K = 3:
  Precision@3: 1.0000
  Recall@3:    1.0000
  F1@3:        1.0000
  NDCG@3:      1.0000

Mean Reciprocal Rank (MRR): 1.0000
================================================================================

✓ BENCHMARK COMPLETED SUCCESSFULLY
```

## Extending the Testbed

### Using Markdown Format for Documents

The testbed supports both FAQ format and markdown-based facts:

**FAQ Format (default):**
```bash
python benchmark.py
```

**Markdown Format:**
```bash
python benchmark.py --markdown
```

The markdown format is ideal for storing knowledge bases, documentation, or factual content. Each document can include:
- Headings and subheadings
- Code blocks
- Lists and structured content
- Full markdown syntax

### Adding Custom Datasets

1. Create your document list:
```python
custom_documents = [
    {
        "content": "Your document text here",
        "metadata": {"id": 0, "category": "custom"}
    },
    # ... more documents
]
```

2. For markdown content:
```python
markdown_documents = [
    {
        "content": """# Title
Your markdown content here with **formatting**

- List item 1
- List item 2

```code
Example code
```
""",
        "metadata": {"id": 0, "title": "Title", "format": "markdown"}
    }
]
```

3. Ingest them:
```python
manager.ingest_documents(custom_documents)
```

### Using Different Embedding Models

**Option 1: Ollama (Locally Hosted - No API Key Required)**

Perfect for using models like Qwen, Llama, etc. without any paid services:

1. Install and start Ollama: https://ollama.ai/
2. Pull your desired model: `ollama pull qwen:latest`
3. Update `.env`:
```bash
EMBEDDING_MODEL=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen:latest
OLLAMA_LLM_MODEL=llama3:latest
VECTOR_DIMENSION=1536  # Check your model's dimension
```

**Option 2: HuggingFace Models**

Update `.env` to use HuggingFace models:
```bash
EMBEDDING_MODEL=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-mpnet-base-v2
VECTOR_DIMENSION=768
```

**Option 3: OpenAI**

Update `.env` for OpenAI:
```bash
EMBEDDING_MODEL=openai
OPENAI_API_KEY=your-api-key
VECTOR_DIMENSION=1536
```

### Custom Evaluation

Implement your own relevance judgments:
```python
from evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator()

# Your ground truth data
ground_truth = {
    "query1": [0, 1, 5],  # Relevant document IDs
    "query2": [2, 3, 4],
}

for query, relevant_ids in ground_truth.items():
    results = retriever.get_relevant_documents(query)
    metrics = evaluator.evaluate_query(query, results, relevant_ids)
```

## Troubleshooting

### Connection Issues
- Verify TiDB host and port in `.env`
- Check network connectivity and firewall rules
- Ensure TiDB user has appropriate permissions

### Embedding Issues
- For OpenAI: Verify `OPENAI_API_KEY` is set correctly
- For HuggingFace: First run may download models (requires internet)
- Check `VECTOR_DIMENSION` matches your embedding model

### Performance Issues
- Create vector indexes for better search performance
- Consider batch ingestion for large datasets
- Use appropriate k values for your use case

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms of the LICENSE file.

## Resources

- [TiDB Documentation](https://docs.pingcap.com/tidb/stable)
- [TiDB Vector Search Guide](https://docs.pingcap.com/tidbcloud/vector-search-overview)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.
