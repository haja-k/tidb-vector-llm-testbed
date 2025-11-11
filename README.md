# TiDB Vector LLM Testbed 🚀

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![TiDB](https://img.shields.io/badge/TiDB-Vector-orange.svg)](https://docs.pingcap.com/tidbcloud/vector-search-overview)
[![LangChain](https://img.shields.io/badge/LangChain-Integrated-green.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A cutting-edge testbed demonstrating advanced vector database capabilities with TiDB, showcasing end-to-end LLM-powered retrieval systems using remote API embeddings.

## 🌟 What This Project Demonstrates

This project showcases expertise in:

- **Vector Databases & AI Integration**: Implementing TiDB's vector search with LangChain for semantic retrieval
- **Full-Stack Data Engineering**: From document ingestion to evaluation metrics
- **Modern Python Development**: Clean, modular code with comprehensive testing
- **Performance Benchmarking**: Latency analysis and relevance scoring
- **Knowledge Base Systems**: Processing and querying large document collections
- **Remote API Integration**: Working with cloud-hosted embedding models

## ✨ Key Features

- 🔗 **Seamless TiDB Integration**: Direct connection to TiDB Cloud or self-hosted clusters
- 🧠 **Remote API Embeddings**: Support for OpenAI-compatible remote embedding models (Qwen, Llama, etc.)
- 📚 **Rich Document Processing**: Markdown-based knowledge base with intelligent chunking
- ⚡ **High-Performance Retrieval**: Optimized vector indexing and similarity search
- 📊 **Comprehensive Evaluation**: Precision, Recall, NDCG, MRR, and latency metrics
- 🛠️ **Modular Architecture**: Clean, extensible codebase for easy customization
- 🔧 **Flexible Configuration**: Environment-based setup with sensible defaults
- 📈 **Benchmarking Suite**: Automated performance testing and reporting

## 🛠️ Tech Stack

- **Database**: TiDB Vector Database
- **AI/ML**: LangChain, Remote API Embeddings (OpenAI-compatible)
- **Backend**: Python 3.12+, SQLAlchemy, PyMySQL
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Development**: Modern Python packaging (pyproject.toml)

## 📁 Project Structure

```
tidb-vector-llm-testbed/
├── 📄 benchmark.py           # Main orchestration script
├── ⚙️ config.py              # Environment configuration
├── 🗄️ db_connection.py       # TiDB connection & schema management
├── 🧠 embedding_models.py    # Remote API embedding model loader
├── 🔍 vector_store.py        # LangChain-compatible vector store
├── 📊 evaluation.py          # Retrieval metrics & benchmarking
├── 📚 sample_data.py         # Document loading & preprocessing
├── 📋 scspedia/              # Knowledge base documents (Sarawak/Malaysia)
├── 📦 pyproject.toml         # Modern Python packaging
├── 📋 requirements.txt       # Dependencies
├── 🔐 .env.example           # Configuration template
└── 📖 README.md              # This file
```

## 🚀 Quick Start

Get up and running in minutes:

```bash
# Clone and setup
git clone https://github.com/haja-k/tidb-vector-llm-testbed.git
cd tidb-vector-llm-testbed

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your TiDB credentials

# Run the complete benchmark
python benchmark.py
```

## 📖 Installation

### Prerequisites

- Python 3.12 or higher
- TiDB cluster (Cloud or self-hosted)
- API keys for remote embedding provider

### Detailed Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/haja-k/tidb-vector-llm-testbed.git
   cd tidb-vector-llm-testbed
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Set up TiDB**
   - Create a TiDB Cloud account or set up self-hosted TiDB
   - Create a database for testing
   - Update `.env` with connection details

## ⚙️ Configuration

The `.env` file supports remote API embedding models:

```bash
# TiDB Connection
TIDB_HOST=your-tidb-host.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=vector_testbed

# Remote API Settings (for Qwen and other OpenAI-compatible models)
REMOTE_EMBEDDING_BASE_URL=https://api.example.com/v1
REMOTE_EMBEDDING_API_KEY=your-embedding-api-key
REMOTE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

REMOTE_LLM_BASE_URL=https://api.example.com/v1
REMOTE_LLM_API_KEY=your-llm-api-key
REMOTE_LLM_MODEL=Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic

# Vector dimensions (must match your model)
VECTOR_DIMENSION=1536
```

## 🎯 Usage

### Complete Benchmark Pipeline

Run the full 6-step workflow:

```bash
python benchmark.py
```

This executes:
1. ✅ Validate configuration
2. ✅ Load embedding model
3. ✅ Create vector tables and indexes
4. ✅ Ingest and embed documents
5. ✅ Set up LangChain retriever
6. ✅ Evaluate performance metrics

### Command Options

```bash
# Fresh start (drop existing data)
python benchmark.py --drop-existing

# Skip ingestion (reuse existing embeddings)
python benchmark.py --skip-ingest

# Use full documents instead of chunks
python benchmark.py --markdown
```

### Programmatic Usage

Use components in your own applications:

```python
from vector_store import TiDBVectorStoreManager
from sample_data import get_documents

# Initialize vector store
manager = TiDBVectorStoreManager()
manager.initialize()

# Load and ingest documents
documents = get_documents()
manager.ingest_documents(documents)

# Create retriever for queries
retriever = manager.get_retriever(k=5)
results = retriever.get_relevant_documents("What is Sarawak?")

for doc in results:
    print(f"Content: {doc.page_content[:200]}...")
```

## 📊 Sample Dataset

The testbed includes a comprehensive knowledge base of **13 documents** about Sarawak, Malaysia:

- Federal Constitution
- State Constitution of Sarawak
- Cabinet and Premier information
- Economic development plans (PCDS 2030)
- Digital economy blueprint
- Cultural and geographical facts
- Government orders and policies

Documents are automatically chunked for optimal retrieval performance.

## 📈 Evaluation Metrics

Comprehensive benchmarking includes:

- **Precision@K & Recall@K**: Relevance accuracy
- **F1 Score**: Balanced precision/recall metric
- **NDCG@K**: Ranking quality assessment
- **MRR**: Mean Reciprocal Rank
- **Latency Analysis**: Response time statistics

## 🏆 Skills Demonstrated

This project highlights proficiency in:

- **Database Engineering**: Vector database design, indexing, and optimization
- **AI/ML Integration**: Embedding models, semantic search, and LLM workflows
- **Software Architecture**: Modular design, dependency injection, and clean code
- **Data Pipeline Development**: ETL processes, document processing, and chunking strategies
- **Performance Engineering**: Benchmarking, latency optimization, and metrics analysis
- **DevOps Practices**: Environment configuration, dependency management, and deployment
- **API Integration**: Working with remote AI services and cloud APIs

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional embedding model support
- Custom evaluation metrics
- UI dashboard for results visualization
- Multi-language document support
- Distributed deployment patterns

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Resources

- [TiDB Vector Search Documentation](https://docs.pingcap.com/tidbcloud/vector-search-overview)
- [LangChain Python Docs](https://python.langchain.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

**Built with ❤️ for demonstrating cutting-edge AI database technologies**

# Vector Configuration
VECTOR_DIMENSION=1536  # Dimension of your remote embedding model
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

embeddings = EmbeddingModelLoader.load_model()
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
  - Embedding Model: Qwen/Qwen3-Embedding-8B

STEP 2: Loading Embedding Model
================================================================================
Loading remote API embedding model...
Remote API embeddings loaded successfully (dimension: 1536)
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
Loaded 13 sample documents (scspedia dataset)
Ingesting 13 documents...
Successfully ingested 13 documents with embeddings.
✓ Successfully ingested 13 documents with embeddings

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

### Using Remote API Embedding Models

The testbed supports OpenAI-compatible remote API models. You can use any embedding service that follows the OpenAI API format, such as:

- **Qwen Models**: Qwen/Qwen3-Embedding-8B
- **Llama Models**: Various Llama-based embedding models
- **Other OpenAI-compatible APIs**: Any service with OpenAI-compatible endpoints

Configure your API provider in `.env`:

```bash
REMOTE_EMBEDDING_BASE_URL=https://your-api-provider.com/v1
REMOTE_EMBEDDING_API_KEY=your-api-key
REMOTE_EMBEDDING_MODEL=your-model-name
VECTOR_DIMENSION=1536  # Check your model's dimension
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
- Verify `REMOTE_EMBEDDING_API_KEY` and `REMOTE_EMBEDDING_BASE_URL` are set correctly
- Ensure your API provider supports OpenAI-compatible endpoints
- Check `VECTOR_DIMENSION` matches your embedding model
- Verify API connectivity and authentication

### Performance Issues
- Create vector indexes for better search performance
- Consider batch ingestion for large datasets
- Use appropriate k values for your use case

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms of the LICENSE file.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
