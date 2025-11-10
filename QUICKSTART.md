# Quick Start Guide

Get started with TiDB Vector LLM Testbed in 5 minutes!

## Prerequisites

- Python 3.8+
- TiDB cluster (TiDB Cloud or self-hosted)
- OpenAI API key (or use HuggingFace models)

## Installation

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/haja-k/tidb-vector-llm-testbed.git
cd tidb-vector-llm-testbed
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Minimum required configuration
TIDB_HOST=your-tidb-host.com
TIDB_PORT=4000
TIDB_USER=your-username
TIDB_PASSWORD=your-password
TIDB_DATABASE=vector_testbed

# For OpenAI embeddings (recommended for quick start)
OPENAI_API_KEY=sk-your-api-key-here
EMBEDDING_MODEL=openai
```

### 3. Run the Benchmark

```bash
python benchmark.py
```

That's it! The benchmark will:
- ✓ Connect to your TiDB cluster
- ✓ Load the embedding model
- ✓ Create vector table and index
- ✓ Ingest 15 sample FAQ documents
- ✓ Run 8 test queries
- ✓ Evaluate and report performance metrics

## Expected Output

```
================================================================================
TiDB Vector LLM Testbed - Benchmark Suite
================================================================================

STEP 1: Connecting to TiDB Cluster
✓ Configuration validated
  - Host: your-tidb-host.com:4000
  - Database: vector_testbed
  - Embedding Model: openai

STEP 2: Loading Embedding Model
✓ Embedding model loaded successfully

STEP 3: Creating Vector Index Table
✓ Vector index table created: documents_vector

STEP 4: Ingesting and Embedding Documents
✓ Successfully ingested 15 documents with embeddings

STEP 5: Querying Through LangChain Retriever
✓ Completed 8 queries

STEP 6: Evaluating Retrieval Performance
Mean latency: ~45ms

RETRIEVAL EVALUATION REPORT
Average Metrics:
  Precision@5: 1.0000
  Recall@5:    1.0000
  NDCG@5:      1.0000
  MRR:         1.0000

✓ BENCHMARK COMPLETED SUCCESSFULLY
================================================================================
```

## Alternative: Use HuggingFace (No API Key Required)

If you don't have an OpenAI API key, use HuggingFace models:

```bash
# In .env file
EMBEDDING_MODEL=huggingface
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
```

## Command-Line Options

```bash
# Drop existing table and start fresh
python benchmark.py --drop-existing

# Skip ingestion (use existing data)
python benchmark.py --skip-ingest
```

## Next Steps

### Use Your Own Data

Edit `sample_data.py` or create your own document list:

```python
from vector_store import TiDBVectorStoreManager

manager = TiDBVectorStoreManager()
manager.initialize()

my_documents = [
    {"content": "Your text here", "metadata": {"id": 0}},
    {"content": "More text", "metadata": {"id": 1}},
]

manager.ingest_documents(my_documents)
```

### Query the Vector Store

```python
from vector_store import TiDBVectorStoreManager

manager = TiDBVectorStoreManager()
manager.initialize()

retriever = manager.get_retriever(k=5)
results = retriever.get_relevant_documents("your query here")

for doc in results:
    print(doc.page_content)
```

### Evaluate Performance

```python
from evaluation import RetrievalEvaluator

evaluator = RetrievalEvaluator()

# Measure latency
latency = evaluator.evaluate_retrieval_latency(
    retriever, 
    "test query",
    num_runs=10
)
print(f"Mean latency: {latency['mean_latency_ms']:.2f}ms")

# Evaluate quality
metrics = evaluator.evaluate_query(
    query="What is TiDB?",
    retrieved_docs=results,
    relevant_ids=[0, 1, 2],  # Ground truth
    k_values=[1, 3, 5, 10]
)
```

## Troubleshooting

### "Connection refused" or "Can't connect to MySQL server"
- Check that your TiDB cluster is running
- Verify host, port, and credentials in `.env`
- Check firewall rules and network connectivity

### "No module named 'dotenv'" or similar
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY is required"
- Set your OpenAI API key in `.env`
- Or switch to HuggingFace (see above)

### "Table already exists"
- Use `--drop-existing` flag to recreate table
- Or connect to TiDB and drop the table manually

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review code comments for implementation details
- Open an issue on GitHub for bugs or questions

## Performance Tips

1. **Create indexes before bulk inserts**: Create the table first, insert data, then create vector index
2. **Batch your inserts**: Insert documents in batches of 100-1000 for better performance
3. **Choose the right k**: Start with k=5-10, adjust based on your use case
4. **Monitor latency**: Use the evaluation module to track query performance over time

Happy benchmarking! 🚀
