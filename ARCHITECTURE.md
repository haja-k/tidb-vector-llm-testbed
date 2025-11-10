# Architecture Overview

This document describes the architecture and data flow of the TiDB Vector LLM Testbed.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         benchmark.py                             │
│                    (Main Orchestration)                          │
│  • Runs 6-step workflow                                          │
│  • CLI interface                                                 │
│  • Progress reporting                                            │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ coordinates
             │
     ┌───────┴───────────────────────────────────────┐
     │                                                │
     ▼                                                ▼
┌─────────────────┐                        ┌──────────────────┐
│   config.py     │                        │ sample_data.py   │
│                 │                        │                  │
│ • Environment   │                        │ • 15 FAQ docs    │
│   variables     │                        │ • Test queries   │
│ • Settings      │                        │ • Data helpers   │
└─────────────────┘                        └──────────────────┘
         │
         │ provides config
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Components                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐         ┌─────────────────────┐
│  db_connection.py   │         │ embedding_models.py │
│                     │         │                     │
│ • TiDB connection   │         │ • OpenAI loader     │
│ • Table creation    │         │ • HuggingFace       │
│ • Vector indexes    │         │ • Model factory     │
│ • SQLAlchemy ORM    │         │                     │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │ provides                      │ provides
           │ connection                    │ embeddings
           │                               │
           └───────────┬───────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  vector_store.py    │
            │                     │
            │ • Document ingest   │
            │ • LangChain Retriever│
            │ • Similarity search │
            │ • TiDBVectorStore   │
            └──────────┬──────────┘
                       │
                       │ provides
                       │ retriever
                       │
                       ▼
            ┌─────────────────────┐
            │   evaluation.py     │
            │                     │
            │ • Precision@K       │
            │ • Recall@K          │
            │ • F1, NDCG, MRR     │
            │ • Latency metrics   │
            └─────────────────────┘
```

## Data Flow

### Step 1-3: Setup Phase

```
User Input (.env)
      │
      ▼
  config.py
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
db_connection.py    embedding_models.py
      │                      │
      │ connect()            │ load_model()
      │                      │
      ▼                      ▼
   TiDB                   OpenAI/HF
   Cluster                  Model
      │
      │ create_vector_table()
      │
      ▼
  Vector Table
  (with indexes)
```

### Step 4: Document Ingestion

```
sample_data.py
      │
      │ get_documents()
      │
      ▼
  Documents
  [{"content": "...", "metadata": {...}}, ...]
      │
      ▼
vector_store.py
      │
      │ ingest_documents()
      │
      ├─────────────┐
      │             │
      ▼             ▼
  Embedding    Store in TiDB
  Generation   (content + vector)
      │             │
      └──────┬──────┘
             │
             ▼
        TiDB Vector
        Storage
```

### Step 5: Retrieval Query

```
User Query
      │
      ▼
vector_store.py
      │
      │ get_retriever()
      │
      ▼
LangChain Retriever
      │
      │ get_relevant_documents(query)
      │
      ├─────────────┐
      │             │
      ▼             ▼
  Embed Query   Vector Search
  (via model)   in TiDB
      │             │
      │             │ ORDER BY VEC_COSINE_DISTANCE
      │             │
      └──────┬──────┘
             │
             ▼
      Top K Documents
      [doc1, doc2, ..., docK]
```

### Step 6: Evaluation

```
Query Results
      │
      ▼
evaluation.py
      │
      ├──────────────────┬──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
  Quality Metrics   Ranking Metrics   Performance
  • Precision@K     • NDCG@K           • Latency
  • Recall@K        • MRR              • Throughput
  • F1@K
      │                  │                  │
      └──────────────────┴──────────────────┘
                         │
                         ▼
               Evaluation Report
```

## Module Dependencies

```
benchmark.py
├── config.py
│   └── python-dotenv
│
├── sample_data.py
│
├── db_connection.py
│   ├── config.py
│   ├── sqlalchemy
│   ├── pymysql
│   └── tidb-vector
│
├── embedding_models.py
│   ├── config.py
│   ├── langchain-community
│   ├── openai (optional)
│   └── sentence-transformers (optional)
│
├── vector_store.py
│   ├── config.py
│   ├── db_connection.py
│   ├── embedding_models.py
│   └── langchain-community
│
└── evaluation.py
    ├── numpy
    └── scikit-learn
```

## Key Design Decisions

### 1. Modular Architecture
- Each module has a single, clear responsibility
- Modules can be used independently or together
- Easy to extend or replace components

### 2. Configuration Management
- All settings in one place (config.py)
- Environment variables for secrets
- Validation before execution

### 3. LangChain Integration
- Native LangChain Retriever interface
- Compatible with LangChain RAG pipelines
- Standard document format

### 4. Error Handling
- Try-catch blocks at each step
- Informative error messages
- Graceful degradation where possible

### 5. Evaluation Framework
- Multiple complementary metrics
- Both quality and performance measurements
- Formatted, human-readable reports

## Extension Points

### Adding New Embedding Models

```python
# In embedding_models.py
@staticmethod
def _load_custom_model():
    """Load your custom embedding model."""
    embeddings = CustomEmbeddings(...)
    return embeddings
```

### Adding Custom Metrics

```python
# In evaluation.py
@staticmethod
def calculate_custom_metric(retrieved_ids, relevant_ids):
    """Your custom evaluation metric."""
    # Implementation
    return score
```

### Using Custom Vector Store

```python
# In vector_store.py
class CustomVectorStore(TiDBVectorStoreManager):
    def __init__(self):
        super().__init__()
        # Custom initialization
```

## Performance Considerations

### Indexing
- Vector indexes created automatically
- Uses VEC_COSINE_DISTANCE for efficient similarity search
- Indexes created after bulk inserts for better performance

### Connection Pooling
- SQLAlchemy connection pooling enabled
- Pool recycling every 3600 seconds
- Pre-ping to verify connections

### Batch Processing
- Documents can be ingested in batches
- Embeddings generated in parallel where possible
- Configurable batch sizes for large datasets

## Security

### Credential Management
- All credentials via environment variables
- No hardcoded secrets
- .env file excluded from git

### SQL Injection Prevention
- SQLAlchemy ORM for data operations
- Parameterized queries with text()
- No string concatenation for SQL

### Input Validation
- Configuration validation on startup
- Type checking for parameters
- Error handling for invalid inputs

## Testing Strategy

### Validation Script
- Checks project structure
- Verifies imports (with graceful handling of missing deps)
- Validates sample data format

### Manual Testing
- Run benchmark with sample data
- Verify each step completes successfully
- Check evaluation metrics are reasonable

### Integration Testing
- Full end-to-end workflow
- Multiple embedding models
- Different configuration scenarios
