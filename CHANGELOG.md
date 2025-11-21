# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec:2.0.0.html).

## [Unreleased] - 2025-11-21

### Fixed
- **TiDBVectorStore Parameter**: Changed `embedding` to `embedding_function` in TiDBVectorStore initialization to match LangChain API
- **Embedding API Compatibility**: Replaced built-in `OpenAIEmbeddings` with custom `CustomOpenAIEmbeddings` class for self-hosted Qwen model compatibility
- **Vector Dimension**: Updated default `VECTOR_DIMENSION` from 1536 to 4096 to match Qwen/Qwen3-Embedding-8B model output
- **Database Schema**: Modified table creation to use raw SQL with correct column types (`id` as VARCHAR(255), `document` column) and proper vector dimensions
- **Retriever API**: Updated retriever method calls from deprecated `get_relevant_documents()` to `invoke()` for LangChain compatibility

### Added
- **Custom Embedding Class**: New `CustomOpenAIEmbeddings` class in `embedding_models.py` for handling self-hosted OpenAI-compatible APIs
- **Session Management**: Added `Session` attribute to `TiDBConnection` class for proper SQLAlchemy session handling

### Changed
- **Dependencies**: Updated to use `langchain-openai` instead of deprecated `langchain_community.embeddings.OpenAIEmbeddings`
- **Table Creation**: Switched from SQLAlchemy metadata-based table creation to raw SQL for better vector type handling

### Technical Details
- Fixed 404 API errors with self-hosted Qwen embedding model
- Resolved dimension mismatch errors between embedding model (4096) and database schema
- Fixed column name mismatches (`document` vs `content`, string IDs vs integer)
- Updated LangChain API calls for version compatibility