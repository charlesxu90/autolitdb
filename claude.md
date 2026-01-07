# Claude Code Instructions for AutoLitDB

## Project Overview

AutoLitDB is an automatic literature database system for querying, downloading, filtering, and searching academic literature. It integrates multiple components:

1. **Literature Sources**: Query PubMed (and potentially other sources) for article metadata
2. **LLM Filtering**: Filter articles by relevance using Gemma-3 via vLLM or OpenAI-compatible APIs
3. **PDF Downloading**: Download PDFs and supplementary materials via the external Lite_downloader service
4. **RAG Database**: Build and query a ChromaDB-based vector database for semantic search

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LiteraturePipeline                        │
│  (src/autolitdb/pipeline.py - main orchestration)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Sources    │  │  Filtering   │  │     Downloader       │  │
│  │  (PubMed)    │→ │  (LLM/vLLM)  │→ │  (Lite_downloader)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                 │                    │                 │
│         └─────────────────┴────────────────────┘                │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │     RAG     │                              │
│                    │  (ChromaDB) │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start Commands

```bash
# Install
pip install -e ".[dev]"

# Install with vLLM support
pip install -e ".[dev,vllm]"

# Run complete pipeline
autolitdb run "enzyme thermostability" \
    "Papers about protein engineering for thermostability" \
    --start-date 2020/01/01 \
    --max-results 500

# Individual commands
autolitdb search "query" --output results.csv
autolitdb filter results.csv "relevance criteria" --output filtered.csv
autolitdb download filtered.csv
autolitdb index --from-csv filtered.csv
autolitdb query "your question"
autolitdb stats
```

## Key Files and Their Purposes

| File | Purpose |
|------|---------|
| `src/autolitdb/pipeline.py` | Main pipeline class orchestrating all components |
| `src/autolitdb/cli.py` | Click-based CLI interface |
| `src/autolitdb/config.py` | Pydantic configuration with YAML loading |
| `src/autolitdb/sources/pubmed.py` | PubMed API integration (NCBI E-utilities) |
| `src/autolitdb/sources/base.py` | `Article` dataclass and `LiteratureSource` ABC |
| `src/autolitdb/filtering/llm_filter.py` | LLM-based article filtering |
| `src/autolitdb/downloader/client.py` | REST client for Lite_downloader service |
| `src/autolitdb/rag/database.py` | ChromaDB vector database wrapper |
| `src/autolitdb/rag/pdf_processor.py` | PDF text extraction with PyMuPDF |

## Key Classes and Entry Points

| Class/Function | File | Purpose |
|----------------|------|---------|
| `LiteraturePipeline` | pipeline.py:20 | Main orchestration class |
| `Config` | config.py:55 | Configuration model |
| `Article` | sources/base.py:10 | Core data model |
| `PubMedSource` | sources/pubmed.py:17 | PubMed API client |
| `LLMFilter` | filtering/llm_filter.py:18 | LLM-based filtering |
| `DownloaderClient` | downloader/client.py:49 | PDF download client |
| `LiteratureRAG` | rag/database.py:17 | ChromaDB RAG database |

## Pipeline Steps (run_full_pipeline)

1. **Search** - Query PubMed and fetch article metadata
2. **Filter** - Use LLM to determine relevance (requires vLLM)
3. **Download** - Download PDFs via Lite_downloader (optional)
4. **Index** - Add articles to ChromaDB RAG database

## External Dependencies

### vLLM Server (Required for LLM filtering)

- Used for running Gemma-3 locally for article filtering
- Can also use any OpenAI-compatible API

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-12b-it \
    --port 8000 \
    --max-model-len 8192
```

### Lite_downloader (Required for PDF downloads)

- **Repository**: Maintained separately at `/home/xux/Desktop/DBAgent/Lite_downloader`
- **Integration**: REST API client in `src/autolitdb/downloader/client.py`
- **Server must be running**: `python cli.py start-server --port 8080`
- **Do not modify** Lite_downloader code from this project

## Configuration

Default config in `config.py`. Override with YAML file:

```yaml
# config/config.yaml
llm:
  provider: vllm  # or openai, anthropic
  model_name: google/gemma-3-12b-it
  base_urls:
    - http://localhost:8000/v1
  max_concurrent_requests: 32

downloader:
  server_url: http://localhost:8080

rag:
  persist_directory: ./data/chroma_db
  chunk_size: 1000
```

- Config uses Pydantic models in `src/autolitdb/config.py`
- Environment variables can be referenced in YAML as `$VAR_NAME`
- Example config at `config/example_config.yaml`

## Development Patterns

### Adding a New Literature Source

1. Create a new file in `src/autolitdb/sources/` (e.g., `arxiv.py`)
2. Inherit from `LiteratureSource` base class
3. Implement `search()` and `fetch_metadata()` methods
4. Return `Article` objects with populated fields
5. Export from `src/autolitdb/sources/__init__.py`

```python
from autolitdb.sources.base import Article, LiteratureSource

class ArxivSource(LiteratureSource):
    source_name = "arxiv"

    def search(self, query, start_date=None, end_date=None, max_results=None, **kwargs):
        # Return list of article IDs
        pass

    def fetch_metadata(self, ids, **kwargs):
        # Return list of Article objects
        pass
```

### Adding CLI Commands

Add commands to `src/autolitdb/cli.py` using Click decorators:

```python
@main.command()
@click.argument("arg")
@click.option("--option", help="Description")
@click.pass_context
def new_command(ctx, arg, option):
    """Command description."""
    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)
    # Implementation
```

## Code Style and Conventions

- **Type hints**: Use throughout, with `from __future__ import annotations`
- **Logging**: Use `loguru.logger` for all logging
- **HTTP clients**: Use `httpx` for sync, `aiohttp` for async
- **Data validation**: Use Pydantic for configs and data models
- **CLI output**: Use `rich` for tables and formatted output
- **Error handling**: Log errors with context, don't silently fail

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=autolitdb tests/

# Type checking
mypy src/autolitdb

# Linting
ruff check src/
```

Tests should:
- Mock external APIs (PubMed, vLLM, Lite_downloader)
- Test data transformations and parsing logic
- Verify CLI commands work correctly

## Common Issues

1. **"Connection refused" on port 8000**: vLLM server not running
2. **"Connection refused" on port 8080**: Lite_downloader server not running
3. **LLM filtering returns 0 relevant**: Check system prompt and relevance criteria

## Common Tasks

### Search PubMed and save results
```bash
autolitdb search "query terms" --start-date 2022/01/01 -o results.csv
```

### Filter articles with LLM
```bash
autolitdb filter results.csv "relevance criteria description" -o filtered.csv
```

### Run full pipeline
```bash
autolitdb run "search query" "relevance criteria" --max-results 500
```

### Query RAG database
```bash
autolitdb query "What methods are used for X?"
```

## Python API Example

```python
from autolitdb import LiteraturePipeline, load_config

config = load_config("config/config.yaml")  # Optional
pipeline = LiteraturePipeline(config)

# Search
articles = pipeline.search_pubmed("enzyme thermostability", max_results=100)

# Filter (requires vLLM server)
filtered = pipeline.filter_articles(articles, "Papers about protein engineering")
relevant = pipeline.get_relevant_articles(filtered)

# Query RAG
results = pipeline.query_rag("What methods improve thermostability?")

pipeline.close()
```

## Future Enhancements (Not Yet Implemented)

- [ ] ArXiv source integration
- [ ] BioRxiv/MedRxiv source integration
- [ ] AI-trend integration for conference paper analysis
- [ ] Async pipeline execution
- [ ] Web UI interface
- [ ] Citation network analysis
- [ ] Batch job scheduling

## Reference Repositories

When implementing new features, refer to:

1. **AutoBioDB** (`/home/xux/Desktop/DBAgent/AutoBioDB`):
   - PubMed querying patterns in `cli_tools/query_pmids.py`
   - Gemma-3 filtering in `cli_tools/filter_articles_gemma3.py`
   - LangGraph workflow patterns in `src/graph/`

2. **Lite_downloader** (`/home/xux/Desktop/DBAgent/Lite_downloader`):
   - Publisher configurations in `src/publishers/`
   - Download strategies in `src/downloader.py`
   - API design in `src/server.py`

3. **AI-trend** (`/home/xux/Desktop/DBAgent/AI-trend`):
   - OpenReview scraping in `data/scrapy_crawl/`
   - Topic assignment in `1.assign_topics.ipynb`
   - Trend analysis patterns
