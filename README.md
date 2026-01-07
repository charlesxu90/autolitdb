# AutoLitDB

Automatic literature query, download, and analysis from diverse sources.

## Features

- **Query Literature**: Search PubMed with adaptive batching for large result sets
- **Download Metadata**: Fetch detailed MEDLINE records (title, abstract, authors, etc.)
- **LLM Filtering**: Filter articles by relevance using Gemma-3 with vLLM (or OpenAI-compatible APIs)
- **PDF Download**: Download PDFs and supplementary materials via Lite_downloader integration
- **RAG Database**: Build and query a ChromaDB-based RAG database for semantic literature search

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/autolitdb.git
cd autolitdb

# Install in development mode
pip install -e ".[dev]"

# Or install with server dependencies
pip install -e ".[dev,server]"
```

## Quick Start

### 1. Search PubMed

```bash
# Search for articles
autolitdb search "enzyme thermostability" --start-date 2022/01/01 --output results.csv

# With review filter
autolitdb search "CRISPR therapy" --review-filter no_review -o crispr_papers.csv
```

### 2. Filter with LLM

First, start a vLLM server with Gemma-3:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-12b-it \
    --port 8000 \
    --max-model-len 8192
```

Then filter:

```bash
autolitdb filter results.csv "Papers about protein engineering methods to improve thermostability" -o filtered.csv
```

### 3. Download PDFs

First, start the Lite_downloader server:

```bash
# In Lite_downloader directory
python cli.py start-server --port 8080
```

Then download:

```bash
autolitdb download filtered.csv --supplements
```

### 4. Build RAG Database

```bash
# Index from CSV
autolitdb index filtered.csv --from-csv

# Or index PDFs directly
autolitdb index ./downloads --from-pdf-dir
```

### 5. Query the Database

```bash
autolitdb query "What methods improve enzyme thermostability?"
```

### 6. Run Complete Pipeline

```bash
autolitdb run "enzyme thermostability" \
    "Papers about protein engineering methods to improve thermostability" \
    --start-date 2020/01/01 \
    --max-results 1000
```

## Python API

```python
from autolitdb import LiteraturePipeline, load_config

# Initialize pipeline
config = load_config("config/config.yaml")  # Optional
pipeline = LiteraturePipeline(config)

# Search PubMed
articles = pipeline.search_pubmed(
    query="enzyme thermostability",
    start_date="2022/01/01",
    max_results=500,
)

# Filter with LLM
filtered = pipeline.filter_articles(
    articles,
    relevance_criteria="Papers about protein engineering for thermostability"
)
relevant = pipeline.get_relevant_articles(filtered)

# Download PDFs (requires Lite_downloader server)
results = pipeline.download_pdfs(relevant)

# Index to RAG database
pipeline.index_articles(relevant)

# Query
results = pipeline.query_rag("What methods improve thermostability?")

# Cleanup
pipeline.close()
```

## Configuration

Copy `config/example_config.yaml` to `config/config.yaml` and customize:

```yaml
# PubMed settings
pubmed:
  api_key: $NCBI_API_KEY  # Optional, for higher rate limits

# LLM settings (vLLM/OpenAI compatible)
llm:
  provider: vllm
  model_name: google/gemma-3-12b-it
  base_urls:
    - http://localhost:8000/v1
  max_concurrent_requests: 32

# Downloader settings
downloader:
  server_url: http://localhost:8080

# RAG settings
rag:
  persist_directory: ./data/chroma_db
  chunk_size: 1000
```

## Project Structure

```
autolitdb/
├── src/autolitdb/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── pipeline.py           # Main pipeline orchestration
│   ├── cli.py                # Command-line interface
│   ├── sources/              # Literature sources
│   │   ├── base.py           # Article dataclass and base source
│   │   └── pubmed.py         # PubMed source implementation
│   ├── filtering/            # LLM filtering
│   │   └── llm_filter.py     # Gemma-3/vLLM filtering
│   ├── downloader/           # PDF downloading
│   │   └── client.py         # Lite_downloader client
│   └── rag/                  # RAG database
│       ├── database.py       # ChromaDB implementation
│       └── pdf_processor.py  # PDF text extraction
├── config/                   # Configuration files
├── prompts/                  # Example prompts
├── data/                     # Data directory (created at runtime)
└── output/                   # Output directory (created at runtime)
```

## Dependencies

### Core Dependencies
- `requests`, `httpx`, `aiohttp` - HTTP clients
- `pandas` - Data processing
- `pydantic` - Configuration validation
- `chromadb` - Vector database
- `langchain` - RAG utilities
- `pymupdf` - PDF processing
- `click`, `rich` - CLI interface
- `loguru` - Logging

### External Services
- **vLLM** (optional): For running Gemma-3 locally
- **Lite_downloader**: For PDF downloading (maintained separately)

## Related Projects

This project integrates concepts and code from:

- [AutoBioDB](../AutoBioDB): PubMed querying and Gemma-3 filtering
- [Lite_downloader](../Lite_downloader): PDF downloading (standalone package)
- [AI-trend](../AI-trend): AI conference trend analysis

## License

MIT
