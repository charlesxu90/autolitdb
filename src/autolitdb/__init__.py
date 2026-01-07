"""
AutoLitDB - Automatic Literature Database

A comprehensive tool for:
- Querying literature from PubMed and other sources
- Downloading metadata and filtering with LLMs
- Downloading PDFs and supplementary materials
- Building RAG databases for literature search
"""

__version__ = "0.1.0"

from autolitdb.config import Config, load_config
from autolitdb.pipeline import LiteraturePipeline

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "LiteraturePipeline",
]
