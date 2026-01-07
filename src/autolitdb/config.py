"""Configuration management for AutoLitDB."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PubMedConfig(BaseModel):
    """PubMed API configuration."""

    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    batch_size: int = 200
    rate_limit_delay: float = 0.34  # ~3 requests per second
    max_retries: int = 3
    api_key: str | None = None  # Optional NCBI API key for higher rate limits


class LLMConfig(BaseModel):
    """LLM configuration for filtering."""

    provider: str = "vllm"  # vllm, openai, anthropic
    model_name: str = "google/gemma-3-12b-it"
    base_urls: list[str] = Field(default_factory=lambda: ["http://localhost:8000/v1"])
    temperature: float = 0.1
    max_tokens: int = 512
    max_concurrent_requests: int = 32
    batch_size: int = 16


class DownloaderConfig(BaseModel):
    """PDF downloader configuration."""

    server_url: str = "http://localhost:8080"
    download_supplements: bool = True
    timeout: int = 300
    max_concurrent: int = 3


class RAGConfig(BaseModel):
    """RAG database configuration."""

    collection_name: str = "literature"
    persist_directory: str = "./data/chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class Config(BaseModel):
    """Main configuration for AutoLitDB."""

    # Project paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./output"))

    # Module configs
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path(self.rag.persist_directory).mkdir(parents=True, exist_ok=True)


def _replace_env_vars(value: str) -> str:
    """Replace $ENV_VAR patterns with actual environment values."""
    pattern = r"\$([A-Z_][A-Z0-9_]*)"

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, f"${var_name}")

    return re.sub(pattern, replacer, value)


def _process_config_values(obj: Any) -> Any:
    """Recursively process config values to replace environment variables."""
    if isinstance(obj, str):
        return _replace_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    return obj


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Config object with loaded settings.
    """
    if config_path is None:
        return Config()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Process environment variables
    processed_config = _process_config_values(raw_config)

    return Config(**processed_config)
