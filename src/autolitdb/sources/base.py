"""Base class for literature sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Article:
    """Represents a literature article."""

    id: str  # Source-specific ID (e.g., PMID for PubMed)
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    journal_abbrev: str | None = None
    year: int | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    source: str = ""  # e.g., "pubmed", "arxiv"
    url: str | None = None
    pdf_url: str | None = None

    # Fields added during processing
    relevance: str | None = None  # "0" or "1" after LLM filtering
    relevance_reason: str | None = None

    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "journal_abbrev": self.journal_abbrev,
            "year": self.year,
            "doi": self.doi,
            "keywords": self.keywords,
            "source": self.source,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "relevance": self.relevance,
            "relevance_reason": self.relevance_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Article:
        """Create from dictionary."""
        return cls(**data)


class LiteratureSource(ABC):
    """Abstract base class for literature sources."""

    source_name: str = "base"

    @abstractmethod
    def search(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int | None = None,
        **kwargs,
    ) -> list[str]:
        """Search for article IDs matching the query.

        Args:
            query: Search query string.
            start_date: Start date for filtering (format depends on source).
            end_date: End date for filtering.
            max_results: Maximum number of results to return.

        Returns:
            List of article IDs.
        """
        pass

    @abstractmethod
    def fetch_metadata(self, ids: list[str], **kwargs) -> list[Article]:
        """Fetch detailed metadata for article IDs.

        Args:
            ids: List of article IDs to fetch.

        Returns:
            List of Article objects with metadata.
        """
        pass

    def search_and_fetch(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int | None = None,
        **kwargs,
    ) -> list[Article]:
        """Search and fetch metadata in one step.

        Args:
            query: Search query string.
            start_date: Start date for filtering.
            end_date: End date for filtering.
            max_results: Maximum number of results.

        Returns:
            List of Article objects.
        """
        ids = self.search(
            query, start_date=start_date, end_date=end_date, max_results=max_results, **kwargs
        )
        if not ids:
            return []
        return self.fetch_metadata(ids, **kwargs)
