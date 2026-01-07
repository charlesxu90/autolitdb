"""RAG database for literature search using ChromaDB."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger

from autolitdb.config import RAGConfig
from autolitdb.sources.base import Article


class LiteratureRAG:
    """RAG database for searching literature."""

    def __init__(self, config: RAGConfig | None = None):
        """Initialize RAG database.

        Args:
            config: RAG configuration. Uses defaults if not provided.
        """
        self.config = config or RAGConfig()

        # Ensure persist directory exists
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Initialized RAG database at {self.config.persist_directory} "
            f"with {self.collection.count()} documents"
        )

    def _generate_chunk_id(self, article_id: str, chunk_index: int) -> str:
        """Generate unique ID for a chunk.

        Args:
            article_id: Article identifier.
            chunk_index: Index of chunk within article.

        Returns:
            Unique chunk ID.
        """
        content = f"{article_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", ".\n", "? ", "!\n"]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.config.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config.chunk_overlap

        return chunks

    def add_article(self, article: Article) -> int:
        """Add an article to the database.

        Args:
            article: Article to add.

        Returns:
            Number of chunks added.
        """
        # Prepare full text
        text_parts = []
        if article.title:
            text_parts.append(f"Title: {article.title}")
        if article.abstract:
            text_parts.append(f"Abstract: {article.abstract}")

        full_text = "\n\n".join(text_parts)
        if not full_text:
            return 0

        # Create chunks
        chunks = self._chunk_text(full_text)
        if not chunks:
            return 0

        # Prepare metadata
        base_metadata = {
            "article_id": article.id,
            "title": article.title or "",
            "source": article.source,
            "year": article.year or 0,
            "journal": article.journal or "",
            "doi": article.doi or "",
            "authors": ", ".join(article.authors[:5]) if article.authors else "",
        }

        # Add chunks to collection
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(article.id, i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk_index": i, "total_chunks": len(chunks)})

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def add_articles(
        self,
        articles: list[Article],
        show_progress: bool = True,
    ) -> int:
        """Add multiple articles to the database.

        Args:
            articles: Articles to add.
            show_progress: Whether to show progress bar.

        Returns:
            Total number of chunks added.
        """
        if not articles:
            return 0

        logger.info(f"Adding {len(articles)} articles to RAG database")

        total_chunks = 0
        iterator = articles
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(articles, desc="Indexing articles")

        for article in iterator:
            try:
                chunks = self.add_article(article)
                total_chunks += chunks
            except Exception as e:
                logger.warning(f"Failed to add article {article.id}: {e}")

        logger.info(f"Added {total_chunks} chunks from {len(articles)} articles")
        return total_chunks

    def add_pdf_content(
        self,
        article_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add PDF content to the database.

        Args:
            article_id: Article identifier.
            content: PDF text content.
            metadata: Additional metadata.

        Returns:
            Number of chunks added.
        """
        if not content:
            return 0

        chunks = self._chunk_text(content)
        if not chunks:
            return 0

        base_metadata = metadata or {}
        base_metadata["article_id"] = article_id
        base_metadata["content_type"] = "pdf"

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(f"{article_id}_pdf", i)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk_index": i, "total_chunks": len(chunks)})

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include_distances: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the database.

        Args:
            query_text: Query text.
            n_results: Maximum number of results.
            where: Optional filter conditions.
            include_distances: Whether to include similarity distances.

        Returns:
            List of result dictionaries.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }

        if where:
            kwargs["where"] = where

        if include_distances:
            kwargs["include"] = ["documents", "metadatas", "distances"]
        else:
            kwargs["include"] = ["documents", "metadatas"]

        results = self.collection.query(**kwargs)

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            if include_distances:
                result["distance"] = results["distances"][0][i]
            formatted.append(result)

        return formatted

    def query_by_article(
        self,
        query_text: str,
        n_results: int = 5,
        year_min: int | None = None,
        year_max: int | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query and return unique articles.

        Args:
            query_text: Query text.
            n_results: Maximum number of articles.
            year_min: Minimum publication year.
            year_max: Maximum publication year.
            source: Filter by source (e.g., "pubmed").

        Returns:
            List of unique articles with best matching chunks.
        """
        # Build filter
        filters = []
        if year_min:
            filters.append({"year": {"$gte": year_min}})
        if year_max:
            filters.append({"year": {"$lte": year_max}})
        if source:
            filters.append({"source": source})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        # Query more chunks than needed to ensure enough unique articles
        raw_results = self.query(
            query_text,
            n_results=n_results * 3,
            where=where,
        )

        # Deduplicate by article
        seen_articles = set()
        unique_results = []

        for result in raw_results:
            article_id = result["metadata"].get("article_id")
            if article_id and article_id not in seen_articles:
                seen_articles.add(article_id)
                unique_results.append(result)

                if len(unique_results) >= n_results:
                    break

        return unique_results

    def delete_article(self, article_id: str) -> int:
        """Delete an article from the database.

        Args:
            article_id: Article identifier.

        Returns:
            Number of chunks deleted.
        """
        # Find all chunks for this article
        results = self.collection.get(
            where={"article_id": article_id},
        )

        if not results["ids"]:
            return 0

        self.collection.delete(ids=results["ids"])
        return len(results["ids"])

    def count(self) -> int:
        """Get total number of documents in database.

        Returns:
            Number of documents.
        """
        return self.collection.count()

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database stats.
        """
        count = self.count()

        # Get unique article count
        results = self.collection.get(include=["metadatas"])
        article_ids = set()
        sources = {}

        for metadata in results.get("metadatas", []):
            if metadata:
                article_id = metadata.get("article_id")
                if article_id:
                    article_ids.add(article_id)
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

        return {
            "total_chunks": count,
            "unique_articles": len(article_ids),
            "sources": sources,
            "collection_name": self.config.collection_name,
            "persist_directory": self.config.persist_directory,
        }
