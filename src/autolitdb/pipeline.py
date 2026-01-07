"""Main pipeline orchestrating all AutoLitDB components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from autolitdb.config import Config, load_config
from autolitdb.downloader.client import DownloaderClient, DownloadResult
from autolitdb.filtering.llm_filter import LLMFilter, DEFAULT_SYSTEM_PROMPT
from autolitdb.rag.database import LiteratureRAG
from autolitdb.rag.pdf_processor import PDFProcessor
from autolitdb.sources.base import Article
from autolitdb.sources.pubmed import PubMedSource


class LiteraturePipeline:
    """Main pipeline for literature processing.

    This pipeline orchestrates:
    1. Querying literature sources (PubMed, etc.)
    2. Filtering with LLMs
    3. Downloading PDFs
    4. Building RAG database
    """

    def __init__(self, config: Config | str | Path | None = None):
        """Initialize pipeline.

        Args:
            config: Config object or path to config file.
        """
        if config is None:
            self.config = Config()
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.config.ensure_directories()

        # Initialize components lazily
        self._pubmed: PubMedSource | None = None
        self._llm_filter: LLMFilter | None = None
        self._downloader: DownloaderClient | None = None
        self._rag: LiteratureRAG | None = None
        self._pdf_processor: PDFProcessor | None = None

    @property
    def pubmed(self) -> PubMedSource:
        """Get PubMed source instance."""
        if self._pubmed is None:
            self._pubmed = PubMedSource(self.config.pubmed)
        return self._pubmed

    @property
    def llm_filter(self) -> LLMFilter:
        """Get LLM filter instance."""
        if self._llm_filter is None:
            self._llm_filter = LLMFilter(self.config.llm)
        return self._llm_filter

    @property
    def downloader(self) -> DownloaderClient:
        """Get downloader client instance."""
        if self._downloader is None:
            self._downloader = DownloaderClient(self.config.downloader)
        return self._downloader

    @property
    def rag(self) -> LiteratureRAG:
        """Get RAG database instance."""
        if self._rag is None:
            self._rag = LiteratureRAG(self.config.rag)
        return self._rag

    @property
    def pdf_processor(self) -> PDFProcessor:
        """Get PDF processor instance."""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor()
        return self._pdf_processor

    def search_pubmed(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int | None = None,
        review_filter: str = "both",
    ) -> list[Article]:
        """Search PubMed and fetch article metadata.

        Args:
            query: PubMed search query.
            start_date: Start date (YYYY/MM/DD).
            end_date: End date (YYYY/MM/DD).
            max_results: Maximum results.
            review_filter: 'both', 'no_review', or 'only_review'.

        Returns:
            List of Article objects.
        """
        return self.pubmed.search_and_fetch(
            query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            review_filter=review_filter,
        )

    def filter_articles(
        self,
        articles: list[Article],
        relevance_criteria: str,
        system_prompt: str | None = None,
    ) -> list[Article]:
        """Filter articles using LLM.

        Args:
            articles: Articles to filter.
            relevance_criteria: Description of relevance criteria.
            system_prompt: Optional custom system prompt.

        Returns:
            Articles with relevance scores.
        """
        return self.llm_filter.filter_articles(
            articles=articles,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            user_prompt=relevance_criteria,
        )

    def get_relevant_articles(self, articles: list[Article]) -> list[Article]:
        """Get only relevant articles from filtered list.

        Args:
            articles: Filtered articles.

        Returns:
            Articles marked as relevant.
        """
        return self.llm_filter.get_relevant_articles(articles)

    def download_pdfs(
        self,
        articles: list[Article],
        download_supplements: bool | None = None,
    ) -> list[DownloadResult]:
        """Download PDFs for articles.

        Args:
            articles: Articles to download.
            download_supplements: Whether to download supplementary materials.

        Returns:
            List of download results.
        """
        if not self.downloader.health_check():
            logger.error(
                f"Downloader server not available at {self.config.downloader.server_url}. "
                "Please start the Lite_downloader server."
            )
            return []

        return self.downloader.download_articles(
            articles,
            download_supplements=download_supplements,
        )

    def index_articles(
        self,
        articles: list[Article],
        show_progress: bool = True,
    ) -> int:
        """Add articles to RAG database.

        Args:
            articles: Articles to index.
            show_progress: Whether to show progress bar.

        Returns:
            Number of chunks added.
        """
        return self.rag.add_articles(articles, show_progress=show_progress)

    def index_pdfs(
        self,
        pdf_directory: str | Path,
        article_id_from_path: bool = True,
    ) -> int:
        """Index PDFs from a directory into RAG database.

        Args:
            pdf_directory: Directory containing PDFs.
            article_id_from_path: Use folder name as article ID.

        Returns:
            Number of chunks added.
        """
        processed = self.pdf_processor.process_directory(pdf_directory)
        total_chunks = 0

        for pdf_data in processed:
            if article_id_from_path:
                # Use parent folder name as article ID
                article_id = Path(pdf_data["path"]).parent.name
            else:
                article_id = Path(pdf_data["path"]).stem

            chunks = self.rag.add_pdf_content(
                article_id=article_id,
                content=pdf_data["text"],
                metadata=pdf_data["metadata"],
            )
            total_chunks += chunks

        return total_chunks

    def query_rag(
        self,
        query: str,
        n_results: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query the RAG database.

        Args:
            query: Query text.
            n_results: Maximum results.
            year_min: Minimum publication year.
            year_max: Maximum publication year.

        Returns:
            List of matching documents.
        """
        return self.rag.query_by_article(
            query,
            n_results=n_results,
            year_min=year_min,
            year_max=year_max,
        )

    def run_full_pipeline(
        self,
        query: str,
        relevance_criteria: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int | None = None,
        download_pdfs: bool = True,
        download_supplements: bool = True,
        save_results: bool = True,
        output_prefix: str = "results",
    ) -> dict[str, Any]:
        """Run the complete literature processing pipeline.

        Args:
            query: Search query.
            relevance_criteria: Criteria for LLM filtering.
            start_date: Start date for search.
            end_date: End date for search.
            max_results: Maximum articles to retrieve.
            download_pdfs: Whether to download PDFs.
            download_supplements: Whether to download supplementary materials.
            save_results: Whether to save results to CSV.
            output_prefix: Prefix for output files.

        Returns:
            Dictionary with pipeline results.
        """
        results = {
            "query": query,
            "total_found": 0,
            "filtered_relevant": 0,
            "downloaded": 0,
            "indexed_chunks": 0,
        }

        # Step 1: Search
        logger.info(f"Step 1: Searching PubMed for '{query}'")
        articles = self.search_pubmed(
            query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
        results["total_found"] = len(articles)
        logger.info(f"Found {len(articles)} articles")

        if not articles:
            return results

        # Step 2: Filter
        logger.info("Step 2: Filtering articles with LLM")
        filtered_articles = self.filter_articles(articles, relevance_criteria)
        relevant_articles = self.get_relevant_articles(filtered_articles)
        results["filtered_relevant"] = len(relevant_articles)
        logger.info(f"Filtered to {len(relevant_articles)} relevant articles")

        # Save filtered results
        if save_results and filtered_articles:
            self._save_articles_csv(
                filtered_articles,
                self.config.output_dir / f"{output_prefix}_filtered.csv",
            )

        # Step 3: Download PDFs
        if download_pdfs and relevant_articles:
            logger.info("Step 3: Downloading PDFs")
            download_results = self.download_pdfs(
                relevant_articles,
                download_supplements=download_supplements,
            )
            results["downloaded"] = sum(1 for r in download_results if r.success)
            logger.info(f"Downloaded {results['downloaded']} papers")

        # Step 4: Index to RAG
        logger.info("Step 4: Indexing to RAG database")
        chunks = self.index_articles(relevant_articles)
        results["indexed_chunks"] = chunks
        logger.info(f"Indexed {chunks} chunks to RAG database")

        # Save final results
        if save_results and relevant_articles:
            self._save_articles_csv(
                relevant_articles,
                self.config.output_dir / f"{output_prefix}_relevant.csv",
            )

        return results

    def _save_articles_csv(
        self,
        articles: list[Article],
        path: str | Path,
    ) -> None:
        """Save articles to CSV file.

        Args:
            articles: Articles to save.
            path: Output file path.
        """
        data = [a.to_dict() for a in articles]
        df = pd.DataFrame(data)

        # Flatten lists
        for col in ["authors", "keywords"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else x
                )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(articles)} articles to {path}")

    def close(self):
        """Clean up resources."""
        if self._llm_filter:
            self._llm_filter.close()
        if self._downloader:
            self._downloader.close()
