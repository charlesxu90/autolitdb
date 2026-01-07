"""Client for Lite_downloader service.

This module provides a client to interact with the Lite_downloader
server for downloading PDFs and supplementary materials from academic
publishers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from autolitdb.config import DownloaderConfig
from autolitdb.sources.base import Article


@dataclass
class DownloadJob:
    """Represents a download job."""

    job_id: str
    status: str  # pending, running, completed, failed
    total_papers: int
    completed_papers: int
    failed_papers: int
    results: list[dict[str, Any]]
    error: str | None
    created_at: datetime | None
    updated_at: datetime | None


@dataclass
class DownloadResult:
    """Result of downloading a single article."""

    article_id: str
    success: bool
    downloaded_files: list[str]
    paper_folder: str | None
    error: str | None


class DownloaderClient:
    """Client for Lite_downloader service.

    This client communicates with a running Lite_downloader server
    to download PDFs and supplementary materials.
    """

    def __init__(self, config: DownloaderConfig | None = None):
        """Initialize downloader client.

        Args:
            config: Downloader configuration. Uses defaults if not provided.
        """
        self.config = config or DownloaderConfig()
        self.client = httpx.Client(
            base_url=self.config.server_url,
            timeout=httpx.Timeout(self.config.timeout, connect=30.0),
        )

    def health_check(self) -> bool:
        """Check if the downloader server is healthy.

        Returns:
            True if server is running and healthy.
        """
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def _build_urls_for_articles(self, articles: list[Article]) -> list[str]:
        """Build download URLs for articles.

        Args:
            articles: List of articles to download.

        Returns:
            List of URLs for downloading.
        """
        urls = []
        for article in articles:
            # Try PDF URL first
            if article.pdf_url:
                urls.append(article.pdf_url)
            # Fall back to DOI
            elif article.doi:
                # Use DOI resolver
                urls.append(f"https://doi.org/{article.doi}")
            # Fall back to PubMed URL for PMC articles
            elif article.source == "pubmed" and article.id:
                urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{article.id}/")
            # Use article URL if available
            elif article.url:
                urls.append(article.url)

        return urls

    def create_job(
        self,
        urls: list[str],
        download_supplements: bool | None = None,
    ) -> str:
        """Create a download job.

        Args:
            urls: List of URLs to download.
            download_supplements: Whether to download supplementary materials.

        Returns:
            Job ID string.
        """
        if download_supplements is None:
            download_supplements = self.config.download_supplements

        payload = {
            "urls": urls,
            "download_supplements": download_supplements,
        }

        response = self.client.post("/download", json=payload)
        response.raise_for_status()
        data = response.json()

        return data["job_id"]

    def get_job_status(self, job_id: str) -> DownloadJob:
        """Get status of a download job.

        Args:
            job_id: Job ID to check.

        Returns:
            DownloadJob with current status.
        """
        response = self.client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        data = response.json()

        return DownloadJob(
            job_id=data["job_id"],
            status=data["status"],
            total_papers=data["total_papers"],
            completed_papers=data["completed_papers"],
            failed_papers=data["failed_papers"],
            results=data.get("results", []),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
        )

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> DownloadJob:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait (None for no limit).

        Returns:
            Final DownloadJob status.

        Raises:
            TimeoutError: If job doesn't complete within timeout.
        """
        start_time = time.time()

        while True:
            job = self.get_job_status(job_id)

            if job.status in ("completed", "failed"):
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            logger.debug(
                f"Job {job_id}: {job.completed_papers}/{job.total_papers} complete"
            )
            time.sleep(poll_interval)

    def list_jobs(self) -> list[DownloadJob]:
        """List all download jobs.

        Returns:
            List of DownloadJob objects.
        """
        response = self.client.get("/jobs")
        response.raise_for_status()
        data = response.json()

        return [
            DownloadJob(
                job_id=j["job_id"],
                status=j["status"],
                total_papers=j["total_papers"],
                completed_papers=j["completed_papers"],
                failed_papers=j["failed_papers"],
                results=j.get("results", []),
                error=j.get("error"),
                created_at=datetime.fromisoformat(j["created_at"])
                if j.get("created_at")
                else None,
                updated_at=datetime.fromisoformat(j["updated_at"])
                if j.get("updated_at")
                else None,
            )
            for j in data.get("jobs", [])
        ]

    def get_downloaded_files(self, job_id: str) -> list[str]:
        """Get list of downloaded files for a job.

        Args:
            job_id: Job ID to get files for.

        Returns:
            List of file paths.
        """
        response = self.client.get(f"/download/{job_id}/files")
        response.raise_for_status()
        data = response.json()
        return data.get("files", [])

    def download_file(
        self,
        file_path: str,
        save_path: str | Path,
    ) -> Path:
        """Download a specific file from the server.

        Args:
            file_path: Path on server.
            save_path: Local path to save file.

        Returns:
            Path to saved file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.client.get(
            f"/download/file/{file_path}",
            follow_redirects=True,
        )
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return save_path

    def download_articles(
        self,
        articles: list[Article],
        download_supplements: bool | None = None,
        wait_for_completion: bool = True,
        poll_interval: float = 5.0,
    ) -> list[DownloadResult]:
        """Download PDFs for articles.

        Args:
            articles: List of articles to download.
            download_supplements: Whether to download supplementary materials.
            wait_for_completion: Whether to wait for job to finish.
            poll_interval: Seconds between status checks.

        Returns:
            List of DownloadResult objects.
        """
        if not articles:
            return []

        urls = self._build_urls_for_articles(articles)
        if not urls:
            logger.warning("No downloadable URLs found for articles")
            return []

        logger.info(f"Creating download job for {len(urls)} URLs")

        job_id = self.create_job(urls, download_supplements)
        logger.info(f"Created download job: {job_id}")

        if not wait_for_completion:
            return []

        job = self.wait_for_job(job_id, poll_interval)

        # Map results back to articles
        results = []
        url_to_article = {self._build_urls_for_articles([a])[0]: a for a in articles if self._build_urls_for_articles([a])}

        for result in job.results:
            url = result.get("url", "")
            article = url_to_article.get(url)
            article_id = article.id if article else url

            results.append(
                DownloadResult(
                    article_id=article_id,
                    success=result.get("success", False),
                    downloaded_files=result.get("downloaded_files", []),
                    paper_folder=result.get("paper_folder"),
                    error=result.get("error"),
                )
            )

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Download complete: {success_count}/{len(results)} successful")

        return results

    def close(self):
        """Close HTTP client."""
        self.client.close()


def download_papers(
    articles: list[Article],
    download_supplements: bool = True,
    config: DownloaderConfig | None = None,
) -> list[DownloadResult]:
    """Convenience function to download papers.

    Args:
        articles: List of articles to download.
        download_supplements: Whether to download supplementary materials.
        config: Downloader configuration.

    Returns:
        List of DownloadResult objects.

    Note:
        Requires a running Lite_downloader server.
    """
    client = DownloaderClient(config)
    try:
        if not client.health_check():
            logger.error(
                f"Downloader server not available at {client.config.server_url}. "
                "Please start the Lite_downloader server first."
            )
            return []

        return client.download_articles(
            articles,
            download_supplements=download_supplements,
        )
    finally:
        client.close()
