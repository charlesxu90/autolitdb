"""PDF processing for RAG ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # pymupdf
from loguru import logger


class PDFProcessor:
    """Process PDFs for RAG database ingestion."""

    def __init__(self):
        """Initialize PDF processor."""
        pass

    def extract_text(self, pdf_path: str | Path) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Extracted text content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def extract_metadata(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract metadata from a PDF file.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with PDF metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {}

        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata or {}
            page_count = len(doc)
            doc.close()

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": page_count,
            }

        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            return {}

    def process_pdf(
        self,
        pdf_path: str | Path,
    ) -> dict[str, Any]:
        """Process a PDF file and extract all information.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Dictionary with text and metadata.
        """
        return {
            "text": self.extract_text(pdf_path),
            "metadata": self.extract_metadata(pdf_path),
            "path": str(pdf_path),
        }

    def process_directory(
        self,
        directory: str | Path,
        pattern: str = "*.pdf",
    ) -> list[dict[str, Any]]:
        """Process all PDFs in a directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern for PDF files.

        Returns:
            List of processed PDF data.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        results = []
        for pdf_path in directory.rglob(pattern):
            logger.debug(f"Processing: {pdf_path}")
            result = self.process_pdf(pdf_path)
            if result["text"]:
                results.append(result)

        logger.info(f"Processed {len(results)} PDFs from {directory}")
        return results
