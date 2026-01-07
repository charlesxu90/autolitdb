"""PubMed literature source implementation."""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

import requests
from loguru import logger

from autolitdb.config import PubMedConfig
from autolitdb.sources.base import Article, LiteratureSource


class PubMedSource(LiteratureSource):
    """PubMed literature source using NCBI E-utilities API."""

    source_name = "pubmed"

    def __init__(self, config: PubMedConfig | None = None):
        """Initialize PubMed source.

        Args:
            config: PubMed configuration. Uses defaults if not provided.
        """
        self.config = config or PubMedConfig()
        self.session = requests.Session()

    def _build_search_url(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        start: int = 0,
        retmax: int = 10000,
        review_filter: str = "both",
    ) -> str:
        """Build ESearch URL.

        Args:
            query: Search query.
            start_date: Start date (YYYY/MM/DD format).
            end_date: End date (YYYY/MM/DD format).
            start: Starting index for results.
            retmax: Maximum results to return.
            review_filter: 'both', 'no_review', or 'only_review'.

        Returns:
            Formatted URL string.
        """
        url = f"{self.config.base_url}/esearch.fcgi?db=pubmed&retmode=json"
        url += f"&retstart={start}&retmax={retmax}"

        # Add date range if specified
        if start_date and end_date:
            url += f"&datetype=pdat&mindate={start_date}&maxdate={end_date}"

        # Handle review filter
        filter_query = query
        if review_filter == "no_review":
            exclude = (
                "NOT Review[Publication Type] "
                "NOT Systematic Review[Publication Type] "
                "NOT Meta-Analysis[Publication Type]"
            )
            filter_query = f"({query}) {exclude}"
        elif review_filter == "only_review":
            include = (
                "(Review[Publication Type] OR "
                "Systematic Review[Publication Type] OR "
                "Meta-Analysis[Publication Type])"
            )
            filter_query = f"({query}) AND {include}"

        url += f"&term={requests.utils.quote(filter_query)}"

        # Add API key if available
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"

        return url

    def _search_batch(
        self,
        query: str,
        start_date: str,
        end_date: str,
        review_filter: str = "both",
    ) -> list[str]:
        """Search PubMed for a specific date range.

        Args:
            query: Search query.
            start_date: Start date (YYYY/MM/DD).
            end_date: End date (YYYY/MM/DD).
            review_filter: Review filter setting.

        Returns:
            List of PMIDs.
        """
        url = self._build_search_url(
            query,
            start_date=start_date,
            end_date=end_date,
            start=0,
            retmax=10000,
            review_filter=review_filter,
        )

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            result = data.get("esearchresult", {})
            count = int(result.get("count", 0))
            ids = result.get("idlist", [])

            if count > 10000:
                logger.warning(
                    f"Query returned {count} results, only first 10000 retrieved. "
                    "Consider narrowing date range."
                )

            return ids

        except requests.RequestException as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    def search(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int | None = None,
        review_filter: str = "both",
        adaptive_batching: bool = True,
        **kwargs,
    ) -> list[str]:
        """Search PubMed for article IDs.

        Args:
            query: PubMed search query.
            start_date: Start date (YYYY/MM/DD format).
            end_date: End date (YYYY/MM/DD format).
            max_results: Maximum number of results to return.
            review_filter: 'both', 'no_review', or 'only_review'.
            adaptive_batching: Whether to use adaptive date batching for large results.

        Returns:
            List of PMIDs.
        """
        # Default to last 5 years if no dates specified
        if not end_date:
            end_date = datetime.now().strftime("%Y/%m/%d")
        if not start_date:
            start_date = (
                datetime.now().replace(year=datetime.now().year - 5).strftime("%Y/%m/%d")
            )

        logger.info(f"Searching PubMed: '{query}' from {start_date} to {end_date}")

        if not adaptive_batching:
            ids = self._search_batch(query, start_date, end_date, review_filter)
            if max_results:
                ids = ids[:max_results]
            logger.info(f"Found {len(ids)} PMIDs")
            return ids

        # Adaptive batching for large result sets
        all_ids = set()
        interval_days = 60  # Start with 60-day intervals
        update_ratio = 2
        max_count = 10000
        min_count = 5000
        max_interval = 365 * 10

        current_start = datetime.strptime(start_date, "%Y/%m/%d")
        final_end = datetime.strptime(end_date, "%Y/%m/%d")

        while current_start < final_end:
            current_end = min(
                current_start + __import__("datetime").timedelta(days=interval_days),
                final_end,
            )

            batch_start = current_start.strftime("%Y/%m/%d")
            batch_end = current_end.strftime("%Y/%m/%d")

            ids = self._search_batch(query, batch_start, batch_end, review_filter)
            count = len(ids)

            logger.debug(f"Batch {batch_start} to {batch_end}: {count} results")

            if count >= max_count and interval_days > 7:
                # Too many results, reduce interval
                interval_days = max(7, interval_days // update_ratio)
                continue

            if count < min_count and interval_days < max_interval:
                # Too few results, increase interval
                interval_days = min(max_interval, interval_days * update_ratio)

            all_ids.update(ids)
            current_start = current_end

            # Rate limiting
            time.sleep(self.config.rate_limit_delay)

            if max_results and len(all_ids) >= max_results:
                break

        result = list(all_ids)
        if max_results:
            result = result[:max_results]

        logger.info(f"Found {len(result)} unique PMIDs")
        return result

    def _parse_medline_entry(self, entry_text: str) -> dict[str, Any]:
        """Parse a single MEDLINE format entry.

        Args:
            entry_text: Raw MEDLINE text for one entry.

        Returns:
            Dictionary with parsed fields.
        """
        fields = {}
        current_field = None
        current_value = []

        for line in entry_text.split("\n"):
            if not line.strip():
                continue

            # Check if this is a new field (4-char tag followed by hyphen and space)
            match = re.match(r"^([A-Z]{2,4})\s*-\s*(.*)$", line)
            if match:
                # Save previous field
                if current_field and current_value:
                    if current_field in fields:
                        if isinstance(fields[current_field], list):
                            fields[current_field].append(" ".join(current_value))
                        else:
                            fields[current_field] = [
                                fields[current_field],
                                " ".join(current_value),
                            ]
                    else:
                        fields[current_field] = " ".join(current_value)

                current_field = match.group(1)
                current_value = [match.group(2)]
            elif line.startswith("      "):  # Continuation line
                current_value.append(line.strip())

        # Save last field
        if current_field and current_value:
            if current_field in fields:
                if isinstance(fields[current_field], list):
                    fields[current_field].append(" ".join(current_value))
                else:
                    fields[current_field] = [
                        fields[current_field],
                        " ".join(current_value),
                    ]
            else:
                fields[current_field] = " ".join(current_value)

        return fields

    def fetch_metadata(
        self,
        ids: list[str],
        batch_size: int | None = None,
        **kwargs,
    ) -> list[Article]:
        """Fetch detailed metadata for PMIDs.

        Args:
            ids: List of PMIDs to fetch.
            batch_size: Number of IDs to fetch per request.

        Returns:
            List of Article objects.
        """
        if not ids:
            return []

        batch_size = batch_size or self.config.batch_size
        articles = []

        logger.info(f"Fetching metadata for {len(ids)} PMIDs")

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            id_str = ",".join(batch_ids)

            url = (
                f"{self.config.base_url}/efetch.fcgi?"
                f"db=pubmed&id={id_str}&rettype=medline&retmode=text"
            )

            if self.config.api_key:
                url += f"&api_key={self.config.api_key}"

            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                content = response.text

                # Split into individual entries
                entries = re.split(r"\n\nPMID-", content)

                for j, entry in enumerate(entries):
                    if not entry.strip():
                        continue

                    # Add back PMID prefix for all but first entry
                    if j > 0:
                        entry = "PMID- " + entry

                    fields = self._parse_medline_entry(entry)

                    # Extract authors
                    authors = []
                    if "AU" in fields:
                        au = fields["AU"]
                        authors = au if isinstance(au, list) else [au]

                    # Extract keywords
                    keywords = []
                    for key in ["MH", "OT"]:  # MeSH terms and other terms
                        if key in fields:
                            kw = fields[key]
                            if isinstance(kw, list):
                                keywords.extend(kw)
                            else:
                                keywords.append(kw)

                    # Parse year from date
                    year = None
                    dp = fields.get("DP", "")
                    year_match = re.search(r"(\d{4})", dp)
                    if year_match:
                        year = int(year_match.group(1))

                    article = Article(
                        id=fields.get("PMID", ""),
                        title=fields.get("TI", ""),
                        abstract=fields.get("AB"),
                        authors=authors,
                        journal=fields.get("JT"),
                        journal_abbrev=fields.get("TA"),
                        year=year,
                        doi=fields.get("LID", "").replace(" [doi]", "")
                        if "[doi]" in fields.get("LID", "")
                        else fields.get("AID", "").replace(" [doi]", "")
                        if "[doi]" in fields.get("AID", "")
                        else None,
                        keywords=keywords,
                        source="pubmed",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{fields.get('PMID', '')}/",
                    )
                    articles.append(article)

            except requests.RequestException as e:
                logger.error(f"Failed to fetch metadata for batch: {e}")
                continue

            # Rate limiting
            time.sleep(self.config.rate_limit_delay)

            logger.debug(f"Fetched {len(articles)}/{len(ids)} articles")

        logger.info(f"Successfully fetched {len(articles)} articles")
        return articles


def search_pubmed(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int | None = None,
    review_filter: str = "both",
    config: PubMedConfig | None = None,
) -> list[Article]:
    """Convenience function to search PubMed and fetch metadata.

    Args:
        query: PubMed search query.
        start_date: Start date (YYYY/MM/DD).
        end_date: End date (YYYY/MM/DD).
        max_results: Maximum results to return.
        review_filter: 'both', 'no_review', or 'only_review'.
        config: PubMed configuration.

    Returns:
        List of Article objects.
    """
    source = PubMedSource(config)
    return source.search_and_fetch(
        query,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
        review_filter=review_filter,
    )
