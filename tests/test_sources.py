"""Tests for literature sources."""

import pytest

from autolitdb.sources.base import Article
from autolitdb.sources.pubmed import PubMedSource


class TestArticle:
    """Tests for Article dataclass."""

    def test_create_article(self):
        """Test creating an article."""
        article = Article(
            id="12345678",
            title="Test Article",
            abstract="This is a test abstract.",
            source="pubmed",
        )
        assert article.id == "12345678"
        assert article.title == "Test Article"
        assert article.source == "pubmed"

    def test_article_to_dict(self):
        """Test converting article to dictionary."""
        article = Article(
            id="12345678",
            title="Test Article",
            authors=["Author A", "Author B"],
            source="pubmed",
        )
        data = article.to_dict()
        assert data["id"] == "12345678"
        assert data["authors"] == ["Author A", "Author B"]

    def test_article_from_dict(self):
        """Test creating article from dictionary."""
        data = {
            "id": "12345678",
            "title": "Test Article",
            "source": "pubmed",
        }
        article = Article.from_dict(data)
        assert article.id == "12345678"


class TestPubMedSource:
    """Tests for PubMed source."""

    def test_init(self):
        """Test initializing PubMed source."""
        source = PubMedSource()
        assert source.source_name == "pubmed"
        assert source.config.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def test_build_search_url_basic(self):
        """Test building basic search URL."""
        source = PubMedSource()
        url = source._build_search_url("cancer")
        assert "esearch.fcgi" in url
        assert "db=pubmed" in url
        assert "cancer" in url

    def test_build_search_url_with_dates(self):
        """Test building search URL with date range."""
        source = PubMedSource()
        url = source._build_search_url(
            "cancer",
            start_date="2022/01/01",
            end_date="2022/12/31",
        )
        assert "mindate=2022/01/01" in url
        assert "maxdate=2022/12/31" in url

    def test_build_search_url_no_review(self):
        """Test building search URL excluding reviews."""
        source = PubMedSource()
        url = source._build_search_url("cancer", review_filter="no_review")
        assert "NOT%20Review" in url or "NOT Review" in url

    def test_parse_medline_entry(self):
        """Test parsing MEDLINE format entry."""
        source = PubMedSource()
        entry = """PMID- 12345678
TI  - Test Article Title That Is
      Very Long
AB  - This is a test abstract.
AU  - Smith J
AU  - Doe J
DP  - 2022 Jan 15"""

        fields = source._parse_medline_entry(entry)
        assert fields["PMID"] == "12345678"
        assert "Test Article Title" in fields["TI"]
        assert fields["AB"] == "This is a test abstract."
        assert isinstance(fields["AU"], list)
        assert len(fields["AU"]) == 2
