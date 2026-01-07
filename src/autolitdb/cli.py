"""Command-line interface for AutoLitDB."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from autolitdb.config import load_config
from autolitdb.pipeline import LiteraturePipeline


console = Console()


def setup_logging(verbose: bool = False, log_file: str | None = None):
    """Configure logging."""
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")

    if log_file:
        logger.add(log_file, level="DEBUG", rotation="10 MB")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def main(ctx, verbose: bool, config: str | None):
    """AutoLitDB - Automatic Literature Database

    Query, filter, download, and search academic literature.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    setup_logging(verbose)


@main.command()
@click.argument("query")
@click.option("--start-date", help="Start date (YYYY/MM/DD)")
@click.option("--end-date", help="End date (YYYY/MM/DD)")
@click.option("--max-results", type=int, help="Maximum results")
@click.option(
    "--review-filter",
    type=click.Choice(["both", "no_review", "only_review"]),
    default="both",
    help="Filter for review articles",
)
@click.option("--output", "-o", type=click.Path(), help="Output CSV file")
@click.pass_context
def search(
    ctx,
    query: str,
    start_date: str | None,
    end_date: str | None,
    max_results: int | None,
    review_filter: str,
    output: str | None,
):
    """Search PubMed for literature.

    QUERY is the PubMed search query.

    Example: autolitdb search "enzyme thermostability" --start-date 2022/01/01
    """
    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        articles = pipeline.search_pubmed(
            query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            review_filter=review_filter,
        )

        console.print(f"\n[green]Found {len(articles)} articles[/green]\n")

        if articles:
            # Show first few results
            table = Table(title="Search Results (First 10)")
            table.add_column("PMID", style="cyan")
            table.add_column("Year")
            table.add_column("Title", max_width=60)
            table.add_column("Journal")

            for article in articles[:10]:
                table.add_row(
                    article.id,
                    str(article.year or ""),
                    article.title[:60] + "..." if len(article.title) > 60 else article.title,
                    article.journal_abbrev or "",
                )

            console.print(table)

            if output:
                pipeline._save_articles_csv(articles, output)
                console.print(f"\n[green]Saved to {output}[/green]")

    finally:
        pipeline.close()


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("criteria")
@click.option("--output", "-o", type=click.Path(), help="Output CSV file")
@click.option(
    "--system-prompt",
    type=click.Path(exists=True),
    help="Path to custom system prompt file",
)
@click.pass_context
def filter(
    ctx,
    input_file: str,
    criteria: str,
    output: str | None,
    system_prompt: str | None,
):
    """Filter articles using LLM.

    INPUT_FILE is a CSV file with articles.
    CRITERIA describes what makes an article relevant.

    Example: autolitdb filter articles.csv "Papers about protein engineering for thermostability"
    """
    import pandas as pd
    from autolitdb.sources.base import Article

    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        # Load articles
        df = pd.read_csv(input_file)
        articles = []
        for _, row in df.iterrows():
            articles.append(
                Article(
                    id=str(row.get("id", row.get("PMID", ""))),
                    title=str(row.get("title", row.get("TI", ""))),
                    abstract=str(row.get("abstract", row.get("AB", "")))
                    if pd.notna(row.get("abstract", row.get("AB")))
                    else None,
                    source=str(row.get("source", "unknown")),
                )
            )

        console.print(f"Loaded {len(articles)} articles")

        # Load custom system prompt if provided
        sys_prompt = None
        if system_prompt:
            with open(system_prompt) as f:
                sys_prompt = f.read()

        # Filter
        filtered = pipeline.filter_articles(articles, criteria, sys_prompt)
        relevant = pipeline.get_relevant_articles(filtered)

        console.print(f"\n[green]Relevant: {len(relevant)}/{len(filtered)}[/green]\n")

        if output:
            pipeline._save_articles_csv(filtered, output)
            console.print(f"[green]Saved to {output}[/green]")

    finally:
        pipeline.close()


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--supplements/--no-supplements", default=True, help="Download supplementary materials")
@click.pass_context
def download(ctx, input_file: str, supplements: bool):
    """Download PDFs for articles.

    INPUT_FILE is a CSV file with articles (must have DOI or URL).

    Note: Requires Lite_downloader server running.
    """
    import pandas as pd
    from autolitdb.sources.base import Article

    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        # Load articles
        df = pd.read_csv(input_file)
        articles = []
        for _, row in df.iterrows():
            articles.append(
                Article(
                    id=str(row.get("id", row.get("PMID", ""))),
                    title=str(row.get("title", "")),
                    doi=str(row.get("doi", "")) if pd.notna(row.get("doi")) else None,
                    url=str(row.get("url", "")) if pd.notna(row.get("url")) else None,
                    pdf_url=str(row.get("pdf_url", ""))
                    if pd.notna(row.get("pdf_url"))
                    else None,
                    source=str(row.get("source", "unknown")),
                )
            )

        console.print(f"Loaded {len(articles)} articles for download")

        results = pipeline.download_pdfs(articles, download_supplements=supplements)

        success = sum(1 for r in results if r.success)
        console.print(f"\n[green]Downloaded: {success}/{len(results)}[/green]")

    finally:
        pipeline.close()


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--from-csv", is_flag=True, help="Source is a CSV file of articles")
@click.option("--from-pdf-dir", is_flag=True, help="Source is a directory of PDFs")
@click.pass_context
def index(ctx, source: str, from_csv: bool, from_pdf_dir: bool):
    """Index literature into RAG database.

    SOURCE is either a CSV file (--from-csv) or a directory of PDFs (--from-pdf-dir).
    """
    import pandas as pd
    from autolitdb.sources.base import Article

    if not from_csv and not from_pdf_dir:
        console.print("[red]Specify either --from-csv or --from-pdf-dir[/red]")
        return

    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        if from_csv:
            df = pd.read_csv(source)
            articles = []
            for _, row in df.iterrows():
                articles.append(
                    Article(
                        id=str(row.get("id", row.get("PMID", ""))),
                        title=str(row.get("title", "")),
                        abstract=str(row.get("abstract", ""))
                        if pd.notna(row.get("abstract"))
                        else None,
                        source=str(row.get("source", "unknown")),
                        year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                        journal=str(row.get("journal", ""))
                        if pd.notna(row.get("journal"))
                        else None,
                    )
                )

            chunks = pipeline.index_articles(articles)
            console.print(f"\n[green]Indexed {chunks} chunks from {len(articles)} articles[/green]")

        elif from_pdf_dir:
            chunks = pipeline.index_pdfs(source)
            console.print(f"\n[green]Indexed {chunks} chunks from PDFs[/green]")

    finally:
        pipeline.close()


@main.command()
@click.argument("query")
@click.option("-n", "--num-results", default=10, help="Number of results")
@click.option("--year-min", type=int, help="Minimum publication year")
@click.option("--year-max", type=int, help="Maximum publication year")
@click.pass_context
def query(ctx, query: str, num_results: int, year_min: int | None, year_max: int | None):
    """Query the RAG database.

    QUERY is your search question.

    Example: autolitdb query "What methods improve enzyme thermostability?"
    """
    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        results = pipeline.query_rag(
            query,
            n_results=num_results,
            year_min=year_min,
            year_max=year_max,
        )

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"\n[green]Found {len(results)} relevant passages[/green]\n")

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            console.print(f"[cyan]Result {i}[/cyan]")
            console.print(f"  Article: {metadata.get('title', 'Unknown')[:80]}")
            console.print(f"  Source: {metadata.get('source', 'N/A')} | Year: {metadata.get('year', 'N/A')}")
            console.print(f"  Relevance: {1 - result.get('distance', 0):.3f}")
            console.print(f"  [dim]{result['document'][:300]}...[/dim]\n")

    finally:
        pipeline.close()


@main.command()
@click.pass_context
def stats(ctx):
    """Show RAG database statistics."""
    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        db_stats = pipeline.rag.get_stats()

        table = Table(title="RAG Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("Collection", db_stats["collection_name"])
        table.add_row("Location", db_stats["persist_directory"])
        table.add_row("Total Chunks", str(db_stats["total_chunks"]))
        table.add_row("Unique Articles", str(db_stats["unique_articles"]))

        console.print(table)

        if db_stats["sources"]:
            console.print("\n[cyan]Sources:[/cyan]")
            for source, count in db_stats["sources"].items():
                console.print(f"  {source}: {count} chunks")

    finally:
        pipeline.close()


@main.command()
@click.argument("query")
@click.argument("criteria")
@click.option("--start-date", help="Start date (YYYY/MM/DD)")
@click.option("--end-date", help="End date (YYYY/MM/DD)")
@click.option("--max-results", type=int, help="Maximum articles to retrieve")
@click.option("--download/--no-download", default=True, help="Download PDFs")
@click.option("--supplements/--no-supplements", default=True, help="Download supplements")
@click.option("--output-prefix", default="results", help="Output file prefix")
@click.pass_context
def run(
    ctx,
    query: str,
    criteria: str,
    start_date: str | None,
    end_date: str | None,
    max_results: int | None,
    download: bool,
    supplements: bool,
    output_prefix: str,
):
    """Run the complete pipeline.

    QUERY is the PubMed search query.
    CRITERIA describes what makes an article relevant.

    Example:
        autolitdb run "enzyme thermostability" \\
            "Papers about protein engineering methods to improve thermostability"
    """
    config = load_config(ctx.obj["config_path"])
    pipeline = LiteraturePipeline(config)

    try:
        results = pipeline.run_full_pipeline(
            query=query,
            relevance_criteria=criteria,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            download_pdfs=download,
            download_supplements=supplements,
            output_prefix=output_prefix,
        )

        console.print("\n[green]Pipeline Complete![/green]\n")

        table = Table(title="Pipeline Results")
        table.add_column("Stage", style="cyan")
        table.add_column("Count")

        table.add_row("Articles Found", str(results["total_found"]))
        table.add_row("Relevant After Filtering", str(results["filtered_relevant"]))
        table.add_row("PDFs Downloaded", str(results["downloaded"]))
        table.add_row("Chunks Indexed", str(results["indexed_chunks"]))

        console.print(table)

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
