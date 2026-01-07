"""LLM-based article filtering using vLLM/OpenAI compatible APIs."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from loguru import logger
from tqdm import tqdm

from autolitdb.config import LLMConfig
from autolitdb.sources.base import Article


class LLMFilter:
    """Filter articles using LLM for relevance classification."""

    def __init__(self, config: LLMConfig | None = None):
        """Initialize LLM filter.

        Args:
            config: LLM configuration. Uses defaults if not provided.
        """
        self.config = config or LLMConfig()
        self._server_index = 0

        # Configure HTTP client with retry
        self.client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    def _get_next_server(self) -> str:
        """Get next server URL using round-robin."""
        url = self.config.base_urls[self._server_index]
        self._server_index = (self._server_index + 1) % len(self.config.base_urls)
        return url

    def _format_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        text: str,
    ) -> str | list[dict[str, str]]:
        """Format prompt based on provider.

        Args:
            system_prompt: System instructions.
            user_prompt: User-specific instructions.
            text: Text to analyze (title + abstract).

        Returns:
            Formatted prompt for the API.
        """
        if self.config.provider == "vllm" and "gemma" in self.config.model_name.lower():
            # Gemma-3 instruction format
            return (
                f"<bos><start_of_turn>user\n"
                f"{system_prompt}\n\n"
                f"{user_prompt}\n\n"
                f"Text to analyze:\n{text}\n\n"
                f"Please provide your response as a valid JSON object only.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            # Standard chat format for OpenAI-compatible APIs
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\n\nText to analyze:\n{text}",
                },
            ]

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed JSON dictionary with 'relevance' and 'reason' keys.
        """
        # Remove common formatting artifacts
        response = response.strip()
        response = re.sub(r"<end_of_turn>.*", "", response, flags=re.DOTALL)
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)

        # Try direct JSON parsing
        try:
            result = json.loads(response)
            if "relevance" in result:
                return result
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_patterns = [
            r'\{[^{}]*"relevance"[^{}]*\}',
            r"\{.*?\}",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if "relevance" in result:
                        return result
                except json.JSONDecodeError:
                    continue

        # Default fallback
        logger.warning(f"Could not parse JSON from response: {response[:200]}")
        return {"relevance": "0", "reason": "Failed to parse response"}

    def _process_single(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Process a single text through the LLM.

        Args:
            text: Text to analyze.
            system_prompt: System instructions.
            user_prompt: User-specific instructions.

        Returns:
            Dictionary with 'relevance' and 'reason'.
        """
        base_url = self._get_next_server()

        prompt = self._format_prompt(system_prompt, user_prompt, text)

        # Build request payload
        if isinstance(prompt, str):
            # vLLM completions endpoint
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            endpoint = f"{base_url}/completions"
        else:
            # Chat completions endpoint
            payload = {
                "model": self.config.model_name,
                "messages": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            endpoint = f"{base_url}/chat/completions"

        try:
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract response text
            if "choices" in data:
                if "message" in data["choices"][0]:
                    response_text = data["choices"][0]["message"]["content"]
                else:
                    response_text = data["choices"][0]["text"]
            else:
                response_text = ""

            return self._extract_json_from_response(response_text)

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"relevance": "0", "reason": f"Error: {str(e)}"}

    def filter_articles(
        self,
        articles: list[Article],
        system_prompt: str,
        user_prompt: str,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[Article]:
        """Filter articles using LLM.

        Args:
            articles: List of articles to filter.
            system_prompt: System instructions for the LLM.
            user_prompt: User instructions explaining relevance criteria.
            batch_size: Number of concurrent requests.
            show_progress: Whether to show progress bar.

        Returns:
            List of articles with relevance scores added.
        """
        if not articles:
            return []

        batch_size = batch_size or self.config.batch_size
        max_workers = min(self.config.max_concurrent_requests, len(articles))

        logger.info(f"Filtering {len(articles)} articles using {self.config.model_name}")

        def process_article(article: Article) -> Article:
            text = f"Title: {article.title}"
            if article.abstract:
                text += f"\n\nAbstract: {article.abstract}"

            result = self._process_single(text, system_prompt, user_prompt)
            article.relevance = str(result.get("relevance", "0"))
            article.relevance_reason = result.get("reason", "")
            return article

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_article, a): a for a in articles}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(articles), desc="Filtering")

            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    article = futures[future]
                    article.relevance = "0"
                    article.relevance_reason = f"Processing error: {e}"
                    results.append(article)

        # Count relevant articles
        relevant_count = sum(1 for a in results if a.relevance == "1")
        logger.info(f"Filtering complete: {relevant_count}/{len(results)} relevant")

        return results

    def get_relevant_articles(self, articles: list[Article]) -> list[Article]:
        """Get only relevant articles from filtered list.

        Args:
            articles: List of filtered articles.

        Returns:
            List of articles marked as relevant.
        """
        return [a for a in articles if a.relevance == "1"]

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are an expert research paper classifier. Your task is to determine if a research paper is relevant to a specific research topic based on its title and abstract.

Respond ONLY with a valid JSON object in this exact format:
{
    "relevance": "1" or "0",
    "reason": "Brief explanation of your decision"
}

Where:
- "1" means the paper IS relevant to the specified topic
- "0" means the paper is NOT relevant to the specified topic"""


def filter_articles(
    articles: list[Article],
    relevance_criteria: str,
    system_prompt: str | None = None,
    config: LLMConfig | None = None,
    show_progress: bool = True,
) -> list[Article]:
    """Convenience function to filter articles.

    Args:
        articles: List of articles to filter.
        relevance_criteria: Description of what makes an article relevant.
        system_prompt: Optional custom system prompt.
        config: LLM configuration.
        show_progress: Whether to show progress bar.

    Returns:
        List of articles with relevance scores.
    """
    llm_filter = LLMFilter(config)
    try:
        return llm_filter.filter_articles(
            articles=articles,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            user_prompt=relevance_criteria,
            show_progress=show_progress,
        )
    finally:
        llm_filter.close()
