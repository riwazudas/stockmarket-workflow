from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Mapping

import requests

from stock_ai_system.agents.base_agent import BaseAgent
from stock_ai_system.output.output_schema import OutputSchema


logger = logging.getLogger(__name__)


class NewsCollectorAgent(BaseAgent):
    """Fetches real-time stock news from MIAPI /v1/news."""

    def __init__(self, llm_client: Any | None = None) -> None:
        super().__init__(
            name="NewsCollectorAgent",
            output_key="news",
            description="Collects real-time market headlines using MIAPI news search.",
            llm_client=llm_client,
        )

    def run(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        settings = context.get("settings")
        if settings is None:
            return self._fallback_payload(ticker)

        miapi_key = getattr(settings, "miapi_api_key", "")
        miapi_base_url = getattr(settings, "miapi_base_url", "https://api.miapi.uk")
        if not miapi_key:
            return self._fallback_payload(ticker)

        endpoint = f"{miapi_base_url.rstrip('/')}/v1/news"
        headers = {
            "Authorization": f"Bearer {miapi_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": f"Latest news for {ticker} stock",
            "num_results": 20,
        }

        logger.info(
            "MIAPI request | endpoint=%s ticker=%s query=%s num_results=%s",
            endpoint,
            ticker,
            payload["query"],
            payload["num_results"],
        )

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            raw_data = response.json()
            normalized_payload = self._normalize_miapi_response(raw_data, ticker)
            logger.info(
                "MIAPI response | status=%s ticker=%s articles=%s",
                response.status_code,
                ticker,
                len(normalized_payload.get("articles", [])),
            )

        except Exception as exc:
            logger.exception("MIAPI error | ticker=%s error=%s", ticker, exc)
            normalized_payload = self._fallback_payload(ticker)

        return OutputSchema.validate_news_collector_output(normalized_payload, ticker)

    def _normalize_miapi_response(self, raw_data: Any, ticker: str) -> dict[str, Any]:
        articles: list[dict[str, str]] = []
        raw_articles = raw_data.get("articles", []) if isinstance(raw_data, dict) else []

        for item in raw_articles:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue

            source_data = item.get("source", {})
            source_name = ""
            if isinstance(source_data, dict):
                source_name = str(source_data.get("name", "")).strip()

            articles.append(
                {
                    "headline": title,
                    "summary": str(item.get("snippet", "")).strip(),
                    "source": source_name or str(item.get("source", "")).strip(),
                    "date": str(item.get("date", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                }
            )

        return {"stock": ticker, "articles": articles}

    def _fallback_payload(self, ticker: str) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        return {
            "stock": ticker,
            "articles": [
                {
                    "headline": f"Real-time news unavailable for {ticker}.",
                    "summary": "Check MIAPI API key, endpoint availability, or network connectivity.",
                    "source": "System Fallback",
                    "date": today,
                    "url": "",
                }
            ],
        }