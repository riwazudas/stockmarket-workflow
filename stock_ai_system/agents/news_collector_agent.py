from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from stock_ai_system.agents.base_agent import BaseAgent
from stock_ai_system.output.output_schema import OutputSchema


class NewsCollectorAgent(BaseAgent):
    """Fetches real-time stock news using Gemini Grounding with Google Search."""

    def __init__(self, llm_client: Any | None = None) -> None:
        super().__init__(
            name="NewsCollectorAgent",
            output_key="news",
            description="Collects real-time market headlines using Google Search grounding.",
            llm_client=llm_client,
        )

    def run(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        llm_client = self.llm_client or context.get("llm_client")
        if llm_client is None:
            return self._fallback_payload(ticker)

        prompt = (
            f"Search for the latest financial news for the stock ticker {ticker} "
            f"as of {datetime.now().date()}. "
            "Identify the 30 most significant recent news items from reputable financial sources. "
            "For each article include the actual source URL where available. "
            "Return the data as a strict JSON object with this exact structure: "
            '{"stock":"' + ticker + '","articles":[{"headline":"","summary":"","source":"","date":"YYYY-MM-DD","url":""}]}'
        )

        try:
            # generate_with_grounding enables real-time Google Search results.
            # To switch back to non-grounded output, use llm_client.generate_json() instead.
            raw_text = llm_client.generate_with_grounding(
                prompt=prompt,
                system_instruction=(
                    "You are a professional financial news aggregator. "
                    "Always use Google Search to find real-time events. "
                    "Return ONLY raw JSON. No markdown backticks or extra text."
                ),
                temperature=0.0,
            )

            # Strip markdown fences that the model may include despite instructions.
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1].removeprefix("json").strip()

            payload = json.loads(raw_text)

        except Exception as exc:
            print(f"[NewsCollectorAgent] Error fetching news for {ticker}: {exc}")
            payload = self._fallback_payload(ticker)

        return OutputSchema.validate_news_collector_output(payload, ticker)

    def _fallback_payload(self, ticker: str) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        return {
            "stock": ticker,
            "articles": [
                {
                    "headline": f"Real-time news unavailable for {ticker}.",
                    "summary": "Check network connectivity or API quota for Google Search grounding.",
                    "source": "System Fallback",
                    "date": today,
                }
            ],
        }