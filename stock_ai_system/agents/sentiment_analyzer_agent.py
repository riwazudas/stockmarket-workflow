from __future__ import annotations

from typing import Any, Mapping

from stock_ai_system.agents.base_agent import BaseAgent
from stock_ai_system.output.output_schema import OutputSchema


class SentimentAnalyzerAgent(BaseAgent):
    """Analyzes sentiment for news articles returned by NewsCollectorAgent."""

    def __init__(self, llm_client: Any | None = None) -> None:
        super().__init__(
            name="SentimentAnalyzerAgent",
            output_key="sentiment",
            description="Classifies article sentiment and provides reasoning.",
            llm_client=llm_client,
        )

    def run(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        llm_client = self.llm_client or context.get("llm_client")
        news_output = context.get("agent_outputs", {}).get("news", {})
        articles = news_output.get("articles", []) if isinstance(news_output, dict) else []

        if not articles:
            return self._fallback_payload([])

        article_inputs = [
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
            }
            for item in articles
            if isinstance(item, dict)
        ]

        if llm_client is None:
            return self._fallback_payload(article_inputs)

        prompt = (
            f"Analyze sentiment for {ticker} news articles and return strict JSON. "
            "Use only these classes: Positive, Neutral, Negative. "
            "Output schema must be: "
            '{"article_sentiments":[{"headline":"","sentiment":"Positive|Neutral|Negative","reason":""}],'
            '"overall_score":"-1.0_to_1.0","overall_sentiment":"Positive|Neutral|Negative"}.\n\n'
            f"Articles:\n{article_inputs}"
        )

        try:
            payload = llm_client.generate_json(
                prompt=prompt,
                system_instruction=(
                    "You are a financial sentiment analyst. Be concise, objective, and deterministic. "
                    "Return only valid JSON."
                ),
            )
        except Exception as exc:
            print(f"[SentimentAnalyzerAgent] Error analyzing sentiment for {ticker}: {exc}")
            payload = self._fallback_payload(article_inputs)

        return OutputSchema.validate_sentiment_analyzer_output(payload, article_inputs)

    def _fallback_payload(self, article_inputs: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "article_sentiments": [
                {
                    "headline": item.get("headline", ""),
                    "sentiment": "Neutral",
                    "reason": "Fallback used because sentiment model output was unavailable.",
                }
                for item in article_inputs
                if item.get("headline")
            ],
            "overall_score": "0.00",
            "overall_sentiment": "Neutral",
        }