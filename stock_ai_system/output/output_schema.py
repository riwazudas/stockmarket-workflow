from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class NewsHeadline:
    headline: str
    sentiment: str = "neutral"
    source: str = ""
    confidence: float | None = None


@dataclass
class PricePoint:
    timestamp: str
    price: float


@dataclass
class PricePrediction:
    predicted_price: float | None = None
    horizon: str = "1d"
    rationale: str = "No prediction agent configured yet."


@dataclass
class OutlookSummary:
    outlook: str = "pending"
    summary: str = "No outlook agent configured yet."
    confidence: float | None = None


@dataclass
class OutputSchema:
    ticker: str
    generated_at: str = field(default_factory=_utc_timestamp)
    headlines: list[NewsHeadline] = field(default_factory=list)
    price_series: list[PricePoint] = field(default_factory=list)
    current_price: float | None = None
    trend: str = "pending"
    overall_sentiment: str = "Neutral"
    overall_score: str = "0.00"
    predicted_price: PricePrediction = field(default_factory=PricePrediction)
    overall_outlook: OutlookSummary = field(default_factory=OutlookSummary)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def apply_agent_output(self, output_key: str, payload: dict[str, Any]) -> None:
        self.agent_outputs[output_key] = payload

        if output_key == "news":
            self._apply_news(payload)
        elif output_key == "prices":
            self._apply_prices(payload)
        elif output_key == "prediction":
            self._apply_prediction(payload)
        elif output_key == "sentiment":
            self._apply_sentiment(payload)
        elif output_key == "outlook":
            self._apply_outlook(payload)

    def ensure_defaults(self) -> None:
        if not self.headlines:
            self.headlines.append(
                NewsHeadline(
                    headline="No news agent configured yet.",
                    sentiment="neutral",
                    source="pipeline scaffold",
                )
            )

        if not self.notes:
            self.notes.append("Add agents to replace scaffold placeholders with live analysis.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def validate_news_collector_output(payload: Any, ticker: str) -> dict[str, Any]:
        """Validate and normalize the NewsCollectorAgent payload format."""
        if not isinstance(payload, dict):
            raise TypeError("NewsCollectorAgent output must be a dictionary.")

        stock_value = str(payload.get("stock", ticker)).strip() or ticker
        raw_articles = payload.get("articles", [])
        if not isinstance(raw_articles, list):
            raise TypeError("NewsCollectorAgent output field 'articles' must be a list.")

        articles: list[dict[str, str]] = []
        for item in raw_articles:
            if not isinstance(item, dict):
                continue

            headline = str(item.get("headline", "")).strip()
            summary = str(item.get("summary", "")).strip()
            source = str(item.get("source", "")).strip()
            date_value = str(item.get("date", "")).strip()
            if not headline:
                continue

            url = str(item.get("url", "")).strip()
            articles.append(
                {
                    "headline": headline,
                    "summary": summary,
                    "source": source,
                    "date": date_value,
                    "url": url,
                }
            )

        return {"stock": stock_value, "articles": articles}

    @staticmethod
    def validate_sentiment_analyzer_output(
        payload: Any,
        article_inputs: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Validate and normalize SentimentAnalyzerAgent output shape."""
        if not isinstance(payload, dict):
            raise TypeError("SentimentAnalyzerAgent output must be a dictionary.")

        raw_items = payload.get("article_sentiments", [])
        if not isinstance(raw_items, list):
            raise TypeError("SentimentAnalyzerAgent field 'article_sentiments' must be a list.")

        allowed = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
        normalized_items: list[dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline", "")).strip()
            sentiment_raw = str(item.get("sentiment", "Neutral")).strip().lower()
            reason = str(item.get("reason", "")).strip()
            if not headline:
                continue
            sentiment = allowed.get(sentiment_raw, "Neutral")
            normalized_items.append(
                {
                    "headline": headline,
                    "sentiment": sentiment,
                    "reason": reason,
                }
            )

        if not normalized_items:
            normalized_items = [
                {
                    "headline": item.get("headline", ""),
                    "sentiment": "Neutral",
                    "reason": "No sentiment output available.",
                }
                for item in article_inputs
                if item.get("headline")
            ]

        overall_sentiment_raw = str(payload.get("overall_sentiment", "Neutral")).strip().lower()
        overall_sentiment = allowed.get(overall_sentiment_raw, "Neutral")
        overall_score = str(payload.get("overall_score", "0.00")).strip() or "0.00"

        return {
            "article_sentiments": normalized_items,
            "overall_score": overall_score,
            "overall_sentiment": overall_sentiment,
        }

    @staticmethod
    def validate_market_data_output(payload: Any, ticker: str) -> dict[str, Any]:
        """Validate and normalize MarketDataAgent output shape."""
        if not isinstance(payload, dict):
            raise TypeError("MarketDataAgent output must be a dictionary.")

        def _normalize_price_list(raw_prices: Any, field_name: str) -> list[dict[str, float | str]]:
            if not isinstance(raw_prices, list):
                raise TypeError(f"MarketDataAgent field '{field_name}' must be a list.")

            normalized: list[dict[str, float | str]] = []
            for item in raw_prices:
                if not isinstance(item, dict):
                    continue

                timestamp = str(item.get("timestamp", "")).strip()
                if not timestamp:
                    continue

                try:
                    close_value = float(item.get("close", item.get("price", 0.0)))
                except (TypeError, ValueError):
                    continue

                def _safe_float(value: Any, fallback: float) -> float:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return fallback

                open_value = _safe_float(item.get("open", close_value), close_value)
                high_value = _safe_float(item.get("high", close_value), close_value)
                low_value = _safe_float(item.get("low", close_value), close_value)
                volume_value = _safe_float(item.get("volume", 0.0), 0.0)

                normalized.append(
                    {
                        "timestamp": timestamp,
                        "open": open_value,
                        "high": high_value,
                        "low": low_value,
                        "close": close_value,
                        "volume": volume_value,
                    }
                )
            return normalized

        historical_prices = _normalize_price_list(
            payload.get("historical_prices", []),
            "historical_prices",
        )
        historical_prices_daily = _normalize_price_list(
            payload.get("historical_prices_daily", []),
            "historical_prices_daily",
        )

        indicators = payload.get("technical_indicators", {})
        if not isinstance(indicators, dict):
            raise TypeError("MarketDataAgent field 'technical_indicators' must be a dictionary.")

        trend = str(indicators.get("trend", "pending")).strip().lower()
        if trend not in {"up", "down", "sideways", "pending"}:
            trend = "pending"

        validated_payload = {
            "stock": str(payload.get("stock", ticker)).strip() or ticker,
            "historical_prices": historical_prices,
            "historical_prices_daily": historical_prices_daily,
            "technical_indicators": {
                "ma5": str(indicators.get("ma5", "")).strip(),
                "ma7": str(indicators.get("ma7", "")).strip(),
                "rsi": str(indicators.get("rsi", "")).strip(),
                "trend": trend,
            },
        }
        return validated_payload

    def _apply_news(self, payload: dict[str, Any]) -> None:
        if "articles" in payload:
            payload = self.validate_news_collector_output(payload, self.ticker)
            items = payload.get("articles", [])
            self.headlines = [
                NewsHeadline(
                    headline=item.get("headline", ""),
                    sentiment=item.get("sentiment", "neutral"),
                    source=item.get("source", ""),
                    confidence=None,
                )
                for item in items
                if isinstance(item, dict)
            ]
            return

        items = payload.get("headlines", [])
        self.headlines = [
            NewsHeadline(
                headline=item.get("headline", ""),
                sentiment=item.get("sentiment", "neutral"),
                source=item.get("source", ""),
                confidence=item.get("confidence"),
            )
            for item in items
            if isinstance(item, dict)
        ]

    def _apply_prices(self, payload: dict[str, Any]) -> None:
        # Preferred format for MarketDataAgent.
        if "historical_prices" in payload:
            payload = self.validate_market_data_output(payload, self.ticker)
            history = payload.get("historical_prices", [])
            if not history:
                history = payload.get("historical_prices_daily", [])
            self.price_series = [
                PricePoint(timestamp=item.get("timestamp", ""), price=float(item.get("close", 0.0)))
                for item in history
                if isinstance(item, dict) and item.get("close") is not None
            ]
            self.current_price = (
                float(history[-1].get("close")) if history else self.current_price
            )
            indicators = payload.get("technical_indicators", {})
            self.trend = str(indicators.get("trend", self.trend))
            return

        # Backward-compatible scaffold format.
        series = payload.get("series", [])
        self.price_series = [
            PricePoint(timestamp=item.get("timestamp", ""), price=float(item.get("price", 0.0)))
            for item in series
            if isinstance(item, dict) and item.get("price") is not None
        ]
        self.current_price = payload.get("current_price")
        self.trend = payload.get("trend", self.trend)

    def _apply_prediction(self, payload: dict[str, Any]) -> None:
        self.predicted_price = PricePrediction(
            predicted_price=payload.get("predicted_price"),
            horizon=payload.get("horizon", "1d"),
            rationale=payload.get("rationale", ""),
        )

    def _apply_sentiment(self, payload: dict[str, Any]) -> None:
        validated = self.validate_sentiment_analyzer_output(payload, [])
        sentiment_by_headline = {
            item.get("headline", ""): item.get("sentiment", "Neutral")
            for item in validated.get("article_sentiments", [])
            if isinstance(item, dict)
        }

        self.headlines = [
            NewsHeadline(
                headline=item.headline,
                sentiment=sentiment_by_headline.get(item.headline, item.sentiment),
                source=item.source,
                confidence=item.confidence,
            )
            for item in self.headlines
        ]
        self.overall_score = validated.get("overall_score", "0.00")
        self.overall_sentiment = validated.get("overall_sentiment", "Neutral")

    def _apply_outlook(self, payload: dict[str, Any]) -> None:
        self.overall_outlook = OutlookSummary(
            outlook=payload.get("outlook", "pending"),
            summary=payload.get("summary", ""),
            confidence=payload.get("confidence"),
        )