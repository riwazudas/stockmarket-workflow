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

    def _apply_news(self, payload: dict[str, Any]) -> None:
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

    def _apply_outlook(self, payload: dict[str, Any]) -> None:
        self.overall_outlook = OutlookSummary(
            outlook=payload.get("outlook", "pending"),
            summary=payload.get("summary", ""),
            confidence=payload.get("confidence"),
        )