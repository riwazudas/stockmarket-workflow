from __future__ import annotations

from datetime import timezone
from typing import Any, Mapping

import pandas as pd
import yfinance as yf

from stock_ai_system.agents.base_agent import BaseAgent
from stock_ai_system.output.output_schema import OutputSchema


class MarketDataAgent(BaseAgent):
    """Fetches intraday and long-term market data and computes indicators."""

    def __init__(self, llm_client: Any | None = None) -> None:
        super().__init__(
            name="MarketDataAgent",
            output_key="prices",
            description=(
                "Fetches 7-day 1m intraday data plus max daily history and computes MA5, MA7, RSI, trend."
            ),
            llm_client=llm_client,
        )

    def run(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        intraday_df = self._download_ohlcv(ticker, period="7d", interval="1m")
        daily_df = self._download_ohlcv(ticker, period="max", interval="1d")

        if intraday_df.empty and daily_df.empty:
            return self._fallback_payload()

        indicator_source = daily_df if not daily_df.empty else intraday_df
        close_series = indicator_source["Close"]
        ma5 = close_series.rolling(window=5, min_periods=1).mean().iloc[-1]
        ma7 = close_series.rolling(window=7, min_periods=1).mean().iloc[-1]
        rsi_series = self._compute_rsi(close_series, period=14)
        latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        latest_close_source = intraday_df if not intraday_df.empty else daily_df
        latest_close = float(latest_close_source["Close"].iloc[-1])
        if latest_close > ma5 > ma7:
            trend = "up"
        elif latest_close < ma5 < ma7:
            trend = "down"
        else:
            trend = "sideways"

        intraday_prices = self._to_price_records(intraday_df)
        daily_prices = self._to_price_records(daily_df)

        payload = {
            "historical_prices": intraday_prices or daily_prices,
            "historical_prices_daily": daily_prices,
            "technical_indicators": {
                "ma5": f"{ma5:.2f}",
                "ma7": f"{ma7:.2f}",
                "rsi": f"{latest_rsi:.2f}",
                "trend": trend,
            },
        }
        return OutputSchema.validate_market_data_output(payload, ticker)

    def _download_ohlcv(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            raw_df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception:
            return pd.DataFrame()

        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        df = raw_df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(1):
                df = df.xs(ticker, axis=1, level=1)
            else:
                df.columns = [str(col[0]) for col in df.columns]

        # Keep only expected OHLCV fields if present.
        columns = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
        if not columns or "Close" not in columns:
            return pd.DataFrame()

        df = df[columns].dropna(subset=["Close"])
        if df.empty:
            return pd.DataFrame()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def _to_price_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        historical_prices: list[dict[str, Any]] = []
        for timestamp, row in df.iterrows():
            historical_prices.append(
                {
                    "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
                    "open": float(row.get("Open", row["Close"])),
                    "high": float(row.get("High", row["Close"])),
                    "low": float(row.get("Low", row["Close"])),
                    "close": float(row["Close"]),
                    "volume": float(row.get("Volume", 0.0)),
                }
            )
        return historical_prices

    def _compute_rsi(self, close_series: pd.Series, period: int = 14) -> pd.Series:
        delta = close_series.diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0.0, 1e-9)
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _fallback_payload(self) -> dict[str, Any]:
        return {
            "historical_prices": [],
            "historical_prices_daily": [],
            "technical_indicators": {
                "ma5": "",
                "ma7": "",
                "rsi": "",
                "trend": "pending",
            },
        }
