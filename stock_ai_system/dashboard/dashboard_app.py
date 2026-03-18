from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State, dash_table, dcc, html
import pandas as pd
import plotly.graph_objects as go

from stock_ai_system.output.output_schema import OutputSchema


def create_dashboard_app(pipeline_manager: Any, settings: Any) -> Dash:
    app = Dash(__name__, title="Stock AI Dashboard", suppress_callback_exceptions=True)
    default_ticker = settings.available_tickers[0]

    app.layout = html.Div(
        style={"fontFamily": "Segoe UI, sans-serif", "padding": "24px", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H1("Multi-Agent Stock Analysis Dashboard"),
            html.P(
                "This dashboard is wired to the pipeline manager. Real agent outputs will appear here once agents are registered."
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "16px"},
                children=[
                    dcc.Dropdown(
                        id="ticker-dropdown",
                        options=[{"label": ticker, "value": ticker} for ticker in settings.available_tickers],
                        value=default_ticker,
                        clearable=False,
                        style={"width": "240px"},
                    ),
                    html.Button("Refresh Pipeline", id="refresh-button", n_clicks=0),
                    html.Div(id="last-refresh", style={"color": "#555"}),
                ],
            ),
            dcc.Store(id="pipeline-output"),
            dcc.Tabs(
                id="output-tabs",
                value="news",
                children=[
                    dcc.Tab(label="News", value="news"),
                    dcc.Tab(label="Prices", value="prices"),
                    dcc.Tab(label="Prediction", value="prediction"),
                    dcc.Tab(label="Outlook", value="outlook"),
                ],
            ),
            html.Div(id="tab-content", style={"paddingTop": "20px"}),
        ],
    )

    @app.callback(
        Output("pipeline-output", "data"),
        Output("last-refresh", "children"),
        Input("ticker-dropdown", "value"),
        Input("refresh-button", "n_clicks"),
        prevent_initial_call=False,
    )
    def refresh_pipeline(ticker: str, _n_clicks: int) -> tuple[dict[str, Any], str]:
        result = pipeline_manager.run(ticker)
        return result.to_dict(), f"Last refresh: {result.generated_at}"

    @app.callback(
        Output("tab-content", "children"),
        Input("output-tabs", "value"),
        State("pipeline-output", "data"),
    )
    def render_tab(active_tab: str, pipeline_output: dict[str, Any] | None) -> html.Div:
        output = pipeline_output or OutputSchema(ticker=default_ticker).to_dict()

        if active_tab == "news":
            news_rows = _build_news_rows(output)
            overall_sentiment = str(output.get("overall_sentiment", "Neutral"))
            overall_score = str(output.get("overall_score", "0.00"))
            kpi_bg = _sentiment_color(overall_sentiment)
            return html.Div(
                [
                    html.H3("News Headlines + Sentiment"),
                    html.Div(
                        [
                            html.Div("Overall Sentiment", style={"fontSize": "13px", "opacity": 0.85}),
                            html.Div(
                                f"{overall_sentiment} ({overall_score})",
                                style={"fontSize": "24px", "fontWeight": 700},
                            ),
                        ],
                        style={
                            "background": kpi_bg,
                            "color": "#111",
                            "padding": "14px 16px",
                            "borderRadius": "10px",
                            "display": "inline-block",
                            "marginBottom": "12px",
                        },
                    ),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Headline", "id": "headline"},
                            {"name": "Source", "id": "source"},
                            {"name": "Date", "id": "date"},
                            {"name": "Sentiment", "id": "sentiment"},
                            {"name": "Link", "id": "url", "presentation": "markdown"},
                        ],
                        data=news_rows,
                        markdown_options={"html": True},
                        page_size=15,
                        page_action="native",
                        sort_action="native",
                        filter_action="native",
                        style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal"},
                        style_cell_conditional=[
                            {"if": {"column_id": "headline"}, "width": "38%"},
                            {"if": {"column_id": "source"}, "width": "12%"},
                            {"if": {"column_id": "date"}, "width": "10%"},
                            {"if": {"column_id": "sentiment"}, "width": "8%"},
                            {"if": {"column_id": "url"}, "width": "7%", "textAlign": "center"},
                        ],
                        style_header={"fontWeight": "bold"},
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{sentiment} = Positive", "column_id": "sentiment"},
                                "backgroundColor": "#e8f5e9",
                                "color": "#1b5e20",
                            },
                            {
                                "if": {"filter_query": "{sentiment} = Negative", "column_id": "sentiment"},
                                "backgroundColor": "#ffebee",
                                "color": "#b71c1c",
                            },
                            {
                                "if": {"filter_query": "{sentiment} = Neutral", "column_id": "sentiment"},
                                "backgroundColor": "#f5f5f5",
                                "color": "#37474f",
                            },
                        ],
                    ),
                ]
            )

        if active_tab == "prices":
            prices_payload = output.get("agent_outputs", {}).get("prices", {})
            indicators = (
                prices_payload.get("technical_indicators", {})
                if isinstance(prices_payload, dict)
                else {}
            )
            trend = str(indicators.get("trend", output.get("trend", "pending"))).lower()
            trend_arrow = _trend_arrow(trend)
            trend_color = _trend_color(trend)
            return html.Div(
                [
                    html.H3("Stock Prices + Trend"),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                            "gap": "10px",
                            "marginBottom": "12px",
                        },
                        children=[
                            _kpi_card("Trend", f"{trend_arrow} {trend.upper()}", trend_color),
                            _kpi_card("MA5", str(indicators.get("ma5", "N/A")), "#f3f8ff"),
                            _kpi_card("MA7", str(indicators.get("ma7", "N/A")), "#f6fff6"),
                            _kpi_card("RSI", str(indicators.get("rsi", "N/A")), "#fff8ef"),
                        ],
                    ),
                    html.Div(
                        [
                            html.Span("Interval", style={"fontWeight": 600, "marginRight": "10px"}),
                            dcc.RadioItems(
                                id="price-interval-buttons",
                                options=[
                                    {"label": "1m", "value": "1m"},
                                    {"label": "5m", "value": "5m"},
                                    {"label": "15m", "value": "15m"},
                                    {"label": "1h", "value": "1h"},
                                    {"label": "1d", "value": "1d"},
                                    {"label": "1wk", "value": "1wk"},
                                    {"label": "1mo", "value": "1mo"},
                                ],
                                value="5m",
                                inline=True,
                                labelStyle={"marginRight": "12px", "cursor": "pointer"},
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(id="price-quick-status", style={"fontSize": "13px", "color": "#455a64"}),
                    dcc.Graph(
                        id="candlestick-chart",
                        config={
                            "displayModeBar": True,
                            "scrollZoom": True,
                            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
                        },
                    ),
                    dcc.Graph(
                        id="close-line-chart",
                        config={"displayModeBar": True, "scrollZoom": True},
                    ),
                    dcc.Graph(
                        id="rsi-chart",
                        config={"displayModeBar": True, "scrollZoom": True},
                    ),
                ]
            )

        if active_tab == "prediction":
            prediction = output.get("predicted_price", {})
            return html.Div(
                [
                    html.H3("Predicted Price"),
                    html.P(f"Predicted price: {prediction.get('predicted_price') or 'Not available'}"),
                    html.P(f"Horizon: {prediction.get('horizon', '1d')}"),
                    html.P(prediction.get("rationale", "")),
                ]
            )

        outlook = output.get("overall_outlook", {})
        return html.Div(
            [
                html.H3("Overall Outlook"),
                html.P(f"Outlook: {outlook.get('outlook', 'pending')}"),
                html.P(outlook.get("summary", "")),
                html.H4("Pipeline Notes"),
                html.Ul([html.Li(note) for note in output.get("notes", [])]),
            ]
        )

    @app.callback(
        Output("candlestick-chart", "figure"),
        Output("close-line-chart", "figure"),
        Output("rsi-chart", "figure"),
        Output("price-quick-status", "children"),
        Input("price-interval-buttons", "value"),
        State("pipeline-output", "data"),
        prevent_initial_call=False,
    )
    def update_price_visuals(
        selected_interval: str,
        pipeline_output: dict[str, Any] | None,
    ) -> tuple[go.Figure, go.Figure, go.Figure, str]:
        output = pipeline_output or OutputSchema(ticker=default_ticker).to_dict()
        raw_df = _build_price_dataframe(output, selected_interval)
        if raw_df.empty:
            empty_candles = go.Figure()
            empty_candles.add_annotation(
                text="No market data available yet.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            empty_line = go.Figure()
            empty_line.add_annotation(
                text="No market data available yet.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            empty_rsi = go.Figure()
            empty_rsi.add_annotation(
                text="RSI unavailable.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return (
                empty_candles,
                empty_line,
                empty_rsi,
                "No raw yfinance data in pipeline output.",
            )

        df = _resample_prices(raw_df, selected_interval)
        indicators = output.get("agent_outputs", {}).get("prices", {}).get("technical_indicators", {})
        candlestick_figure = _build_candlestick_figure(df, selected_interval)
        close_figure = _build_close_figure(df, indicators, selected_interval)
        rsi_figure = _build_rsi_figure(df, selected_interval)

        status = (
            f"Using max daily + 7-day 1m source data and rendering {selected_interval} view "
            f"without re-running the pipeline. Points: {len(df)}"
        )
        return candlestick_figure, close_figure, rsi_figure, status

    return app


def _build_price_dataframe(output: dict[str, Any], interval: str) -> pd.DataFrame:
    prices_payload = output.get("agent_outputs", {}).get("prices", {})
    history: list[dict[str, Any]] = []
    if isinstance(prices_payload, dict):
        if interval in {"1d", "1wk", "1mo"}:
            history = prices_payload.get("historical_prices_daily", [])
        if not history:
            history = prices_payload.get("historical_prices", [])

    if not history:
        legacy_series = output.get("price_series", [])
        if legacy_series:
            rows = []
            for item in legacy_series:
                if not isinstance(item, dict):
                    continue
                close_value = float(item.get("price", 0.0))
                rows.append(
                    {
                        "timestamp": item.get("timestamp", ""),
                        "open": close_value,
                        "high": close_value,
                        "low": close_value,
                        "close": close_value,
                        "volume": 0.0,
                    }
                )
            history = rows

    if not history:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(history)
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    for field in ["open", "high", "low", "close", "volume"]:
        if field not in frame.columns:
            frame[field] = 0.0
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    frame = frame.dropna(subset=["close"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return frame[["open", "high", "low", "close", "volume"]]


def _resample_prices(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    freq_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "1d": "1D",
        "1wk": "1W",
        "1mo": "1ME",
    }
    freq = freq_map.get(interval, "1min")

    if freq == "1min":
        return frame

    resampled = frame.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["close"])


def _build_candlestick_figure(frame: pd.DataFrame, interval: str) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=frame.index,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="OHLC",
            )
        ]
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=380,
        dragmode="pan",
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        template="plotly_white",
        uirevision="candles-static-ui",
    )
    figure.update_xaxes(
        rangeslider={"visible": True},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1D", "step": "day", "stepmode": "backward"},
                {"count": 3, "label": "3D", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
                {"step": "all", "label": "ALL"},
            ]
        },
        type="date",
        fixedrange=False,
        rangebreaks=_market_rangebreaks(interval),
    )
    figure.update_yaxes(fixedrange=False)
    return figure


def _build_close_figure(frame: pd.DataFrame, indicators: dict[str, Any], interval: str) -> go.Figure:
    plot_frame = _downsample_frame(frame, max_points=2500)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=plot_frame.index,
            y=plot_frame["close"],
            mode="lines",
            name="Close",
            line={"color": "#1565c0", "width": 2},
        )
    )

    ma5 = plot_frame["close"].rolling(window=5, min_periods=1).mean()
    ma7 = plot_frame["close"].rolling(window=7, min_periods=1).mean()
    figure.add_trace(
        go.Scattergl(
            x=plot_frame.index,
            y=ma5,
            mode="lines",
            name="MA5",
            line={"color": "#ff8f00", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=plot_frame.index,
            y=ma7,
            mode="lines",
            name="MA7",
            line={"color": "#2e7d32", "width": 1.5},
        )
    )

    trend = str(indicators.get("trend", "pending")).lower()
    arrow = _trend_arrow(trend)
    figure.add_annotation(
        x=plot_frame.index[-1],
        y=float(plot_frame["close"].iloc[-1]),
        text=f"{arrow} trend: {trend}",
        showarrow=True,
        arrowhead=2,
        bgcolor="#ffffff",
        bordercolor="#455a64",
        borderwidth=1,
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=320,
        dragmode="pan",
        hovermode="x unified",
        template="plotly_white",
        uirevision="close-static-ui",
    )
    figure.update_xaxes(type="date", fixedrange=False, rangebreaks=_market_rangebreaks(interval))
    figure.update_yaxes(fixedrange=False)
    return figure


def _build_rsi_figure(frame: pd.DataFrame, interval: str) -> go.Figure:
    plot_frame = _downsample_frame(frame, max_points=2500)
    rsi = _compute_rsi(plot_frame["close"], period=14)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=plot_frame.index,
            y=rsi,
            mode="lines",
            name="RSI",
            line={"color": "#6a1b9a", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(106, 27, 154, 0.1)",
        )
    )
    figure.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought")
    figure.add_hline(y=30, line_dash="dash", line_color="#66bb6a", annotation_text="Oversold")
    figure.update_layout(
        yaxis={"title": "RSI", "range": [0, 100]},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=280,
        dragmode="pan",
        hovermode="x unified",
        template="plotly_white",
        uirevision="rsi-static-ui",
    )
    figure.update_xaxes(type="date", fixedrange=False, rangebreaks=_market_rangebreaks(interval))
    figure.update_yaxes(fixedrange=False)
    return figure


def _market_rangebreaks(interval: str) -> list[dict[str, Any]]:
    if interval in {"1m", "5m", "15m", "1h"}:
        return [
            {"bounds": ["sat", "mon"]},
            {"pattern": "hour", "bounds": [20, 13.5]},
        ]
    return [{"bounds": ["sat", "mon"]}]


def _downsample_frame(frame: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame

    stride = max(1, len(frame) // max_points)
    sampled = frame.iloc[::stride].copy()

    # Always keep the latest point so status annotations align with the newest value.
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]])
    return sampled





def _compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    delta = close_series.diff().fillna(0.0)
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean().replace(0.0, 1e-9)
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _kpi_card(title: str, value: str, background: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 700}),
        ],
        style={
            "background": background,
            "borderRadius": "10px",
            "padding": "12px",
            "border": "1px solid #e0e0e0",
        },
    )


def _trend_arrow(trend: str) -> str:
    if trend == "up":
        return "↑"
    if trend == "down":
        return "↓"
    return "→"


def _trend_color(trend: str) -> str:
    if trend == "up":
        return "#e8f5e9"
    if trend == "down":
        return "#ffebee"
    return "#eceff1"


def _build_news_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    # Prefer canonical NewsCollectorAgent output if present, then fall back to normalized headlines.
    agent_news = output.get("agent_outputs", {}).get("news", {})
    sentiment_output = output.get("agent_outputs", {}).get("sentiment", {})
    raw_articles = agent_news.get("articles", []) if isinstance(agent_news, dict) else []
    article_sentiments = (
        sentiment_output.get("article_sentiments", []) if isinstance(sentiment_output, dict) else []
    )
    sentiment_map = {
        str(item.get("headline", "")): str(item.get("sentiment", "Neutral"))
        for item in article_sentiments
        if isinstance(item, dict)
    }

    rows: list[dict[str, Any]] = []
    for article in raw_articles:
        if not isinstance(article, dict):
            continue
        sentiment = sentiment_map.get(str(article.get("headline", "")), "Neutral")
        url = article.get("url", "").strip()
        url_cell = f"[Open ↗]({url})" if url else ""
        rows.append(
            {
                "headline": article.get("headline", "N/A"),
                "source": article.get("source", "Unknown"),
                "date": article.get("date", ""),
                "sentiment": sentiment,
                "url": url_cell,
            }
        )

    if rows:
        return rows

    for headline_item in output.get("headlines", []):
        if not isinstance(headline_item, dict):
            continue
        rows.append(
            {
                "headline": headline_item.get("headline", "N/A"),
                "source": headline_item.get("source", "Unknown"),
                "date": "",
                "sentiment": headline_item.get("sentiment", "neutral"),
                "url": "",
            }
        )
    return rows


def _sentiment_color(sentiment: str) -> str:
    normalized = sentiment.strip().lower()
    if normalized == "positive":
        return "#e8f5e9"
    if normalized == "negative":
        return "#ffebee"
    return "#f5f5f5"