from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State, dash_table, dcc, html
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
            figure = _build_price_figure(output)
            return html.Div(
                [
                    html.H3("Stock Prices + Trend"),
                    html.P(f"Trend: {output.get('trend', 'pending')}"),
                    html.P(f"Current price: {output.get('current_price') or 'Not available'}"),
                    dcc.Graph(figure=figure),
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

    return app


def _build_price_figure(output: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    series = output.get("price_series", [])
    predicted_price = output.get("predicted_price", {}).get("predicted_price")

    if series:
        x_values = [item.get("timestamp") for item in series]
        y_values = [item.get("price") for item in series]
        figure.add_trace(
            go.Scatter(x=x_values, y=y_values, mode="lines+markers", name="Observed Price")
        )
        if predicted_price is not None:
            figure.add_trace(
                go.Scatter(
                    x=[x_values[-1]],
                    y=[predicted_price],
                    mode="markers",
                    marker={"size": 12, "symbol": "diamond"},
                    name="Predicted Price",
                )
            )
    else:
        figure.add_annotation(
            text="No price agent output yet. Register a prices agent to populate this chart.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    figure.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=420)
    return figure


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