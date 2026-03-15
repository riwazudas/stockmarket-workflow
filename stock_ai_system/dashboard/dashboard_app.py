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
            return html.Div(
                [
                    html.H3("News Headlines + Sentiment"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Headline", "id": "headline"},
                            {"name": "Source", "id": "source"},
                            {"name": "Date", "id": "date"},
                            {"name": "Sentiment", "id": "sentiment"},
                        ],
                        data=news_rows,
                        style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal"},
                        style_header={"fontWeight": "bold"},
                        style_data_conditional=[
                            {
                                "if": {"filter_query": "{sentiment} = positive", "column_id": "sentiment"},
                                "backgroundColor": "#e8f5e9",
                                "color": "#1b5e20",
                            },
                            {
                                "if": {"filter_query": "{sentiment} = negative", "column_id": "sentiment"},
                                "backgroundColor": "#ffebee",
                                "color": "#b71c1c",
                            },
                            {
                                "if": {"filter_query": "{sentiment} = neutral", "column_id": "sentiment"},
                                "backgroundColor": "#f5f5f5",
                                "color": "#37474f",
                            },
                        ],
                    ),
                    html.P(
                        "Sentiment color coding is placeholder logic for now and will become model-driven in later steps.",
                        style={"marginTop": "8px", "color": "#666"},
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
    raw_articles = agent_news.get("articles", []) if isinstance(agent_news, dict) else []

    rows: list[dict[str, Any]] = []
    for article in raw_articles:
        if not isinstance(article, dict):
            continue
        sentiment = _placeholder_sentiment(article)
        rows.append(
            {
                "headline": article.get("headline", "N/A"),
                "source": article.get("source", "Unknown"),
                "date": article.get("date", ""),
                "sentiment": sentiment,
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
            }
        )
    return rows


def _placeholder_sentiment(article: dict[str, Any]) -> str:
    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
    if any(term in text for term in ("beat", "surge", "up", "gain", "bull")):
        return "positive"
    if any(term in text for term in ("miss", "drop", "down", "loss", "bear")):
        return "negative"
    return "neutral"