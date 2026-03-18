from __future__ import annotations

import argparse
import json
import logging

from stock_ai_system.agents.market_data_agent import MarketDataAgent
from stock_ai_system.agents.news_collector_agent import NewsCollectorAgent
from stock_ai_system.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
from stock_ai_system.config.config import get_settings
from stock_ai_system.dashboard.dashboard_app import create_dashboard_app
from stock_ai_system.pipeline.pipeline_manager import PipelineManager
from stock_ai_system.utils.llm_client import LLMClient


def configure_logging() -> None:
    settings = get_settings()
    if settings.enable_llm_request_logging:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def build_pipeline_manager() -> PipelineManager:
    settings = get_settings()
    llm_client = LLMClient.from_settings(settings)
    # Add agents here as the system grows.
    # Agent order matters when downstream agents consume previous outputs.
    return PipelineManager(
        agents=[
            MarketDataAgent(llm_client=llm_client),
            NewsCollectorAgent(llm_client=llm_client),
            SentimentAnalyzerAgent(llm_client=llm_client),
        ],
        llm_client=llm_client,
        settings=settings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stock AI pipeline and optionally launch the Dash dashboard."
    )
    parser.add_argument(
        "--ticker",
        default=get_settings().available_tickers[0],
        help="Ticker symbol to run through the pipeline.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Plotly Dash dashboard instead of printing JSON output.",
    )
    parser.add_argument(
        "--host",
        default=get_settings().dashboard_host,
        help="Dashboard host when running with --dashboard.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_settings().dashboard_port,
        help="Dashboard port when running with --dashboard.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    settings = get_settings()
    pipeline_manager = build_pipeline_manager()

    if args.dashboard:
        app = create_dashboard_app(pipeline_manager, settings)
        # Dashboard setup:
        # 1. Install requirements from requirements.txt.
        # 2. Ensure GEMINI_API_KEY is available in .env.
        # 3. Run: python main.py --dashboard
        #    Optional: python main.py --dashboard --port 8051
        app.run(host=args.host, port=args.port, debug=args.debug)
        return

    result = pipeline_manager.run(args.ticker)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()