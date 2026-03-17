from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
import os


def _load_env_files() -> None:
    config_path = Path(__file__).resolve()
    candidate_paths = [
        Path.cwd() / ".env",
        config_path.parents[1] / ".env",
        config_path.parents[2] / ".env",
    ]

    seen: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in seen or not candidate.exists():
            continue
        load_dotenv(candidate, override=False)
        seen.add(candidate)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    default_llm_model: str = "gemini-3-flash-preview"
    enable_llm_request_logging: bool = False
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050
    available_tickers: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_files()

    # Switching LLM models:
    # Set DEFAULT_LLM_MODEL in .env to another Gemini model and restart the app.
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        default_llm_model=os.getenv("DEFAULT_LLM_MODEL", "gemini-3-flash-preview").strip()
        or "gemini-3-flash-preview",
        enable_llm_request_logging=os.getenv("LLM_LOG_REQUESTS", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        dashboard_host=os.getenv("DASHBOARD_HOST", "127.0.0.1").strip() or "127.0.0.1",
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8050")),
    )