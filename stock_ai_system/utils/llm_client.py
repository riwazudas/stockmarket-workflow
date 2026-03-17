from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from stock_ai_system.config.config import Settings, get_settings


logger = logging.getLogger(__name__)


class LLMClient:
    """Thin Gemini client wrapper used by agents.

    Real agents can call generate_text() for free-form output or generate_json()
    when they need schema-aligned structured responses.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        enable_request_logging: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.enable_request_logging = enable_request_logging
        self.client = genai.Client(api_key=api_key) if api_key else None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LLMClient":
        resolved_settings = settings or get_settings()
        return cls(
            api_key=resolved_settings.gemini_api_key,
            model=resolved_settings.default_llm_model,
            enable_request_logging=resolved_settings.enable_llm_request_logging,
        )

    def _log_request(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str | None,
        grounded: bool,
        temperature: float | None,
    ) -> None:
        if not self.enable_request_logging:
            return

        logger.info(
            "Gemini request | model=%s grounded=%s prompt_chars=%d system_instruction=%s temperature=%s",
            model,
            grounded,
            len(prompt),
            bool(system_instruction and system_instruction.strip()),
            temperature,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> str:
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured. Set it in .env before calling the LLM.")

        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction

        resolved_model = model or self.model
        self._log_request(
            model=resolved_model,
            prompt=prompt,
            system_instruction=system_instruction,
            grounded=False,
            temperature=None,
        )

        response = self.client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=config or None,
        )
        return getattr(response, "text", "") or ""

    def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        text = self.generate_text(
            prompt=f"{prompt}\n\nReturn valid JSON only.",
            model=model,
            system_instruction=system_instruction,
        )
        cleaned_text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned_text)

    def generate_with_grounding(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Call Gemini with Google Search grounding enabled for real-time results.

        Switching grounding off: call generate_text() instead.
        Switching models: pass model= or change DEFAULT_LLM_MODEL in .env.
        """
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured. Set it in .env before calling the LLM.")

        search_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[search_tool],
            system_instruction=system_instruction,
            temperature=temperature,
        )

        resolved_model = model or self.model
        self._log_request(
            model=resolved_model,
            prompt=prompt,
            system_instruction=system_instruction,
            grounded=True,
            temperature=temperature,
        )

        response = self.client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=config,
        )
        return getattr(response, "text", "") or ""