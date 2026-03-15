from __future__ import annotations

import json
from typing import Any

from google import genai

from stock_ai_system.config.config import Settings, get_settings


class LLMClient:
    """Thin Gemini client wrapper used by agents.

    Real agents can call generate_text() for free-form output or generate_json()
    when they need schema-aligned structured responses.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LLMClient":
        resolved_settings = settings or get_settings()
        return cls(
            api_key=resolved_settings.gemini_api_key,
            model=resolved_settings.default_llm_model,
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

        response = self.client.models.generate_content(
            model=model or self.model,
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