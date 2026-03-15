from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stock_ai_system.agents.base_agent import BaseAgent
from stock_ai_system.output.output_schema import OutputSchema


class PipelineManager:
    """Runs registered agents and assembles a shared structured output."""

    def __init__(
        self,
        agents: Sequence[BaseAgent] | None = None,
        llm_client: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.settings = settings
        self._agents: list[BaseAgent] = []

        for agent in agents or []:
            self.register_agent(agent)

    @property
    def agents(self) -> tuple[BaseAgent, ...]:
        return tuple(self._agents)

    def register_agent(self, agent: BaseAgent) -> None:
        if any(existing.output_key == agent.output_key for existing in self._agents):
            raise ValueError(f"Duplicate agent output_key: {agent.output_key}")
        if agent.llm_client is None:
            agent.llm_client = self.llm_client
        self._agents.append(agent)

    def run(self, ticker: str) -> OutputSchema:
        result = OutputSchema(ticker=ticker)
        context: dict[str, Any] = {
            "ticker": ticker,
            "llm_client": self.llm_client,
            "settings": self.settings,
            "agent_outputs": {},
        }

        if not self._agents:
            result.notes.append(
                "No agents are registered yet. The pipeline is returning scaffold output only."
            )

        for agent in self._agents:
            payload = agent.execute(ticker, context)
            context["agent_outputs"][agent.output_key] = payload
            result.apply_agent_output(agent.output_key, payload)

        result.ensure_defaults()
        return result