from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseAgent(ABC):
    """Base contract for all stock-analysis agents.

    Adding a new agent:
    1. Subclass BaseAgent.
    2. Implement run() and return a dictionary payload.
    3. Register the agent with PipelineManager.
    4. Use an output_key such as news, prices, prediction, or outlook so the
       dashboard can map the result into the shared OutputSchema.
    """

    def __init__(
        self,
        name: str,
        output_key: str,
        description: str = "",
        llm_client: Any | None = None,
    ) -> None:
        self.name = name
        self.output_key = output_key
        self.description = description
        self.llm_client = llm_client

    @abstractmethod
    def run(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        """Run the agent and return a JSON-serializable payload."""

    def execute(self, ticker: str, context: Mapping[str, Any]) -> dict[str, Any]:
        payload = self.run(ticker, context)
        return self.validate_output(payload)

    def validate_output(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError(
                f"Agent '{self.name}' returned {type(payload).__name__}; expected dict."
            )
        return payload