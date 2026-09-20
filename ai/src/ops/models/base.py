"""Base model wrapper for OPS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """Abstract base class for LLM model wrappers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Returns a dict with keys:
        - content: str | None
        - tool_calls: list[dict] | None
        - finish_reason: str
        """
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate text completion."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model."""
        ...

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context length."""
        ...
