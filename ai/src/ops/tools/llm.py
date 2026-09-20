"""LLM tool - Call other LLM models."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class LLMTool(Tool):
    """Call external LLM APIs for additional reasoning."""

    def __init__(self):
        self._models = {
            "gpt-4o": {"provider": "openai", "name": "GPT-4o"},
            "gpt-4-turbo": {"provider": "openai", "name": "GPT-4 Turbo"},
            "claude-3-opus": {"provider": "anthropic", "name": "Claude 3 Opus"},
            "claude-3-sonnet": {"provider": "anthropic", "name": "Claude 3 Sonnet"},
            "claude-3-haiku": {"provider": "anthropic", "name": "Claude 3 Haiku"},
            "gemini-pro": {"provider": "google", "name": "Gemini Pro"},
        }

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return "Call other LLM models for additional reasoning or tasks."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": list(self._models.keys()),
                    "description": "Model to use"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt to send to the model"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": 1024
                }
            },
            "required": ["model", "prompt"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute LLM call."""
        model = arguments.get("model", "")
        prompt = arguments.get("prompt", "")
        max_tokens = arguments.get("max_tokens", 1024)

        if model not in self._models:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Unknown model: {model}",
                success=False,
                error=f"Model {model} not supported"
            )

        model_info = self._models[model]
        provider = model_info["provider"]

        try:
            if provider == "openai":
                return await self._call_openai(model, prompt, max_tokens)
            elif provider == "anthropic":
                return await self._call_anthropic(model, prompt, max_tokens)
            elif provider == "google":
                return await self._call_google(model, prompt, max_tokens)
            else:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Unsupported provider: {provider}",
                    success=False
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"LLM call error: {str(e)}",
                success=False,
                error=str(e)
            )

    async def _call_openai(self, model: str, prompt: str, max_tokens: int) -> ToolResult:
        """Call OpenAI API."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: OPENAI_API_KEY not set",
                success=False,
                error="Missing API key"
            )

        import httpx
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=60)
            result = response.json()

        if "choices" in result and result["choices"]:
            content = result["choices"][0].get("message", {}).get("content", "")
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=content
            )
        else:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"OpenAI error: {result.get('error', 'Unknown error')}",
                success=False
            )

    async def _call_anthropic(self, model: str, prompt: str, max_tokens: int) -> ToolResult:
        """Call Anthropic API."""
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: ANTHROPIC_API_KEY not set",
                success=False,
                error="Missing API key"
            )

        import httpx
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=60)
            result = response.json()

        if "content" in result:
            content_parts = result["content"]
            content = " ".join(
                part.get("text", "") for part in content_parts
                if part.get("type") == "text"
            )
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=content
            )
        else:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Anthropic error: {result.get('error', 'Unknown error')}",
                success=False
            )

    async def _call_google(self, model: str, prompt: str, max_tokens: int) -> ToolResult:
        """Call Google Gemini API."""
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: GEMINI_API_KEY not set",
                success=False,
                error="Missing API key"
            )

        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens}
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, params=params, json=data, timeout=60)
            result = response.json()

        if "candidates" in result and result["candidates"]:
            content = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=content
            )
        else:
            error = result.get("error", {}).get("message", "Unknown error")
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Google error: {error}",
                success=False
            )
