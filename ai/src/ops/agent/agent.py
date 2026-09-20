"""OPS Agent - Core Architecture

This module implements the OPS agent architecture as shown in the design diagram.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from loguru import logger

from ops.models.base import BaseModelWrapper
from ops.tools.registry import ToolRegistry, ToolResult


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    DONE = "done"


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentConfig:
    """Configuration for the OPS agent."""
    model: BaseModelWrapper | None = None
    tools: list[str] = field(default_factory=lambda: ["all"])
    max_iterations: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    system_prompt: str | None = None
    verbose: bool = False
    output_dir: Path = field(default_factory=lambda: Path.home() / ".ops" / "output")


class OPSAgent:
    """
    OPS Agent - Core agent that orchestrates LLM calls, tool usage, and conversation.

    Designed to rival Claude Opus with custom-trained Meta LLaMA models.

    Architecture:
    - Agent Core: Manages conversation state and agent loop
    - Tool System: Registry and execution of various tools
    - Model Integration: LLaMA-based models with tool calling
    - CLI Interface: Rich terminal interface
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.state = AgentState.IDLE
        self.conversation: list[Message] = []
        self.tool_registry = ToolRegistry()
        self.iteration_count = 0
        self._stop_event = asyncio.Event()

        # Register default tools
        self._register_default_tools()

        logger.info(f"OPS Agent initialized (version {self._get_version()})")

    def _get_version(self) -> str:
        """Get OPS version."""
        try:
            from ops import __version__
            return __version__
        except ImportError:
            return "dev"

    def _register_default_tools(self) -> None:
        """Register all default tools."""
        from ops.tools.shell import ShellTool
        from ops.tools.file import FileTool
        from ops.tools.browser import BrowserTool
        from ops.tools.code import CodeTool
        from ops.tools.search import SearchTool
        from ops.tools.github import GitHubTool
        from ops.tools.llm import LLMTool

        tools = [
            ShellTool(),
            FileTool(),
            BrowserTool(),
            CodeTool(),
            SearchTool(),
            GitHubTool(),
            LLMTool(),
        ]

        for tool in tools:
            self.tool_registry.register(tool)
            logger.debug(f"Registered tool: {tool.name}")

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.conversation.append(message)
        if self.config.verbose:
            logger.debug(f"Added {message.role} message: {message.content[:100]}...")

    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        if self.config.system_prompt:
            return self.config.system_prompt

        return """You are OPS, an advanced AI assistant built on Meta's LLaMA foundation.
You have access to various tools to help users complete tasks.
Be helpful, accurate, and concise in your responses.
When you need to use a tool, respond with the appropriate tool call format.
Think step by step before taking actions."""

    def _build_messages(self) -> list[dict]:
        """Build messages in the format expected by the LLM."""
        messages = [{"role": "system", "content": self.get_system_prompt()}]

        for msg in self.conversation:
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            messages.append(message_dict)

        return messages

    async def run(self, user_input: str) -> AsyncIterator[str]:
        """
        Run the agent loop for a single user input.

        Yields text chunks as they are generated.
        """
        self.state = AgentState.THINKING
        self.iteration_count = 0
        self._stop_event.clear()

        # Add user message
        self.add_message(Message(role="user", content=user_input))

        try:
            while self.iteration_count < self.config.max_iterations:
                if self._stop_event.is_set():
                    logger.info("Agent stopped by user")
                    break

                # Get LLM response
                response = await self._get_llm_response()

                if response.get("finish_reason") == "stop":
                    # Pure text response
                    yield response["content"]
                    self.add_message(Message(
                        role="assistant",
                        content=response["content"]
                    ))
                    break

                elif response.get("finish_reason") == "tool_calls":
                    # Tool call response
                    self.state = AgentState.EXECUTING
                    tool_calls = response.get("tool_calls", [])

                    # Add assistant message with tool calls
                    self.add_message(Message(
                        role="assistant",
                        content="",
                        tool_calls=tool_calls
                    ))

                    # Execute tools
                    tool_results = await self._execute_tool_calls(tool_calls)

                    # Add tool results
                    for result in tool_results:
                        self.add_message(Message(
                            role="tool",
                            content=result.output,
                            tool_call_id=result.tool_call_id
                        ))

                    self.iteration_count += 1
                    self.state = AgentState.THINKING
                else:
                    logger.warning(f"Unknown finish reason: {response.get('finish_reason')}")
                    break

            self.state = AgentState.DONE

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent error: {e}")
            yield f"Error: {e}"

    async def _get_llm_response(self) -> dict:
        """Get response from the LLM model."""
        if not self.config.model:
            raise RuntimeError("No model configured. Please set config.model")

        messages = self._build_messages()

        logger.debug(f"Requesting LLM response ({len(messages)} messages)")

        response = await self.config.model.chat(
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            tools=self.tool_registry.get_tool_definitions(),
            stream=False,
        )

        return response

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolResult]:
        """Execute tool calls and return results."""
        results = []

        for call in tool_calls:
            tool_name = call.get("function", {}).get("name", "")
            tool_args = call.get("function", {}).get("arguments", {})
            call_id = call.get("id", "")

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            tool_result = await self.tool_registry.execute(
                tool_name=tool_name,
                arguments=tool_args,
                tool_call_id=call_id
            )
            results.append(tool_result)

        return results

    def stop(self) -> None:
        """Stop the agent loop."""
        self._stop_event.set()
        self.state = AgentState.IDLE
        logger.info("Agent stop signal sent")

    def reset(self) -> None:
        """Reset the agent state."""
        self.conversation.clear()
        self.iteration_count = 0
        self.state = AgentState.IDLE
        logger.info("Agent reset")

    def get_status(self) -> dict:
        """Get agent status information."""
        return {
            "state": self.state.value,
            "iteration": self.iteration_count,
            "messages": len(self.conversation),
            "tools": len(self.tool_registry.tools),
            "version": self._get_version(),
        }
