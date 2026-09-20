"""Base classes for tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    tool_call_id: str
    output: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Base protocol for all tools."""

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters schema."""
        ...

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool."""
        ...


@dataclass
class ToolDefinition:
    """JSON schema definition for a tool."""
    name: str
    description: str
    parameters: dict[str, Any]


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools."""
        return self._tools

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI format."""
        definitions = []
        for tool in self._tools.values():
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        return definitions

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                output=f"Tool '{tool_name}' not found",
                success=False,
                error=f"Tool {tool_name} not registered"
            )

        try:
            result = await tool.execute(arguments)
            return result
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                output=str(e),
                success=False,
                error=str(e)
            )