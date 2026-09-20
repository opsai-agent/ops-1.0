"""Search tool - Web search capabilities."""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class SearchTool(Tool):
    """Web search using DuckDuckGo."""

    def __init__(self):
        self._search = None

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Perform web search."""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        if not query.strip():
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: Empty search query",
                success=False,
                error="Query is empty"
            )

        try:
            from duckduckgo_search import DDGS

            logger.info(f"Searching: {query}")

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output="No results found",
                    success=True
                )

            output_lines = []
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("href", "")
                snippet = result.get("body", "")
                output_lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="\n\n".join(output_lines)
            )

        except ImportError:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: duckduckgo-search not installed. Run: pip install duckduckgo-search",
                success=False,
                error="Missing dependency"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Search error: {str(e)}",
                success=False,
                error=str(e)
            )
