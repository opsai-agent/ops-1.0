"""Browser tool - Web browsing capabilities."""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class BrowserTool(Tool):
    """Web browsing tool using headless browser."""

    def __init__(self):
        self._browser = None

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return "Browse websites, extract content, and interact with web pages."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "extract", "click", "fill", "screenshot"],
                    "description": "The browser action to perform"
                },
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for element interaction"
                },
                "text": {
                    "type": "string",
                    "description": "Text to fill in form fields"
                }
            },
            "required": ["action", "url"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute browser action."""
        action = arguments.get("action", "get")
        url = arguments.get("url", "")

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                try:
                    await page.goto(url, timeout=30000)

                    if action == "get":
                        content = await page.content()
                        return ToolResult(
                            tool_name=self.name,
                            tool_call_id="",
                            output=content[:50000]  # Truncate large pages
                        )
                    elif action == "extract":
                        text = await page.evaluate("() => document.body.innerText")
                        return ToolResult(
                            tool_name=self.name,
                            tool_call_id="",
                            output=text[:10000]
                        )
                    elif action == "screenshot":
                        screenshot_path = f"/tmp/screenshot_{int(__import__('time').time())}.png"
                        await page.screenshot(path=screenshot_path, full_page=True)
                        return ToolResult(
                            tool_name=self.name,
                            tool_call_id="",
                            output=f"Screenshot saved to {screenshot_path}"
                        )
                    else:
                        return ToolResult(
                            tool_name=self.name,
                            tool_call_id="",
                            output=f"Action '{action}' not fully implemented",
                            success=False
                        )
                finally:
                    await browser.close()

        except ImportError:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: playwright not installed. Run: pip install playwright",
                success=False,
                error="Missing dependency"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Browser error: {str(e)}",
                success=False,
                error=str(e)
            )
