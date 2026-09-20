"""Code execution tool."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class CodeTool(Tool):
    """Execute code in various languages."""

    SUPPORTED_LANGUAGES = ["python", "javascript", "bash", "typescript"]

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Execute code snippets in Python, JavaScript, Bash, and TypeScript."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": self.SUPPORTED_LANGUAGES,
                    "description": "Programming language"
                },
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                }
            },
            "required": ["language", "code"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute code."""
        language = arguments.get("language", "python")
        code = arguments.get("code", "")

        if language not in self.SUPPORTED_LANGUAGES:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Unsupported language: {language}",
                success=False,
                error=f"Language {language} not supported"
            )

        try:
            if language == "python":
                return await self._run_python(code)
            elif language == "javascript":
                return await self._run_javascript(code)
            elif language == "bash":
                return await self._run_bash(code)
            elif language == "typescript":
                return await self._run_typescript(code)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Execution error: {str(e)}",
                success=False,
                error=str(e)
            )

    async def _run_python(self, code: str) -> ToolResult:
        """Run Python code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=self._timeout,
            )
            stdout, stderr = await asyncio.gather(
                process.stdout.read(),
                process.stderr.read()
            )
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n[stderr]\n{stderr.decode('utf-8', errors='replace')}"
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                success=process.returncode == 0,
                metadata={"returncode": process.returncode}
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _run_javascript(self, code: str) -> ToolResult:
        """Run JavaScript code with Node.js."""
        try:
            process = await asyncio.create_subprocess_exec(
                "node", "-e", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=self._timeout,
            )
            stdout, stderr = await asyncio.gather(
                process.stdout.read(),
                process.stderr.read()
            )
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n[stderr]\n{stderr.decode('utf-8', errors='replace')}"
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                success=process.returncode == 0
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: Node.js not found",
                success=False,
                error="Node.js not installed"
            )

    async def _run_bash(self, code: str) -> ToolResult:
        """Run bash script."""
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output="Use the 'shell' tool for bash commands instead.",
            success=False,
            error="Use shell tool for bash"
        )

    async def _run_typescript(self, code: str) -> ToolResult:
        """Run TypeScript code."""
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output="TypeScript execution requires compilation. Use JavaScript instead or compile first.",
            success=False,
            error="TypeScript needs compilation"
        )
