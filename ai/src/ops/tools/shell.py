"""Shell tool - Execute shell commands."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class ShellTool(Tool):
    """Execute shell commands in a sandboxed environment."""

    def __init__(self, workdir: Path | None = None, timeout: int = 60):
        self._workdir = workdir or Path.cwd()
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return the output. Supports bash commands."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional)"
                }
            },
            "required": ["command"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a shell command."""
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", self._timeout)
        cwd = arguments.get("cwd", str(self._workdir))

        if not command.strip():
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: Empty command",
                success=False,
                error="Command is empty"
            )

        logger.info(f"Executing shell command: {command[:200]}...")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                timeout=timeout,
            )

            stdout, stderr = await asyncio.gather(
                process.stdout.read(),
                process.stderr.read()
            )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            output = stdout_str
            if stderr_str:
                output += f"\n[stderr]\n{stderr_str}"

            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=output,
                success=process.returncode == 0,
                metadata={
                    "returncode": process.returncode,
                    "timeout": timeout,
                }
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Command timed out after {timeout} seconds",
                success=False,
                error="Timeout"
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Error: {str(e)}",
                success=False,
                error=str(e)
            )
