"""File tool - File system operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class FileTool(Tool):
    """File system operations - read, write, list, search."""

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Perform file system operations: read, write, list directory, search files."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "search", "exists", "mkdir", "delete"],
                    "description": "The operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (for search operation)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively (default: false)"
                }
            },
            "required": ["operation", "path"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute file operation."""
        operation = arguments.get("operation", "")
        path = arguments.get("path", "")

        try:
            if operation == "read":
                return await self._read(path)
            elif operation == "write":
                return await self._write(path, arguments.get("content", ""))
            elif operation == "list":
                return await self._list(path)
            elif operation == "search":
                return await self._search(path, arguments.get("pattern", ""),
                                         arguments.get("recursive", False))
            elif operation == "exists":
                return await self._exists(path)
            elif operation == "mkdir":
                return await self._mkdir(path)
            elif operation == "delete":
                return await self._delete(path)
            else:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Unknown operation: {operation}",
                    success=False,
                    error=f"Invalid operation: {operation}"
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Error: {str(e)}",
                success=False,
                error=str(e)
            )

    async def _read(self, path: str) -> ToolResult:
        """Read file contents."""
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"File not found: {path}",
                success=False,
                error="File not found"
            )
        content = file_path.read_text(encoding="utf-8")
        # Truncate very large files
        if len(content) > 50000:
            content = content[:50000] + "\n... [truncated, file too large]"
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=content,
            metadata={"size": len(content)}
        )

    async def _write(self, path: str, content: str) -> ToolResult:
        """Write content to file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=f"Written {len(content)} bytes to {path}",
            metadata={"size": len(content)}
        )

    async def _list(self, path: str) -> ToolResult:
        """List directory contents."""
        dir_path = Path(path)
        if not dir_path.exists():
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"Directory not found: {path}",
                success=False,
                error="Directory not found"
            )
        items = []
        for item in dir_path.iterdir():
            suffix = "/" if item.is_dir() else ""
            items.append(f"{item.name}{suffix}")
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output="\n".join(sorted(items))
        )

    async def _search(self, path: str, pattern: str, recursive: bool) -> ToolResult:
        """Search for files matching pattern."""
        dir_path = Path(path)
        if recursive:
            pattern_str = f"**/{pattern}" if "*" in pattern else f"**/*{pattern}*"
        else:
            pattern_str = f"{pattern}" if "*" in pattern else f"*{pattern}*"
        matches = list(dir_path.glob(pattern_str))
        output = "\n".join(str(m) for m in matches[:100])
        if len(matches) > 100:
            output += "\n... [truncated, too many results]"
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=output,
            metadata={"matches": len(matches)}
        )

    async def _exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        exists = Path(path).exists()
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=f"exists: {exists}"
        )

    async def _mkdir(self, path: str) -> ToolResult:
        """Create directory."""
        Path(path).mkdir(parents=True, exist_ok=True)
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=f"Created directory: {path}"
        )

    async def _delete(self, path: str) -> ToolResult:
        """Delete file or directory."""
        file_path = Path(path)
        if file_path.is_dir():
            import shutil
            shutil.rmtree(file_path)
        else:
            file_path.unlink()
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=f"Deleted: {path}"
        )
