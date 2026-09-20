"""GitHub tool - GitHub API interactions."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from ..tools.base import Tool, ToolResult


class GitHubTool(Tool):
    """Interact with GitHub API."""

    def __init__(self, token: str | None = None):
        self._token = token

    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return "Interact with GitHub: search repos, get issues, read files, etc."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search_repos", "get_repo", "get_issues", "get_file", "create_issue"],
                    "description": "The action to perform"
                },
                "repository": {
                    "type": "string",
                    "description": "Repository in owner/repo format"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "path": {
                    "type": "string",
                    "description": "File path (for get_file action)"
                },
                "issue_title": {
                    "type": "string",
                    "description": "Issue title (for create_issue action)"
                },
                "issue_body": {
                    "type": "string",
                    "description": "Issue body (for create_issue action)"
                }
            },
            "required": ["action"]
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute GitHub action."""
        action = arguments.get("action", "")

        try:
            if action == "search_repos":
                return await self._search_repos(arguments.get("query", ""))
            elif action == "get_repo":
                return await self._get_repo(arguments.get("repository", ""))
            elif action == "get_issues":
                return await self._get_issues(arguments.get("repository", ""))
            elif action == "get_file":
                return await self._get_file(
                    arguments.get("repository", ""),
                    arguments.get("path", "")
                )
            elif action == "create_issue":
                return await self._create_issue(arguments)
            else:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Unknown action: {action}",
                    success=False
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output=f"GitHub error: {str(e)}",
                success=False,
                error=str(e)
            )

    async def _search_repos(self, query: str) -> ToolResult:
        """Search GitHub repositories."""
        import httpx
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 10}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)
            data = response.json()

        repos = data.get("items", [])
        output_lines = []
        for repo in repos:
            name = repo.get("full_name", "")
            desc = repo.get("description", "")
            stars = repo.get("stargazers_count", 0)
            url = repo.get("html_url", "")
            output_lines.append(f"- {name} ⭐{stars}\n  {desc}\n  {url}")

        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output="\n".join(output_lines) if output_lines else "No repositories found"
        )

    async def _get_repo(self, repo: str) -> ToolResult:
        """Get repository information."""
        import httpx
        url = f"https://api.github.com/repos/{repo}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            if response.status_code == 404:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Repository not found: {repo}",
                    success=False
                )
            data = response.json()

        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=f"Name: {data.get('full_name')}\nDescription: {data.get('description')}\nStars: {data.get('stargazers_count')}\nURL: {data.get('html_url')}"
        )

    async def _get_issues(self, repo: str) -> ToolResult:
        """Get issues from a repository."""
        import httpx
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {"state": "open", "per_page": 10}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)
            issues = response.json()

        if not isinstance(issues, list):
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Failed to fetch issues",
                success=False
            )

        output_lines = []
        for issue in issues:
            title = issue.get("title", "")
            number = issue.get("number", "")
            url = issue.get("html_url", "")
            output_lines.append(f"#{number}: {title}\n  {url}")

        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output="\n".join(output_lines) if output_lines else "No open issues"
        )

    async def _get_file(self, repo: str, path: str) -> ToolResult:
        """Get file content from GitHub."""
        import httpx
        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            if response.status_code == 404:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"File not found: {path}",
                    success=False
                )
            data = response.json()

        content = data.get("content", "")
        if content:
            import base64
            content = base64.b64decode(content).decode("utf-8")
            if len(content) > 10000:
                content = content[:10000] + "\n... [truncated]"

        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            output=content or "No content"
        )

    async def _create_issue(self, arguments: dict[str, Any]) -> ToolResult:
        """Create a GitHub issue."""
        repo = arguments.get("repository", "")
        title = arguments.get("issue_title", "")
        body = arguments.get("issue_body", "")

        if not self._token:
            return ToolResult(
                tool_name=self.name,
                tool_call_id="",
                output="Error: GitHub token not configured. Set GITHUB_TOKEN environment variable.",
                success=False,
                error="Missing token"
            )

        import httpx
        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {
            "Authorization": f"token {self._token}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {"title": title, "body": body}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=10)
            if response.status_code in [200, 201]:
                issue_url = response.json().get("html_url", "")
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Issue created: {issue_url}"
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    tool_call_id="",
                    output=f"Failed to create issue: {response.text}",
                    success=False
                )
