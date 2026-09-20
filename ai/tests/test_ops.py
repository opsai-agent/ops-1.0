"""Tests for OPS Agent."""

import pytest
import asyncio
from pathlib import Path

from ops.agent.agent import OPSAgent, AgentConfig, Message, AgentState
from ops.models.llama import LLaMAModel
from ops.tools.shell import ShellTool
from ops.tools.file import FileTool
from ops.tools.search import SearchTool


class TestOPSAgent:
    """Test the OPS Agent."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        config = AgentConfig(verbose=True)
        return OPSAgent(config=config)

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state == AgentState.IDLE
        assert len(agent.conversation) == 0
        assert agent.iteration_count == 0
        assert len(agent.tool_registry.tools) > 0

    def test_add_message(self, agent):
        """Test adding messages."""
        msg = Message(role="user", content="Hello")
        agent.add_message(msg)
        assert len(agent.conversation) == 1
        assert agent.conversation[0].role == "user"

    def test_get_system_prompt(self, agent):
        """Test system prompt generation."""
        prompt = agent.get_system_prompt()
        assert "OPS" in prompt

    def test_reset(self, agent):
        """Test agent reset."""
        agent.add_message(Message(role="user", content="Test"))
        agent.iteration_count = 5
        agent.reset()
        assert len(agent.conversation) == 0
        assert agent.iteration_count == 0
        assert agent.state == AgentState.IDLE

    def test_get_status(self, agent):
        """Test status reporting."""
        status = agent.get_status()
        assert "state" in status
        assert "tools" in status
        assert status["state"] == "idle"


class TestShellTool:
    """Test shell tool."""

    def test_execute_command(self):
        """Test executing a simple command."""
        tool = ShellTool()
        result = asyncio.run(tool.execute({"command": "echo 'hello'"}))
        assert result.success
        assert "hello" in result.output

    def test_empty_command(self):
        """Test empty command handling."""
        tool = ShellTool()
        result = asyncio.run(tool.execute({"command": ""}))
        assert not result.success
        assert "empty" in result.output.lower()


class TestFileTool:
    """Test file tool."""

    def test_read_write(self, tmp_path):
        """Test file read/write operations."""
        tool = FileTool()
        test_file = tmp_path / "test.txt"

        # Write
        result = asyncio.run(tool.execute({
            "operation": "write",
            "path": str(test_file),
            "content": "Hello, OPS!"
        }))
        assert result.success

        # Read
        result = asyncio.run(tool.execute({
            "operation": "read",
            "path": str(test_file)
        }))
        assert result.success
        assert "Hello, OPS!" in result.output

    def test_list_directory(self, tmp_path):
        """Test directory listing."""
        tool = FileTool()
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.txt").write_text("content")

        result = asyncio.run(tool.execute({
            "operation": "list",
            "path": str(tmp_path)
        }))
        assert result.success
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output


class TestLLaMAModel:
    """Test LLaMA model wrapper."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = LLaMAModel(model_name="llama-3-8b")
        assert model.model_name == "llama-3-8b"
        assert model.context_length == 8192

    def test_model_info(self):
        """Test model info retrieval."""
        model = LLaMAModel()
        info = model.get_model_info()
        assert "name" in info
        assert "context_length" in info


class TestToolRegistry:
    """Test tool registry."""

    def test_register_and_execute(self):
        """Test tool registration and execution."""
        from ops.tools.registry import ToolRegistry
        from ops.tools.shell import ShellTool

        registry = ToolRegistry()
        tool = ShellTool()

        registry.register(tool)
        assert tool.name in registry.tools

        # Test tool definitions
        definitions = registry.get_tool_definitions()
        assert len(definitions) > 0
        assert definitions[0]["type"] == "function"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
