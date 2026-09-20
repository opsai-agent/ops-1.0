"""
OPS CLI - Command line interface for OPS Agent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from ops.agent.agent import OPSAgent, AgentConfig
from ops.models.llama import LLaMAModel


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="ops")
def main():
    """OPS - Open Source AI Agent"""
    pass


@main.command()
@click.argument("prompt", nargs=-1, required=False)
@click.option("--model", "-m", default="llama-3-8b", help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--tools", "-t", default="all", help="Comma-separated list of tools")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def chat(prompt, model, verbose, tools, interactive):
    """Start a chat with OPS Agent."""
    config = AgentConfig(
        model=LLaMAModel(model_name=model),
        verbose=verbose,
        tools=tools.split(",") if tools != "all" else ["all"],
    )

    agent = OPSAgent(config=config)

    if interactive:
        _run_interactive(agent)
    else:
        prompt_text = " ".join(prompt) if prompt else ""
        if not prompt_text:
            console.print("[red]Error: Please provide a prompt or use --interactive[/red]")
            sys.exit(1)
        asyncio.run(_run_chat(agent, prompt_text))


def _run_interactive(agent: OPSAgent) -> None:
    """Run interactive chat mode."""
    console.print(Panel(
        "[bold cyan]OPS Agent[/bold cyan]\n"
        "[dim]Type 'quit' or 'exit' to leave[/dim]",
        title="Welcome"
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ")
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            console.print("[bold blue]OPS:[/bold blue] ", end="")
            response = asyncio.run(agent.run(user_input))
            # Collect all output
            output_parts = []
            for chunk in response:
                output_parts.append(chunk)
                console.print(chunk, end="")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def _run_chat(agent: OPSAgent, prompt: str) -> None:
    """Run single chat and output result."""
    full_response = ""
    async for chunk in agent.run(prompt):
        full_response += chunk
        console.print(chunk, end="")
    console.print()


@main.command()
@click.option("--port", "-p", default=8080, help="Port to run on")
def desktop(port):
    """Start the OPS Desktop application."""
    console.print(f"[bold]Starting OPS Desktop on port {port}...[/bold]")
    console.print("[dim]This feature is under development.[/dim]")


@main.command()
def status():
    """Show OPS Agent status."""
    console.print("[bold]OPS Agent Status[/bold]")
    console.print("- Model: LLaMA-based (custom trained)")
    console.print("- Status: Ready")
    console.print(f"- Working directory: {Path.cwd()}")


@main.command()
def init():
    """Initialize OPS configuration."""
    config_dir = Path.home() / ".ops"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        default_config = """
# OPS Configuration
model:
  name: llama-3-8b
  context_length: 8192

agent:
  max_iterations: 50
  temperature: 0.7

tools:
  enabled: ["shell", "file", "search", "code", "browser", "github", "llm"]
"""
        config_file.write_text(default_config.strip())
        console.print(f"[green]✓[/green] Configuration created at {config_file}")

    console.print("[green]OPS initialized successfully![/green]")


if __name__ == "__main__":
    main()
