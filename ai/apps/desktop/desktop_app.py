# OPS Desktop Application
"""
Desktop application for OPS Agent.
Built with PyInstaller for cross-platform distribution.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()


def create_desktop_app():
    """Create and configure the desktop application."""
    # This is a placeholder for the desktop app implementation
    # In production, this would use a framework like:
    # - PySide6/PyQt6 for native GUI
    # - Electron + Python backend for cross-platform
    # - Tauri for lightweight native app

    console.print("""
[bold]OPS Desktop Application[/bold]

The desktop application is under development.
Available options:

  1. [cyan]CLI Mode[/cyan] - Use the terminal interface
     $ ops chat "Your question here"

  2. [cyan]Web UI[/cyan] - Use the web interface (coming soon)
     $ ops server

  3. [cyan]Desktop App[/cyan] - Native application (coming soon)
     Download from releases page
""")


@click.command()
@click.option("--port", "-p", default=8080, help="Port for web interface")
def server(port):
    """Start the OPS web server."""
    console.print(f"[bold]Starting OPS Web Server on port {port}...[/bold]")
    console.print("[dim]This feature is under development.[/dim]")

    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn

        app = FastAPI(title="OPS Agent API")

        @app.get("/")
        async def root():
            return {"status": "ok", "message": "OPS Agent API is running"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        console.print(f"[green]✓[/green] Server started on http://localhost:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)

    except ImportError:
        console.print("[red]Error: Required packages not installed[/red]")
        console.print("Install with: pip install fastapi uvicorn")
        sys.exit(1)


def build_desktop_package():
    """Build desktop application package."""
    import subprocess

    console.print("[bold]Building OPS Desktop Package...[/bold]")

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", "ops",
        "--add-data", "configs:configs",
        "--add-data", "src/ops:ops",
        "--icon", "assets/icon.ico",
        "src/ops/cli/main.py"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]✓[/green] Desktop package built successfully!")
            console.print(f"Output: apps/desktop/dist/ops")
        else:
            console.print(f"[red]Build failed:[/red] {result.stderr}")
    except FileNotFoundError:
        console.print("[red]Error: PyInstaller not installed[/red]")
        console.print("Install with: pip install pyinstaller")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_desktop_package()
    else:
        create_desktop_app()
