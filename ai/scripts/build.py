# OPS - Build Script
"""
Build script for OPS desktop application.
"""

import os
import sys
import shutil
from pathlib import Path


def build():
    """Build the OPS desktop application."""
    print("=" * 50)
    print("OPS Desktop Build")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ required")
        sys.exit(1)

    # Install dependencies
    print("\n[1/4] Installing dependencies...")
    os.system("pip install -e '.[desktop]'")

    # Build with PyInstaller
    print("\n[2/4] Building with PyInstaller...")
    build_cmd = """
    pyinstaller --onefile \\
        --windowed \\
        --name ops \\
        --add-data "configs;configs" \\
        --add-data "src/ops;ops" \\
        src/ops/cli/main.py
    """
    os.system(build_cmd)

    # Copy to output directory
    print("\n[3/4] Organizing output...")
    dist_dir = Path("apps/desktop/dist")
    dist_dir.mkdir(parents=True, exist_ok=True)

    exe_name = "ops.exe" if sys.platform == "win32" else "ops"
    source = Path("dist") / exe_name
    if source.exists():
        shutil.copy2(source, dist_dir / exe_name)
        print(f"  Copied to {dist_dir / exe_name}")

    # Create README
    print("\n[4/4] Creating distribution README...")
    readme = dist_dir / "README.md"
    readme.write_text("""# OPS Desktop Application

## Installation

1. Download the latest release from GitHub
2. Extract the archive
3. Run `ops` (or `ops.exe` on Windows)

## Requirements

- Python 3.10+
- GPU (optional, for local model inference)

## Usage

```bash
# Start interactive chat
./ops

# Run single query
./ops chat "Your question here"

# Start web server
./ops server
```

## Build from Source

```bash
pip install -e ".[desktop]"
python scripts/build.py
```
""")

    print("\n" + "=" * 50)
    print("Build complete!")
    print(f"Output: {dist_dir.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    build()
