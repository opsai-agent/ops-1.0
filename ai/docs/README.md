# OPS Documentation

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Training](#training)
6. [API Reference](#api-reference)
7. [Contributing](#contributing)
8. [License](#license)

---

## Getting Started

OPS is an open-source AI agent built on Meta's LLaMA foundation. It features:

- Custom-trained models with full transparency
- Comprehensive tool ecosystem (shell, browser, code, search, GitHub)
- CLI and Desktop interfaces
- Extensible architecture

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ops-ai/ops.git
cd ops

# Install dependencies
pip install -e ".[all]"

# Initialize configuration
ops init

# Start using OPS
ops chat "Tell me about Python asyncio"
```

---

## Installation

### From Source

```bash
git clone https://github.com/ops-ai/ops.git
cd ops
pip install -e ".[all]"
```

### Required Dependencies

- Python 3.10+
- PyTorch 2.1+
- transformers 4.40+
- GPU (optional, for faster inference)

### Optional Dependencies

```bash
# Desktop app
pip install -e ".[desktop]"

# Training
pip install -e ".[training]"

# Development
pip install -e ".[dev]"
```

---

## Usage

### CLI Interface

```bash
# Interactive chat
ops

# Single query
ops chat "Explain quantum computing"

# With specific model
ops chat -m llama-3-70b "Your question"

# Verbose mode
ops chat -v "Debug my code"

# With specific tools
ops chat -t shell,file "Run ls -la"
```

### Python API

```python
from ops import OPSAgent, AgentConfig
from ops.models.llama import LLaMAModel

# Configure agent
config = AgentConfig(
    model=LLaMAModel(model_name="llama-3-8b"),
    verbose=True,
)

# Create agent
agent = OPSAgent(config=config)

# Run chat
async def main():
    async for chunk in agent.run("Hello, OPS!"):
        print(chunk, end="")

import asyncio
asyncio.run(main())
```

### Desktop Application

```bash
# Start desktop app (when available)
ops desktop

# Or build from source
python scripts/build.py
```

---

## Configuration

### Agent Configuration

Edit `~/.ops/config.yaml`:

```yaml
model:
  name: llama-3-8b
  context_length: 8192

agent:
  max_iterations: 50
  temperature: 0.7

tools:
  enabled:
    - shell
    - file
    - search
    - code
    - browser
    - github
    - llm
```

### Environment Variables

```bash
# Model API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export GITHUB_TOKEN="ghp_..."
```

---

## Training

### Prepare Dataset

```bash
python -m ops.training.datasets --dataset open_orca --format alpaca
```

### Run Training

```bash
python -m ops.training.pipeline \
    --base-model meta-llama/Meta-Llama-3-8B \
    --dataset open-orca/platypus2 \
    --epochs 3 \
    --output-dir ./output/ops-llama-8b
```

### Training Stages

1. **SFT General** - Supervised Fine-Tuning on general instructions
2. **SFT Code** - Code-focused fine-tuning
3. **SFT Tools** - Tool-use fine-tuning
4. **RLHF** - Reinforcement Learning from Human Feedback

See [configs/training/train_config.yaml](../configs/training/train_config.yaml) for full configuration.

---

## API Reference

### OPSAgent

```python
class OPSAgent:
    def __init__(self, config: AgentConfig)
    async def run(self, user_input: str) -> AsyncIterator[str]
    def stop(self) -> None
    def reset(self) -> None
    def get_status(self) -> dict
```

### AgentConfig

```python
@dataclass
class AgentConfig:
    model: BaseModelWrapper | None = None
    tools: list[str] = field(default_factory=lambda: ["all"])
    max_iterations: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    system_prompt: str | None = None
    verbose: bool = False
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None
    def unregister(self, tool_name: str) -> bool
    def get(self, tool_name: str) -> Tool | None
    def get_tool_definitions(self) -> list[dict]
    async def execute(self, tool_name: str, arguments: dict, tool_call_id: str) -> ToolResult
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/ops-ai/ops.git
cd ops
pip install -e ".[dev]"
pre-commit install
pytest  # Run tests
```

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

## Links

- [GitHub Repository](https://github.com/ops-ai/ops)
- [Issue Tracker](https://github.com/ops-ai/ops/issues)
- [Discussions](https://github.com/ops-ai/ops/discussions)
