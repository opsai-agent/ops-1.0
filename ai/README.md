# OPS - Open-source AI Agent

**OPS** is a fully open-source AI agent built from the ground up, using Meta's LLaMA as the foundation model. Designed to rival Claude Opus in capability while being completely transparent and extensible.

## Features

- 🤖 **Custom-trained Model** - Fine-tuned on Meta LLaMA with our curated training pipeline
- 🛠️ **Tool Ecosystem** - Shell, browser, file system, code execution, and more
- 💻 **Desktop App** - Native application for macOS, Windows, and Linux
- 🖥️ **CLI Interface** - Powerful terminal-based agent interface
- 🔧 **Extensible** - Easy-to-add custom tools and capabilities
- 📊 **Training Pipeline** - Built-in support for continued pretraining and fine-tuning

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Initialize
ops init

# Start CLI
ops

# Start Desktop App
ops desktop
```

## Architecture

```
ops/
├── src/ops/           # Core agent library
│   ├── agent/         # Agent core & tool system
│   ├── models/        # LLM models (LLaMA-based)
│   ├── training/      # Training pipeline
│   └── cli/           # CLI interface
├── apps/desktop/      # Desktop application
├── scripts/           # Utility scripts
└── tests/             # Test suite
```

## Model Training

OPS uses a multi-stage training approach:

1. **Base Model**: Meta LLaMA-3 (8B/70B)
2. **SFT**: Supervised Fine-Tuning on curated datasets
3. **RLHF**: Reinforcement Learning from Human Feedback
4. **Tool Use**: Specialized training for tool calling

See [docs/training.md](docs/training.md) for detailed instructions.

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
