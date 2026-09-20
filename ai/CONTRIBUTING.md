# Contributing to OPS

Thank you for your interest in contributing to OPS! This document provides guidelines for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ops-ai/ops.git
cd ops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Run linter
ruff check .

# Run formatter
ruff format .

# Run type checker
mypy src/ops
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ops --cov-report=html

# Run specific test file
pytest tests/test_ops.py -v
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new tool X
fix: resolve issue with Y
docs: update README
test: add tests for Z
chore: update dependencies
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes
6. Push to the branch
7. Open a Pull Request

## Reporting Issues

- Use the [GitHub issue tracker](https://github.com/ops-ai/ops/issues)
- Include steps to reproduce
- Provide system information (OS, Python version, etc.)
- Attach relevant logs or screenshots

## Getting Help

- Join our [Discussions](https://github.com/ops-ai/ops/discussions)
- Check the [documentation](../docs/README.md)
- Search existing issues

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create an inclusive community.

---

Thank you for helping make OPS better! 🚀
