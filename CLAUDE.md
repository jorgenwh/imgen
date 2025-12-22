# imgen - Image Generation Neural Network Experiments

## Project Overview
Educational/experimental repository exploring different neural network architectures for image generation. Focus is on understanding and implementing various methods at small scale rather than production-quality results.

## Tech Stack
- Python 3.12+
- uv for package management
- PyTorch for neural networks

## Commands
- `uv sync` - Install dependencies
- `uv run python main.py` - Run main script
- `uv run pytest` - Run tests (when added)

## Project Structure
```
imgen/
├── main.py          # Entry point
├── pyproject.toml   # Project config and dependencies
└── CLAUDE.md        # This file
```

## Code Style
- Follow PEP 8
- Type hints encouraged
- Keep implementations simple and readable - this is for learning

## Goals
- Implement different image generation architectures
- Keep networks small and trainable on modest hardware
- Prioritize understanding over state-of-the-art results
