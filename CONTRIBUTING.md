# Contributing

Thanks for contributing to Claude STT. This repo is small and fast-moving, so we optimize for clarity and quick review.

## How to Contribute

1) Fork and clone the repo
2) Create a branch
3) Make your changes
4) Run tests and update docs if needed
5) Open a pull request

## Development

```bash
# Clone and setup
git clone https://github.com/jarrodwatts/claude-stt
cd claude-stt

# Install dependencies (uv preferred)
uv sync --python 3.12 --extra dev
```

## Tests

```bash
uv run python -m unittest discover -s tests
```

## Code Style

- Keep changes focused and small
- Prefer tests for behavior changes
- Avoid introducing dependencies unless necessary
- Follow existing patterns in the codebase

## Pull Requests

- Describe the problem and the fix
- Include tests or explain why they are not needed
- Link issues when relevant

## Versioning

We use semantic versioning (`MAJOR.MINOR.PATCH`) in `pyproject.toml`.
