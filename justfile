default:
    just --list

sync:
    uv sync --locked

format:
    uv run ruff check --fix
    uv run ruff format

lint:
    uv run ruff check
    uv run ruff format --check
    uv run mypy

test *ARGS:
    uv run pytest {{ARGS}}

all: format lint test
