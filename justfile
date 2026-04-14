lint-and-fix:
    uvx ruff format .
    uvx ruff check . --fix

test path="tests/":
    uv run pytest {{path}}