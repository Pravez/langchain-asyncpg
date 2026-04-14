lint-and-fix:
    uvx ruff format .
    uvx ruff check . --fix

test path="tests/":
    uv run pytest {{path}}

tag bump="patch":
    git tag "v$(uv version --short)"
    git push origin --tags
    uv version --bump {{bump}}