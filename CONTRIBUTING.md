# Contribute

Do you like this project?
If so, you can contribute to it in various ways,
and you don't need to be a developer!

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/HugoFara/pylinkage.git
cd pylinkage
uv sync
```

You will need to have your own fork for this project if you want to submit pull requests.

## Testing

We use pytest. Run tests with:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=pylinkage --cov-report=html
```

## Linting and Type Checking

```bash
uv run ruff check .
uv run mypy pylinkage
```

## Building

```bash
uv build
```

## Release

This section is mainly intended for maintainers.

1. Update the version in `pyproject.toml` and `pylinkage/__init__.py`.
2. Update CHANGELOG.md with release date and edit subsection titles.
3. Regenerate the documentation (uses Sphinx):

   ```bash
   uv run sphinx-build -b html sphinx/ docs/
   ```

4. Commit and add a tag (e.g., v0.4.0).
5. Publish a new [GitHub release](https://github.com/HugoFara/pylinkage/releases).

## Caveats

Pylinkage is a small project to build 2D linkages in a simple way.
It is not intended for complex simulations.
If you want to do much more, my best advice is to start your own project.
If you simply want to have fun developing new features, you are welcome here!

## Contributions for everyone

Don't forget to drop a star, fork it or share it on social media.
This is a community project, and the bigger the community, the more it will thrive!
