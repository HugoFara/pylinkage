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
uv run task test
```

Run with coverage:

```bash
uv run task test-cov
```

## Linting and Type Checking

```bash
uv run task lint        # Check for issues
uv run task lint-fix    # Auto-fix issues
uv run task format      # Format code
uv run task typecheck   # Run mypy
```

## Building

```bash
uv build
```

## Running the Web App (React + FastAPI)

Install dependencies:

```bash
uv sync --extra api          # Install FastAPI/uvicorn backend dependencies
cd frontend && npm install   # Install React frontend dependencies
```

Run the backend (from project root):

```bash
uv run task api
```

Run the frontend (in a separate terminal):

```bash
uv run task frontend
```

## Documentation

Build the documentation locally:

```bash
uv run task docs
```

Then open `docs/index.html` in your browser.

To clean the build artifacts:

```bash
uv run task docs-clean
```

## Release

This section is mainly intended for maintainers.

1. Update CHANGELOG.md with release date and edit subsection titles.
2. Regenerate the documentation (uses Sphinx):

   ```bash
   uv run sphinx-build -b html docs/source docs/
   ```

3. Bump the version (updates `pyproject.toml` and `src/pylinkage/__init__.py`, commits, and tags):

   ```bash
   uv run bump-my-version bump patch  # For bug fixes (0.6.0 → 0.6.1)
   uv run bump-my-version bump minor  # For new features (0.6.0 → 0.7.0)
   uv run bump-my-version bump major  # For breaking changes (0.6.0 → 1.0.0)
   ```

   Use `--dry-run` to preview changes without applying them.

4. Push the commit and tag:

   ```bash
   git push && git push --tags
   ```

5. Publish a new [GitHub release](https://github.com/HugoFara/pylinkage/releases).
   This triggers automatic PyPI publishing.

## Caveats

Pylinkage is a small project to build 2D linkages in a simple way.
It is not intended for complex simulations.
If you want to do much more, my best advice is to start your own project.
If you simply want to have fun developing new features, you are welcome here!

## Contributions for everyone

Don't forget to drop a star, fork it or share it on social media.
This is a community project, and the bigger the community, the more it will thrive!
