# Pylinkage Modernization Guide

This document outlines recommended improvements to bring pylinkage up to modern
Python standards (2024+).

---

## 1. Packaging Consolidation

### Status: COMPLETED

The project now uses [uv](https://docs.astral.sh/uv/) for dependency management
with a modern `pyproject.toml`:

- Removed `setup.py`, `setup.cfg`, `requirements.txt`, `requirements-dev.txt`
- Using hatchling as build backend
- Lock file at `uv.lock`
- Python version pinned in `.python-version`

### Current Configuration

See `pyproject.toml` for the full configuration. Key highlights:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20.2",
    "tqdm>=4.40.0",
    "matplotlib>=3.3.4",
    "pyswarms>=1.3.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-cov>=4.0.0", "mypy>=1.0.0", "ruff>=0.4.0", "hypothesis>=6.0.0"]
docs = ["sphinx>=7.4.2", "sphinx-rtd-theme>=2.0.0", "myst-parser>=3.0.1"]

[tool.uv]
dev-dependencies = [...]  # All dev + docs dependencies
```

### Common Commands

```bash
uv sync                    # Install all dependencies
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run mypy pylinkage      # Type check
uv build                   # Build package
```

---

## 2. Type Hints Migration

### Current State

Types documented in docstrings (Sphinx/Napoleon style):

```python
def __init__(self, x=0, y=0, joint0=None, name=None):
    """
    :param x: Position on horizontal axis.
    :type x: float | None
    :param y: Position on vertical axis.
    :type y: float | None
    """
```

### Recommended Changes

Migrate to PEP 484/604 annotations:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Coord = tuple[float, float]

def __init__(
    self,
    x: float = 0,
    y: float = 0,
    joint0: Joint | Coord | None = None,
    name: str | None = None,
) -> None:
    """Create a joint.

    Args:
        x: Position on horizontal axis.
        y: Position on vertical axis.
        joint0: Parent joint or fixed coordinates.
        name: Human-readable identifier.
    """
```

### Files to Update (by priority)

1. **Core API** (user-facing):
   - `pylinkage/joints/joint.py` - Base `Static` class
   - `pylinkage/joints/crank.py` - `Crank` class
   - `pylinkage/joints/revolute.py` - `Revolute` class
   - `pylinkage/linkage/linkage.py` - `Linkage` class

2. **Optimization** (frequently used):
   - `pylinkage/optimization/utils.py` - decorators and bounds
   - `pylinkage/optimization/particle_swarm.py` - PSO function
   - `pylinkage/optimization/grid_search.py` - trials function

3. **Utilities**:
   - `pylinkage/geometry/core.py`
   - `pylinkage/geometry/secants.py`
   - `pylinkage/visualizer/*.py`

### Type Aliases to Define

Create `pylinkage/_types.py`:

```python
"""Type definitions for pylinkage."""
from __future__ import annotations
from typing import Callable, TypeAlias

Coord: TypeAlias = tuple[float, float]
Locus: TypeAlias = tuple[Coord, ...]
Constraints: TypeAlias = tuple[float, ...]
Bounds: TypeAlias = tuple[tuple[float, ...], tuple[float, ...]]
FitnessFunc: TypeAlias = Callable[..., float]
```

---

## 3. Python Version Support

### Python Version Current State

- `pyproject.toml` claims `>=3.7`
- CI only tests 3.9-3.12
- Contains Python 3.7 compatibility code

### Python Version Recommended Changes

**Remove legacy compatibility code** in `pylinkage/geometry/core.py:9-27`:

```python
# DELETE THIS:
if sys.version_info >= (3, 8, 0):
    dist = math.dist
else:
    warnings.warn('Unable to import dist from math.')
    dist = dist_builtin

# REPLACE WITH:
from math import dist
```

**Update minimum version** to 3.9 in `pyproject.toml`:

```toml
requires-python = ">=3.9"
```

**Add Python 3.13** to classifiers and CI matrix.

---

## 4. Test Infrastructure

### Test Infrastructure Current State

- Uses `unittest` with `python -m unittest discover`
- No coverage tracking
- ~415 lines of tests for ~2,132 lines of code

### Test Infrastructure Recommended Changes

**Migrate to pytest** with coverage:

```bash
# Install
pip install pytest pytest-cov

# Run with coverage
pytest --cov=pylinkage --cov-report=html
```

**Add missing test modules**:

```text
tests/
├── __init__.py
├── geometry/
│   └── test_geometry.py      # EXISTS - expand
├── joints/
│   └── test_joints.py        # EXISTS - expand
├── linkage/
│   └── test_linkage.py       # EXISTS - expand
├── optimization/
│   ├── test_optimizer.py     # EXISTS
│   └── collections/
│       └── test_collections.py
└── visualizer/               # ADD
    └── test_visualizer.py    # ADD
```

**Example new test** for `visualizer/`:

```python
# tests/visualizer/test_visualizer.py
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI

import pylinkage as pl

@pytest.fixture
def four_bar_linkage():
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1)
    pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1)
    return pl.Linkage(joints=(crank, pin), order=(crank, pin))

def test_plot_static_linkage(four_bar_linkage):
    """Test that static plotting doesn't raise."""
    fig, ax = pl.plot_static_linkage(four_bar_linkage)
    assert fig is not None
    assert ax is not None

def test_plot_kinematic_linkage(four_bar_linkage):
    """Test that kinematic plotting returns animation."""
    anim = pl.plot_kinematic_linkage(four_bar_linkage)
    assert anim is not None
```

**Target coverage**: 70%+ overall, 90%+ for core modules.

---

## 5. CI/CD Improvements

### Current Workflows

- `codeql-analysis.yml` - security scanning
- `python-package-conda.yml` - conda testing
- `versioning-test.yml` - multi-version Python
- `python-publish.yml` - PyPI publishing

### Recommended Additions

**Add to `versioning-test.yml`**:

```yaml
- name: Type check with mypy
  run: |
    pip install mypy
    mypy pylinkage --ignore-missing-imports

- name: Run tests with coverage
  run: |
    pip install pytest pytest-cov
    pytest --cov=pylinkage --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    fail_ci_if_error: false
```

**Add documentation build check** (new file `.github/workflows/docs.yml`):

```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -e ".[docs]"
      - name: Build documentation
        run: sphinx-build -W -b html sphinx/ docs/
```

---

## 6. Code Quality

### String Formatting

Standardize on f-strings throughout:

```python
# BEFORE (pylinkage/joints/joint.py:55)
"<{} {}carefully with name {}".format(...)

# AFTER
f"<{self.__class__.__name__} {self.coord()} with name {self.name}>"
```

### Module Exports

Add `__all__` to public modules for explicit API:

```python
# pylinkage/joints/__init__.py
__all__ = ["Crank", "Fixed", "Linear", "Revolute", "Static"]

from .joint import Static
from .crank import Crank
# ...
```

### Address TODOs

**`pylinkage/linkage/linkage.py:57`** - Automatic solving order:

- Either implement properly with tests, or
- Remove the experimental warning after validation

**`pylinkage/linkage/linkage.py:120`** - `hyperstaticity()` method:

- Add unit tests
- Document the formula used

### Exception Handling

Current exceptions in `pylinkage/exceptions.py` are well-structured. Consider adding:

```python
class OptimizationError(Exception):
    """Raised when optimization fails to converge."""
    pass
```

---

## 7. API Improvements

### Deprecation Strategy

For any breaking changes, use warnings:

```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function()
```

### Consider Adding

1. **Context manager for linkage simulation**:

```python
with linkage.simulation(iterations=100) as sim:
    for step, coords in sim:
        process(coords)
```

1. **Async optimization** for long-running PSO:

```python
async def optimize_async(linkage, fitness_func, ...):
    ...
```

1. **JSON/YAML serialization** for linkage definitions:

```python
linkage.to_json("my_linkage.json")
linkage = Linkage.from_json("my_linkage.json")
```

---

## 8. Documentation

### Documentation Current State

Well-structured Sphinx setup with autodoc.

### Recommended Improvements

1. **Add doctest examples** to docstrings:

```python
def circle_intersect(c1, r1, c2, r2):
    """Find intersection points of two circles.

    Examples:
        >>> circle_intersect((0, 0), 1, (1, 0), 1)
        ((0.5, 0.866...), (0.5, -0.866...))
    """
```

1. **Add tutorials** in `sphinx/tutorials/`:
   - Getting started guide
   - Custom joint creation
   - Advanced optimization techniques

1. **API changelog** - document breaking changes per version

---

## Implementation Roadmap

### Phase 1: Foundation (no breaking changes)

- [x] Consolidate to single `pyproject.toml`
- [x] Add pytest and coverage
- [x] Add ruff for linting
- [x] Set up uv for dependency management
- [x] Add mypy to CI
- [x] Remove Python 3.7/3.8 compatibility code (math.dist fallback)

### Phase 2: Type Safety (COMPLETED)

- [x] Add type aliases module (`pylinkage/_types.py`)
- [x] Annotate core classes (`Joint`, `Linkage`)
- [x] Annotate optimization functions
- [x] Enable strict mypy checks
- [x] Annotate exceptions module
- [x] Annotate visualizer modules (`core.py`, `static.py`, `animated.py`)
- [x] Fix `joints/static.py` type annotations and signature compatibility

### Phase 3: Test Coverage (COMPLETED)

- [x] Add visualizer tests (`tests/visualizer/test_visualizer.py`)
- [x] Add integration tests (`tests/integration/test_integration.py`)
- [x] Reach 70% coverage (achieved 82%)
- [x] Add property-based testing with hypothesis (`tests/geometry/test_geometry_hypothesis.py`)

### Phase 4: API Enhancements (COMPLETED)

- [x] Add serialization support (`Linkage.to_json()`, `Linkage.from_json()`)
- [x] Implement context manager API (`linkage.simulation()`)
- [x] Review and resolve TODOs (improved docs, added tests)
- [x] Add `__all__` exports to all public modules

---

## Quick Wins (< 1 hour each)

1. ~~Delete `setup.py` (empty file)~~ DONE
2. ~~Remove `math.dist` fallback in `geometry/core.py`~~ DONE
3. ~~Add `__all__` to `__init__.py` files~~ DONE
4. ~~Update Python version requirement to >=3.9~~ DONE
5. ~~Add Python 3.13 to CI matrix~~ DONE
6. Replace `.format()` with f-strings (use find/replace)

---

## Resources

- [PEP 621](https://peps.python.org/pep-0621/) - pyproject.toml metadata
- [PEP 484](https://peps.python.org/pep-0484/) - Type hints
- [bump-my-version](https://github.com/callowayproject/bump-my-version) -
  Modern version bumping
- [pytest-cov](https://pytest-cov.readthedocs.io/) - Coverage plugin
- [mypy](https://mypy.readthedocs.io/) - Static type checker
