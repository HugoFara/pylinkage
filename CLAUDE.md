# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

Pylinkage is a Python library for building and optimizing planar linkages using
Particle Swarm Optimization (PSO). It provides tools to define kinematic
linkages, simulate their motion, optimize their geometry against objective
functions, and visualize the results.

## Common Commands

### Setup and Dependencies

```bash
uv sync                              # Install all dependencies (including dev)
uv sync --no-dev                     # Install only production dependencies
```

### Testing

```bash
uv run pytest                        # Run all tests
uv run pytest tests/joints/          # Run specific test directory
uv run pytest -k "test_buildable"    # Run tests matching pattern
uv run pytest --cov=pylinkage        # Run with coverage
```

### Linting and Type Checking

```bash
uv run ruff check .                  # Lint code
uv run ruff check . --fix            # Lint and auto-fix
uv run mypy pylinkage                # Type check
```

### Building

```bash
uv build                             # Build wheel and sdist
uv run sphinx-build -b html sphinx/ docs/  # Build documentation
```

## Architecture

### Package Structure

- **pylinkage/joints/**: Joint types that form linkage building blocks
  - `Static`: Fixed point in space (base class)
  - `Crank`: Rotating motor joint (creates a motor + pin joint)
  - `Revolute`: Pin joint connecting two parents
    (creates 3 internal pin joints forming a deformable triangle)
  - `Fixed`: Static joint with fixed distance constraints
  - `Linear`: Joint constrained to move along a line

- **pylinkage/linkage/**: Linkage class that orchestrates joint collections
  - `Linkage`: Main class managing joints, solving order, and simulation
    via `step()` method
  - `analysis.py`: Helper functions like `bounding_box()` and `kinematic_default_test()`

- **pylinkage/optimization/**: Optimization algorithms
  - `grid_search.py`: `trials_and_errors_optimization()` - exhaustive search
  - `particle_swarm.py`: `particle_swarm_optimization()` - PSO using PySwarms
  - `utils.py`: `@kinematic_minimization`/`@kinematic_maximization` decorators
    and `generate_bounds()`

- **pylinkage/geometry/**: 2D geometry utilities
  - `core.py`: Distance calculations, coordinate conversions
  - `secants.py`: Circle-circle and circle-line intersections

- **pylinkage/visualizer/**: Matplotlib-based visualization
  - `static.py`: Static linkage plots
  - `animated.py`: `show_linkage()` for animated GIF output

### Key Patterns

**Linkage Definition Flow:**

1. Create joint instances (Crank, Revolute, etc.) with parent references
2. Wrap joints in a `Linkage(joints=..., order=...)`
3. Call `linkage.step()` to simulate one full rotation cycle
4. Use `show_linkage()` to visualize

**Optimization Flow:**

1. Define a fitness function decorated with `@kinematic_minimization` or `@kinematic_maximization`
2. Generate bounds with `generate_bounds(linkage.get_num_constraints())`
3. Call `particle_swarm_optimization()` or `trials_and_errors_optimization()`
4. Apply results via `linkage.set_num_constraints(position)`

**Constraint System:**

- `get_num_constraints()`: Returns flat list of distances/angles
- `set_num_constraints()`: Applies constraints back to joints
- `get_coords()`/`set_coords()`: Joint positions (used for initial positions in optimization)

## Dependencies

Core: numpy, matplotlib, pyswarms, tqdm

Dev (managed via uv): pytest, pytest-cov, mypy, ruff, sphinx, sphinx-rtd-theme, myst-parser
