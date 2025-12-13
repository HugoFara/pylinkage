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
uv run ruff format .                 # Format code
uv run mypy src/pylinkage            # Type check
```

### Building

```bash
uv build                             # Build wheel and sdist
uv run sphinx-build -b html docs/source docs/  # Build documentation
```

## Architecture

### Package Structure

- **src/pylinkage/joints/**: Joint types that form linkage building blocks
  - `Static`: Fixed point in space (base class for all joints)
  - `Crank`: Rotating motor joint (creates a motor + pin joint)
  - `Revolute`: Pin joint connecting two parents
    (creates 3 internal pin joints forming a deformable triangle)
  - `Pivot`: Low-level pin joint (used internally by Revolute)
  - `Fixed`: Static joint with fixed distance constraints
  - `Linear`: Joint constrained to move along a line

- **src/pylinkage/linkage/**: Linkage class that orchestrates joint collections
  - `Linkage`: Main class managing joints, solving order, and simulation
    via `step()` method
  - `Simulation`: Container for simulation results (loci, steps)
  - `analysis.py`: Helper functions like `bounding_box()` and `kinematic_default_test()`

- **src/pylinkage/optimization/**: Optimization algorithms
  - `grid_search.py`: `trials_and_errors_optimization()` - exhaustive search
  - `particle_swarm.py`: `particle_swarm_optimization()` - PSO using PySwarms
  - `utils.py`: `@kinematic_minimization`/`@kinematic_maximization` decorators
    and `generate_bounds()`

- **src/pylinkage/geometry/**: 2D geometry utilities
  - `core.py`: Distance calculations, coordinate conversions
  - `secants.py`: Circle-circle and circle-line intersections

- **src/pylinkage/visualizer/**: Multi-backend visualization
  - `static.py`, `animated.py`: Matplotlib backend (`show_linkage()`, GIF output)
  - `plotly_viz.py`: Plotly backend for interactive HTML (`plot_linkage_plotly()`)
  - `drawsvg_viz.py`: drawsvg backend for publication-quality SVG (`save_linkage_svg()`)
  - `pso_plots.py`: PSO visualization dashboards

- **src/pylinkage/hypergraph/**: Hierarchical hypergraph representation (new)
  - Abstract mathematical foundation for linkage definition
  - `HypergraphLinkage`: Graph with nodes, edges, and hyperedges
  - `Component`: Reusable linkage subgraph with ports and parameters
  - `HierarchicalLinkage`: Composition of component instances
  - Built-in components: `FOURBAR`, `CRANK_SLIDER`, `DYAD`
  - Conversion functions: `to_linkage()`, `from_linkage()`, `to_assur_graph()`

- **src/pylinkage/assur/**: Assur group decomposition
  - Graph-based representation using formal kinematic theory
  - `LinkageGraph`: Nodes (joints) and edges (links)
  - Assur groups: `DyadRRR`, `DyadRRP`, `DyadRPR`, `DyadPRR`
  - `decompose_assur_groups()`: Structural decomposition algorithm
  - `graph_to_linkage()`: Convert graph representation to Linkage

- **src/pylinkage/solver/**: High-performance numba simulation backend
  - Pure-numba JIT-compiled solver for optimization hot loops
  - `SolverData`: Numeric arrays replacing Python objects
  - `simulate()`: Fast trajectory computation
  - `linkage_to_solver_data()`: Convert Linkage for fast simulation

### Key Patterns

**Linkage Definition Flow:**

1. Create joint instances (Crank, Revolute, etc.) with parent references
2. Wrap joints in a `Linkage(joints=..., order=...)`
3. Call `linkage.step()` to simulate one full rotation cycle
4. Use `show_linkage()` to visualize

**Alternative Definition via Hypergraph:**

1. Use component instances from library (`FOURBAR`, `DYAD`, etc.)
2. Connect via `HierarchicalLinkage` with `Connection` objects
3. Call `flatten()` then `to_linkage()` to get simulatable Linkage

**Alternative Definition via Assur Graph:**

1. Create `LinkageGraph` with `Node` and `Edge` objects
2. Use `decompose_assur_groups()` for structural analysis
3. Call `graph_to_linkage()` to convert to Linkage

**Optimization Flow:**

1. Define a fitness function decorated with `@kinematic_minimization` or `@kinematic_maximization`
2. Generate bounds with `generate_bounds(linkage.get_num_constraints())`
3. Call `particle_swarm_optimization()` or `trials_and_errors_optimization()`
4. Apply results via `linkage.set_num_constraints(position)`

**Constraint System:**

- `get_num_constraints()`: Returns flat list of distances/angles
- `set_num_constraints()`: Applies constraints back to joints
- `get_coords()`/`set_coords()`: Joint positions (used for initial positions in optimization)

**Exceptions:**

- `UnbuildableError`: Raised when a linkage cannot be assembled (geometric impossibility)
- `UnderconstrainedError`: Raised when a linkage is underconstrained (too few constraints)
- `NotCompletelyDefinedError`: Raised when joint parameters are incomplete

## Dependencies

Requires Python >= 3.10

Core: numpy, numba, matplotlib, pyswarms, tqdm, plotly, drawsvg

Dev (managed via uv): pytest, pytest-cov, hypothesis, mypy, ruff, sphinx, sphinx-rtd-theme, myst-parser
