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
uv run task test                     # Run all tests
uv run task test-cov                 # Run with coverage
uv run pytest tests/joints/          # Run specific test directory
uv run pytest -k "test_buildable"    # Run tests matching pattern
```

### Linting and Type Checking

```bash
uv run task lint                     # Lint code
uv run task lint-fix                 # Lint and auto-fix
uv run task format                   # Format code
uv run task typecheck                # Type check
```

### Building

```bash
uv build                             # Build wheel and sdist
uv run task docs                     # Build documentation
uv run task docs-clean               # Clean documentation artifacts
```

### Running the Web App

```bash
uv sync --extra app                  # Install Streamlit dependency
uv run streamlit run app/main.py     # Launch the Pylinkage Editor web UI
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

- **src/pylinkage/synthesis/**: Classical mechanism synthesis methods
  - `function_generation.py`: Match input/output angle relationships (Freudenstein)
  - `path_generation.py`: Coupler point traces through specified points
  - `motion_generation.py`: Guide body through specified poses
  - `burmester.py`: Burmester theory for circle point/center point curves
  - `utils.py`: Grashof criterion checking (`is_grashof()`, `is_crank_rocker()`)
  - `conversion.py`: `fourbar_from_lengths()`, `solution_to_linkage()`

- **src/pylinkage/symbolic/**: Symbolic computation using SymPy
  - `joints.py`: `SymStatic`, `SymCrank`, `SymRevolute` symbolic joint classes
  - `linkage.py`: `SymbolicLinkage` for symbolic trajectory expressions
  - `solver.py`: `solve_linkage_symbolically()`, `compute_trajectory_numeric()`
  - `optimization.py`: `SymbolicOptimizer` for gradient-based optimization
  - `geometry.py`: Symbolic geometry primitives

- **src/pylinkage/bridge/**: Conversion utilities between representations
  - Bridges between Linkage, Assur graph, Hypergraph, and Solver representations

- **app/**: Streamlit-based web editor for interactive linkage design
  - `main.py`: Entry point, run with `uv run streamlit run app/main.py`
  - `components/`: UI components (sidebar, visualization panel)
  - `state.py`: Session state management

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

**Synthesis Flow (Design from requirements):**

1. Define precision points, angle pairs, or poses depending on synthesis type
2. Call `function_generation()`, `path_generation()`, or `motion_generation()`
3. Iterate over `SynthesisResult.solutions` to get candidate linkages
4. Validate with `grashof_check()` or `is_crank_rocker()`
5. Convert to `Linkage` with `solution_to_linkage()` for simulation

**Symbolic Computation Flow:**

1. Create symbolic linkage with `fourbar_symbolic()` or `linkage_to_symbolic()`
2. Get closed-form expressions via `solve_linkage_symbolically()`
3. Evaluate numerically with `compute_trajectory_numeric()`
4. Optional: Use `SymbolicOptimizer` for gradient-based optimization

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

Core: numpy, numba, scipy, matplotlib, pyswarms, tqdm, plotly, drawsvg, sympy

Dev (managed via uv): pytest, pytest-cov, hypothesis, mypy, ruff, sphinx, sphinx-rtd-theme, myst-parser, taskipy

App (optional): streamlit
