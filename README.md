# Pylinkage

[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
[![Downloads](https://static.pepy.tech/personalized-badge/pylinkage?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/pylinkage)
[![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](https://codecov.io/gh/HugoFara/pylinkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/pylinkage/main/LICENSE.rst)

Pylinkage is a comprehensive Python library for planar linkage mechanisms. It provides tools to:

- **Define** linkages using joints (`Crank`, `Revolute`, `Linear`, etc.)
- **Simulate** kinematic motion with high-performance numba-compiled solvers
- **Optimize** geometry using Particle Swarm Optimization (PSO)
- **Synthesize** linkages from motion requirements (Burmester theory, Freudenstein's equation)
- **Analyze** symbolically using SymPy for closed-form expressions
- **Visualize** with multiple backends (Matplotlib, Plotly, SVG)

📚 **[Full Documentation](https://hugofara.github.io/pylinkage/)** — Complete tutorials, API reference, and examples.

## Installation

```shell
pip install pylinkage
```

For development:

```shell
git clone https://github.com/HugoFara/pylinkage.git
cd pylinkage
uv sync  # or pip install -e ".[dev]"
```

## Quick Start

### Define and Visualize a Four-Bar Linkage

```python
import pylinkage as pl

# Create a four-bar linkage
crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1)
pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1)

my_linkage = pl.Linkage(joints=(crank, pin))
pl.show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/Kinematic%20My%20four-bar%20linkage.gif)

### Synthesize a Linkage from Requirements

Design a four-bar where the coupler passes through specific points:

```python
from pylinkage.synthesis import path_generation

# Find linkages where coupler traces through these points
points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
result = path_generation(points)

for linkage in result.solutions:
    pl.show_linkage(linkage)
```

### Optimize with PSO

```python
@pl.kinematic_minimization
def fitness(loci, **_):
    # Define your objective based on joint trajectories
    tip_locus = tuple(x[-1] for x in loci)
    return pl.bounding_box(tip_locus)[0]  # Minimize min_y

bounds = pl.generate_bounds(my_linkage.get_num_constraints())
score, position, coords = pl.particle_swarm_optimization(
    eval_func=fitness, linkage=my_linkage, bounds=bounds, order_relation=min
)[0]
```

### Symbolic Analysis

Get closed-form trajectory expressions:

```python
from pylinkage.symbolic import fourbar_symbolic, compute_trajectory_numeric
import numpy as np

linkage = fourbar_symbolic(ground_length=4, crank_length=1, coupler_length=3, rocker_length=3)
params = {"L1": 1.0, "L2": 3.0, "L3": 3.0}
trajectories = compute_trajectory_numeric(linkage, params, np.linspace(0, 2*np.pi, 100))
```

## Features Overview

| Module | Purpose |
|--------|---------|
| `pylinkage.joints` | Joint types: `Crank`, `Revolute`, `Linear`, `Fixed`, `Static` |
| `pylinkage.linkage` | `Linkage` class for simulation via `step()` / `step_fast()` |
| `pylinkage.optimization` | PSO and grid search optimization |
| `pylinkage.synthesis` | Classical synthesis: function/path/motion generation |
| `pylinkage.symbolic` | SymPy-based symbolic computation and gradient optimization |
| `pylinkage.visualizer` | Matplotlib, Plotly, and SVG visualization backends |
| `pylinkage.assur` | Assur group decomposition and graph representation |
| `pylinkage.hypergraph` | Hierarchical component-based linkage definition |
| `pylinkage.solver` | High-performance numba-compiled simulation backend |

## Architecture

```text
Level 0: Geometry       → Pure numba math primitives
Level 1: Solver         → Numba-compiled Assur group solvers
Level 2: Hypergraph     → Abstract graph structures for linkage topology
Level 3: Assur          → Formal kinematic theory (DyadRRR, DyadRRP)
Level 4: User API       → Joint classes + Linkage orchestration
Level 5: Applications   → Optimization, Synthesis, Symbolic, Visualization
```

**Performance**: `step_fast()` achieves 1.5-2.5M steps/sec (4-7x faster than `step()`).

## Requirements

- Python ≥ 3.10
- Core: numpy, numba, matplotlib, pyswarms, sympy, plotly, drawsvg, tqdm

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
