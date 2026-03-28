# Pylinkage

[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
[![Downloads](https://static.pepy.tech/personalized-badge/pylinkage?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/pylinkage)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://codecov.io/gh/HugoFara/pylinkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/pylinkage/main/LICENSE.rst)

Pylinkage is a comprehensive Python library for planar linkage mechanisms. It provides tools to:

- **Define** linkages using joints (`Crank`, `Revolute`, `Linear`, etc.)
- **Simulate** kinematic motion with high-performance numba-compiled solvers
- **Optimize** geometry using Particle Swarm Optimization (PSO)
- **Synthesize** linkages from motion requirements (Burmester theory, Freudenstein's equation)
- **Analyze** symbolically using SymPy for closed-form expressions
- **Visualize** with multiple backends (Matplotlib, Plotly, SVG)

📚 **[Full Documentation](https://hugofara.github.io/pylinkage/)** — Complete tutorials, API reference, and examples.

### Related Projects

- **[pylinkage-editor](https://github.com/HugoFara/pylinkage-editor)** — Visual linkage design tool with an easy-to-use interface. Draw mechanisms interactively, run synthesis from the GUI, and export results.
- **[leggedsnake](https://github.com/HugoFara/leggedsnake)** — Dynamic walking simulation built on pylinkage. Adds pymunk physics, genetic algorithm optimization, and walking-specific fitness evaluation.

## Installation

```shell
pip install pylinkage            # Core only (~35 MB): define, simulate, and build linkages
pip install pylinkage[full]      # Everything (~400 MB): all optional backends included
```

Install only what you need:

| Extra | What it adds |
|-------|-------------|
| `numba` | JIT-compiled solvers (1.5-2.5M steps/sec) |
| `scipy` | Differential evolution optimizer, synthesis solvers |
| `pso` | Particle Swarm Optimization via pyswarms |
| `symbolic` | SymPy-based closed-form expressions and gradient optimization |
| `viz` | Matplotlib visualization and animation |
| `plotly` | Interactive HTML visualization |
| `svg` | Publication-quality SVG export via drawsvg |

Extras can be combined: `pip install pylinkage[viz,scipy,pso]`

For development:

```shell
git clone https://github.com/HugoFara/pylinkage.git
cd pylinkage
uv sync  # or pip install -e ".[full,dev]"
```

## Quick Start

### Define and Visualize a Four-Bar Linkage

Using the component-based API (recommended). Visualization requires `pip install pylinkage[viz]`.

```python
from pylinkage.components import Ground
from pylinkage.actuators import Crank
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer import show_linkage  # requires viz extra

# Define ground pivots
O1 = Ground(0, 0, name="O1")
O2 = Ground(3, 0, name="O2")

# Create crank (motor-driven input)
crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.31, name="crank")

# Create rocker via RRR dyad (circle-circle intersection)
rocker = RRRDyad(
    anchor1=crank.output,
    anchor2=O2,
    distance1=3.0,
    distance2=1.0,
    name="rocker"
)

my_linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")
show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/Kinematic%20My%20four-bar%20linkage.gif)

### Alternative: Links-First Builder

For a more mechanical engineering-oriented approach, use `MechanismBuilder` to define links with their lengths first, then connect them:

```python
from pylinkage.mechanism import MechanismBuilder

# Define links by their lengths, then connect with joints
mechanism = (
    MechanismBuilder("four-bar")
    .add_ground_link("ground", ports={"O1": (0, 0), "O2": (4, 0)})
    .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
    .add_link("coupler", length=3.5)
    .add_link("rocker", length=3.0)
    .connect("crank.tip", "coupler.0")
    .connect("coupler.1", "rocker.0")
    .connect("rocker.1", "ground.O2")
    .build()
)

# Joint positions are computed automatically from link lengths
for positions in mechanism.step():
    print(positions)
```

### Synthesize a Linkage from Requirements

Requires `pip install pylinkage[scipy]`. Design a four-bar where the coupler passes through specific points:

```python
from pylinkage.synthesis import path_generation

# Find linkages where coupler traces through these points
points = [(0, 1), (1, 2), (2, 1.5), (3, 0)]
result = path_generation(points)

for linkage in result.solutions:
    pl.show_linkage(linkage)
```

### Optimize with PSO

Requires `pip install pylinkage[pso]`.

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

Requires `pip install pylinkage[symbolic]`. Get closed-form trajectory expressions:

```python
from pylinkage.symbolic import fourbar_symbolic, compute_trajectory_numeric
import numpy as np

linkage = fourbar_symbolic(ground_length=4, crank_length=1, coupler_length=3, rocker_length=3)
params = {"L1": 1.0, "L2": 3.0, "L3": 3.0}
trajectories = compute_trajectory_numeric(linkage, params, np.linspace(0, 2*np.pi, 100))
```

## Features Overview

| Module | Purpose | Extras needed |
|--------|---------|---------------|
| `pylinkage.components` | Base components: `Ground`, `Component` | — |
| `pylinkage.actuators` | Motor drivers: `Crank`, `LinearActuator` | — |
| `pylinkage.dyads` | Assur groups: `RRRDyad`, `RRPDyad`, `FixedDyad` | — |
| `pylinkage.simulation` | `Linkage` class for simulation via `step()` / `step_fast()` | — |
| `pylinkage.mechanism` | Low-level Links+Joints model and `MechanismBuilder` | — |
| `pylinkage.assur` | Assur group decomposition and graph representation | — |
| `pylinkage.hypergraph` | Hierarchical component-based linkage definition | — |
| `pylinkage.solver` | High-performance numba-compiled simulation backend | `numba` |
| `pylinkage.optimization` | PSO, differential evolution, grid search | `pso`, `scipy` |
| `pylinkage.synthesis` | Classical synthesis: function/path/motion generation | `scipy` |
| `pylinkage.symbolic` | SymPy-based symbolic computation and gradient optimization | `symbolic` |
| `pylinkage.visualizer` | Matplotlib, Plotly, and SVG visualization backends | `viz`, `plotly`, `svg` |

## Architecture

```text
Level 0: Geometry       → Pure math primitives (numba-accelerated when installed)
Level 1: Solver         → Assur group solvers (numba-accelerated when installed)
Level 2: Hypergraph     → Abstract graph structures for linkage topology
Level 3: Assur          → Formal kinematic theory (DyadRRR, DyadRRP)
Level 4: User API       → Joint classes + Linkage orchestration
Level 5: Applications   → Optimization, Synthesis, Symbolic, Visualization
```

**Performance**: With the `numba` extra, `step_fast()` achieves 1.5-2.5M steps/sec (4-7x faster than `step()`). Without numba, the same code runs in pure Python/NumPy.

## Requirements

- Python ≥ 3.10
- Core: numpy, tqdm
- Optional (via extras): numba, scipy, sympy, pyswarms, matplotlib, plotly, drawsvg

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
