# Pylinkage

[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
[![Downloads](https://static.pepy.tech/personalized-badge/pylinkage?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/pylinkage)
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen)](https://codecov.io/gh/HugoFara/pylinkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/pylinkage/main/LICENSE.rst)

Pylinkage lets you design planar linkage mechanisms by specifying the motion you need. Tell it where you want a coupler point to go, and it finds mechanism dimensions automatically using classical synthesis theory (Burmester, Freudenstein) and metaheuristic optimization (PSO, differential evolution). You can then simulate, analyze, visualize, and export to DXF or STEP for fabrication.

```python
from pylinkage.synthesis import path_generation
from pylinkage.visualizer import plot_kinematic_linkage

# "I need a coupler that passes through these four points"
result = path_generation([(0, 0), (1, 1), (2, 1), (3, 0)])
plot_kinematic_linkage(result.solutions[0])
```

![Path generation result](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/synthesis_path_generation.png)

## Installation

```shell
pip install pylinkage            # Core only: define, simulate, and build linkages
pip install pylinkage[full]      # Everything: all optional backends included
```

Install only what you need:

| Extra | What it adds |
|-------|-------------|
| `numba` | JIT-compiled solvers (1.5-2.5M steps/sec) |
| `scipy` | Differential evolution optimizer, synthesis solvers |
| `symbolic` | SymPy-based closed-form expressions and gradient optimization |
| `viz` | Matplotlib visualization and animation |
| `plotly` | Interactive HTML visualization |
| `svg` | Publication-quality SVG export via drawsvg |

Extras can be combined: `pip install pylinkage[viz,scipy,numba]`

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
from pylinkage.visualizer import plot_kinematic_linkage

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
plot_kinematic_linkage(my_linkage)
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

### Optimize with PSO

PSO is built-in (no extra needed).

```python
import pylinkage as pl

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

Output (the rocker angle as a function of crank angle):

```
theta_4(theta_2) = atan2(N2, N1) + atan2(sqrt(N1**2 + N2**2 - N3**2), N3)
```

where N1, N2, N3 are Freudenstein coefficients derived from your link lengths.

## Tutorials

The [`docs/notebooks/`](docs/notebooks/) directory contains hands-on tutorials that walk through each major feature:

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 01 | [Straight-Line Synthesis](docs/notebooks/01_straight_line_synthesis.ipynb) | Design a mechanism that traces a straight line from scratch |
| 02 | [Optimize a Coupler Curve](docs/notebooks/02_optimize_coupler_curve.ipynb) | Use PSO to shape a four-bar coupler path |
| 03 | [Tolerance Analysis](docs/notebooks/03_tolerance_analysis.ipynb) | Monte Carlo analysis for manufacturing variation |
| 04 | [Cam-Follower Timing](docs/notebooks/04_cam_follower_timing.ipynb) | Design cam profiles with motion laws |
| 05 | [Symbolic Coupler Curve](docs/notebooks/05_symbolic_coupler_curve.ipynb) | Closed-form trajectory expressions with SymPy |
| 06 | [Function Generation](docs/notebooks/06_function_generation.ipynb) | Match input/output angle relationships (Freudenstein) |
| 07 | [Motion Generation](docs/notebooks/07_motion_generation.ipynb) | Guide a rigid body through specified poses (Burmester) |
| 08 | [Velocity & Acceleration](docs/notebooks/08_velocity_acceleration.ipynb) | Compute joint velocities and accelerations |
| 09 | [Transmission Angle & DOF](docs/notebooks/09_transmission_angle_and_dof.ipynb) | Evaluate mechanism quality and mobility |
| 10 | [Mechanism Builder](docs/notebooks/10_mechanism_builder.ipynb) | Link-first definition with `MechanismBuilder` |
| 11 | [Multi-Objective Optimization](docs/notebooks/11_multi_objective_and_scipy_optimization.ipynb) | Pareto-optimal design with NSGA-II and scipy |
| 12 | [Three Synthesis Problems](docs/notebooks/12_three_synthesis_problems.ipynb) | Side-by-side comparison of path, function, and motion synthesis |
| 13 | [Population Abstractions](docs/notebooks/13_population_abstractions.ipynb) | Batch simulation, ranking, and filtering of mechanism families |
| 14 | [Topology Enumeration](docs/notebooks/14_topology_enumeration_and_synthesis.ipynb) | Enumerate and synthesize across all valid topologies |

## What Else Can It Do?

Pylinkage also supports velocity and acceleration analysis, cam-follower mechanisms with configurable motion laws, transmission angle evaluation, Monte Carlo tolerance analysis for manufacturing, multi-objective optimization (NSGA-II/III via pymoo), and export to DXF and STEP for CNC and CAD workflows. See the [tutorials](#tutorials) for details.

## Architecture

```text
Level 0: Geometry       → Pure math primitives (numba-accelerated when installed)
Level 1: Solver         → Assur group solvers (numba-accelerated when installed)
Level 2: Components     → Ground, Crank, RRRDyad, LinearActuator, cam-followers
Level 3: Simulation     → Linkage orchestration, step(), step_fast()
Level 4: Applications   → Optimization, Synthesis, Symbolic, Visualization
```

**Performance**: With the `numba` extra, `step_fast()` achieves 1.5-2.5M steps/sec (4-7x faster than `step()`). Without numba, the same code runs in pure Python/NumPy.

<details>
<summary>Full module reference</summary>

| Module | Purpose | Extras needed |
|--------|---------|---------------|
| `pylinkage.components` | Base components: `Ground`, `Component` | — |
| `pylinkage.actuators` | Motor drivers: `Crank`, `LinearActuator` | — |
| `pylinkage.dyads` | Assur groups: `RRRDyad`, `RRPDyad`, `FixedDyad` | — |
| `pylinkage.simulation` | `Linkage` class for simulation via `step()` / `step_fast()` | — |
| `pylinkage.mechanism` | Low-level Links+Joints model and `MechanismBuilder` | — |
| `pylinkage.solver` | High-performance numba-compiled simulation backend | `numba` |
| `pylinkage.optimization` | PSO, differential evolution, grid search | `scipy` (DE only) |
| `pylinkage.synthesis` | Classical synthesis: function/path/motion generation | `scipy` |
| `pylinkage.symbolic` | SymPy-based symbolic computation and gradient optimization | `symbolic` |
| `pylinkage.visualizer` | Matplotlib, Plotly, SVG, DXF, and STEP export | `viz`, `plotly`, `svg` |
| `pylinkage.assur` | Assur group decomposition and graph representation | — |
| `pylinkage.hypergraph` | Hierarchical component-based linkage definition | — |

</details>

## Requirements

- Python >= 3.10
- Core: numpy, tqdm
- Optional (via extras): numba, scipy, sympy, matplotlib, plotly, drawsvg

## Related Projects

- **[pylinkage-editor](https://github.com/HugoFara/pylinkage-editor)** — Visual linkage design tool with an easy-to-use interface. Draw mechanisms interactively, run synthesis from the GUI, and export results.
- **[leggedsnake](https://github.com/HugoFara/leggedsnake)** — Dynamic walking simulation built on pylinkage. Adds pymunk physics, genetic algorithm optimization, and walking-specific fitness evaluation.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
