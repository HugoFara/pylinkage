# Pylinkage

[![PyPI version fury.io](https://badge.fury.io/py/pylinkage.svg)](https://pypi.python.org/pypi/pylinkage/)
[![Downloads](https://static.pepy.tech/personalized-badge/pylinkage?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/pylinkage)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/HugoFara/pylinkage/main/LICENSE.rst)

Pylinkage is a Python library for building and optimizing planar linkages using
[Particle Swarm Optimization][pso-wiki].
It is still in beta, so don't hesitate to post pull requests or issues for
features you would like to see!

[pso-wiki]: https://en.wikipedia.org/wiki/Particle_swarm_optimization

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Example](#complete-example)
  - [Joints Definition](#joints-definition)
  - [Linkage Definition and Simulation](#linkage-definition-and-simulation)
  - [Visualization](#visualization)
  - [Optimization](#optimization)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Abstraction Levels](#abstraction-levels)
  - [Key Design Principles](#key-design-principles)
  - [Performance](#performance)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Installation

### Using pip

This package is on PyPI as [pylinkage](https://pypi.org/project/pylinkage/),
and can be installed using:

```shell
pip install pylinkage
```

### Using uv (recommended for development)

If you're using [uv](https://docs.astral.sh/uv/) as your package manager:

```shell
uv add pylinkage
```

For development, clone the repository and run:

```shell
uv sync
```

## Quick Start

Here's a minimal example creating and visualizing a **four-bar linkage**:

```python
import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0),
    distance0=3, distance1=1
)

# Create the linkage and visualize
my_linkage = pl.Linkage(joints=(crank, pin))
pl.show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/Kinematic%20My%20four-bar%20linkage.gif)

The package also supports **automatic optimization** to achieve specific movements.
See the [Complete Example](#complete-example) below for optimization details.

## Complete example

Let's start with a crank-rocker [four-bar linkage][four-bar], as a classic
mechanism.

[four-bar]: https://en.wikipedia.org/wiki/Four-bar_linkage

### Joints definition

First, we define at least one crank because we want a kinematic simulation.

```python
crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1)
```

Here we are actually doing the following:

* ``0, 1``: x and y initial coordinates of the **tail** of the crank link.
* ``joint0``: the position of the parent Joint to link with, here it is a fixed
  point in space. The pin will be created on the position of the parent, which
  is the head of the crank link.
* ``angle``: the crank will rotate with this angle, in radians, at each iteration.
* ``distance``: distance to keep constant between crank link tail and head.

Now we add a pin joint to close the kinematic loop.

```python
pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1)
```

In human language, here is what is happening:

* ``joint0``, ``joint1``: first and second ``Joint``s you want to link to, the
  order is not important.
* ``distance0``, ``distance1``: distance to keep constant between this joint
  and his two parents.

And here comes the trick:
Why do we specify initial coordinates ``3, 2``? They even seem incompatible
with distance to parents/parents' positions!

* This explanation is simple: mathematically a pin joint the intersection of
  two circles.
The intersection is often two points.
To choose the starting point, we calculate both intersection (when possible),
then we keep the intersection closer to the previous position as the solution.

> **Note:** A linkage with a single motor and only one pin joint?
> Behind the curtain, many joints are created on the fly.
> When you define a `Crank` joint, it creates a motor **and** a pin joint on
> the crank's link head.
> For a `Revolute` joint, it creates **3 pin joints**: one on each of its
> parents' positions, and one at its position,
> which forms a deformable triangle.
> This is why pylinkage code is so concise.

### Linkage definition and simulation

Once your linkage is finished, you can either use the `reload` method of each
`Joint` in a loop,
or put everything in a `Linkage` that will handle this for you.

Linkage definition is simple:

```python
my_linkage = pl.Linkage(joints=(crank, pin))
```

That's all!

Now we want to simulate it and to get the locus of ``pin``. Just use the
``step`` method of ``Linkage`` to make a complete rotation.

```python
locus = my_linkage.step()
```

You can also specify the number of steps with the ``iteration`` argument, or
subdivisions of each iteration with``dt``.

Let's recap.

```python
import pylinkage as pl

# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1)
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1
)

my_linkage = pl.Linkage(joints=(crank, pin))

locus = my_linkage.step()
```

### Visualization

First thing first, you made a cool linkage, but only you know what it is.
Let's add friendly names to joints, so the communication is simplified.

```python
crank.name = "B"
pin.name = "C"
# Linkage can also have names
my_linkage.name = "Four-bar linkage"
```

Then you can view your linkage!

```python
pl.show_linkage(my_linkage)
```

![A four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/Kinematic%20My%20four-bar%20linkage.gif)

Last recap, rearranging names:

```python
# Main motor
crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1, name="B")
# Close the loop
pin = pl.Revolute(
    3, 2, joint0=crank, joint1=(3, 0), 
    distance0=3, distance1=1, name="C"
)

# Linkage definition
my_linkage = pl.Linkage(
    joints=(crank, pin),
    order=(crank, pin),
    name="My four-bar linkage"
)

# Visualization
pl.show_linkage(my_linkage)
```

### Optimization

Now, we want automatic optimization of our linkage, using a certain criterion.
Let's find a four-bar linkage that makes a quarter of a circle.
It is a common problem if you want to build a [windshield wiper][wiper] for
instance.

[wiper]: https://en.wikipedia.org/wiki/Windscreen_wiper

Our objective function, often called the fitness function, is the following:

```python
# Save initial state for later reset
init_pos = my_linkage.get_coords()
init_constraints = my_linkage.get_num_constraints()

@pl.kinematic_minimization
def fitness_func(loci, **_kwargs):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minimization problem and the theoretic best score is 0.
    """
    # Locus of the Joint 'pin', last in linkage order
    tip_locus = tuple(x[-1] for x in loci)
    # We get the bounding box
    curr_bb = pl.bounding_box(tip_locus)
    # We set the reference bounding box, in order (min_y, max_x, max_y, min_x)
    ref_bb = (0, 5, 3, 0)
    # Our score is the square sum of the edge distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))
```

Please note that it is a *minimization* problem, with 0 as lower bound.
On the first line, you notice a decorator; which plays a crucial role:

* The decorator arguments are (linkage, constraints), it can also receive ``init_pos``
* It sets the linkage with the constraints.
* Then it verifies if the linkage can do a complete crank turn.
  * If it can, pass the arguments and the resulting loci (path of joints) to
    the decorated function.
  * If not, return the penalty. In a minimization problem the penalty will be ``float('inf')``.
* The decorated function should return the score of this linkage.  

With this constraint, the best theoretic score is 0.0.

Let's start with a candide optimization, the [trial-and-error][trial] method.

[trial]: https://en.wikipedia.org/wiki/Trial_and_error
Here it is a serial test of switches.

```python
# Exhaustive optimization as an example ONLY
score, position, coord = pl.trials_and_errors_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    divisions=25,
    n_results=1,
    order_relation=min,
)[0]
```

Here the problem is simple enough, so that method takes only a few seconds and
returns 0.05.

However, with more complex linkages, you need something more robust and more
efficient.
Then we will use [particle swarm optimization][pso].

[pso]: https://en.wikipedia.org/wiki/Particle_swarm_optimization

Here are the principles:

* The parameters are the geometric constraints (the dimensions) of the linkage.
* A dimension set (an n-uplet) is called a *particle* or an *agent*. Think of
  it like a bee.
* The particles move in an n-vectorial space. That is, if we have n geometric
  constraints, the particles move in an n-D space.
* Together, the particles form the *swarm*.
* Each time they move, their score is evaluated by our fitness function.
* They know their best score, and know the current score of their neighbor.
* Together they will try to find the extreme in the space. Here it is a minimum.

It is particularly relevant when the fitness function is not resource-greedy.

```python
# Reset the linkage to initial state before optimization
my_linkage.set_num_constraints(init_constraints)
my_linkage.set_coords(init_pos)

# Generate bounds for the optimization search space
bounds = pl.generate_bounds(my_linkage.get_num_constraints())

score, position, coord = pl.particle_swarm_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    bounds=bounds,
    order_relation=min,
)[0]
```

Here the result can vary, but it is rarely above 0.2.

So we made something that says it works, let's verify it:

![An optimized four-bar linkage animated](https://github.com/HugoFara/pylinkage/raw/main/docs/assets/Kinematic%20Windscreen%20wiper.gif)

With a bit of imagination, you have a wonderful windshield wiper!

## Project Structure

The recommended workflow for using pylinkage is:

1. **Define joints**: Use joint types from `pylinkage.joints` (`Crank`, `Revolute`, etc.)
2. **Create linkage**: Combine joints in a `Linkage` from `pylinkage.linkage`
3. **Simulate** (optional): Run `linkage.step()` to compute motion
4. **Optimize**: Use `pylinkage.optimization` functions to find optimal dimensions
5. **Visualize**: Display results with `pylinkage.visualizer`

## Architecture

Pylinkage is organized into distinct abstraction layers, from low-level numerics
to high-level user APIs. This design ensures separation of concerns and enables
both ease of use and high performance.

### Abstraction Levels

```text
Level 0: Geometry (pure numba math primitives)
├── geometry/core.py       → cyl_to_cart, get_nearest_point, dist
└── geometry/secants.py    → circle_intersect, circle_line_intersection
         ↑ Single source of mathematical truth

Level 1: Solver (numba-compiled Assur group solvers)
└── solver/joints.py
    ├── solve_crank()      → Driver rotation
    ├── solve_revolute()   → RRR dyad (circle-circle intersection)
    ├── solve_linear()     → RRP dyad (circle-line intersection)
    └── solve_fixed()      → Deterministic polar constraint
         ↑ Single source of solving logic (used by all higher layers)

Level 2: Hypergraph (abstract mathematical structures)
└── hypergraph/
    ├── core.py            → Node, Edge, Hyperedge
    ├── graph.py           → HypergraphLinkage
    └── components.py      → Component, Port (reusable subgraphs)
         ↑ Foundational graph theory (no dependencies on assur)

Level 3: Assur (kinematic theory built on hypergraph)
└── assur/
    ├── graph.py           → LinkageGraph, Node, Edge
    ├── groups.py          → DyadRRR, DyadRRP (delegate to solver)
    └── decomposition.py   → Assur group decomposition algorithm
         ↑ Formal kinematic analysis

Level 4: User API (thin wrappers for ease of use)
├── joints/
│   ├── crank.py           → Crank (calls solve_crank)
│   ├── revolute.py        → Revolute (calls solve_revolute)
│   ├── linear.py          → Linear (calls solve_linear)
│   └── fixed.py           → Fixed (calls solve_fixed)
└── linkage/
    └── linkage.py         → Linkage class (orchestrates simulation)
         ↑ Validation + delegation to solver

Level 5: Applications
├── optimization/          → PSO, grid search algorithms
├── visualizer/            → Matplotlib, Plotly, SVG backends
└── bridge/                → Linkage ↔ SolverData conversion
```

### Key Design Principles

1. **Single Source of Truth**: The `solver/joints.py` module contains the
   canonical implementations of all solving algorithms. Joint classes and
   Assur groups delegate to these functions.

2. **Hypergraph as Foundation**: The `hypergraph` module provides abstract
   mathematical structures. The `assur` module builds kinematic theory on top
   of it, not the other way around.

3. **Pure Numerics in Solver**: The solver module uses only numpy and numba
   with no Python object dependencies. This enables maximum performance for
   optimization loops.

4. **Thin User API**: Joint classes (`Crank`, `Revolute`, etc.) are thin
   wrappers that handle validation and user-friendly errors, then delegate
   to solver functions.

5. **Assur Group Theory**: The package is built on [Assur group][assur-wiki]
   decomposition, a formal approach to analyzing planar linkages:
   - **RRR dyad** (3 revolute joints): Solved via circle-circle intersection
   - **RRP dyad** (2 revolute + 1 prismatic): Solved via circle-line intersection

[assur-wiki]: https://en.wikipedia.org/wiki/Assur_group

### Performance

The solver provides two simulation methods:

- `linkage.step()`: Python-based, ~300-400k steps/sec
- `linkage.step_fast()`: Numba-compiled, ~1.5-2.5M steps/sec (4-7x faster)

Use `step_fast()` for optimization loops where performance matters.

## Requirements

Python 3, numpy for calculation, matplotlib for drawing, and standard libraries.
You will also need PySwarms for the Particle Swarm Optimization.

## Contributing

**Pylinkage is open to contribution**.
I may consider any pull request, but I ask you to respect the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
and follow the guidelines as defined in [CONTRIBUTING.md](CONTRIBUTING.md).
