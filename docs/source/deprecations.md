# Deprecations

Names that still work but are on their way out, with what to use instead.

## What is public

A name is public when it is listed in the `__all__` of a package, and a
package is public when it is reachable as an attribute of `pylinkage`
(`pylinkage.synthesis`, `pylinkage.optimization.collections`, …). That is the
whole rule. The policy below applies to those names and nothing else.

Two consequences:

- **Modules inside a package are implementation.** `from
  pylinkage.synthesis.path_generation import path_generation` works today and
  may stop working without notice; `from pylinkage.synthesis import
  path_generation` is the promise. Newer modules carry a leading underscore
  to make this visible (`population._member`, `synthesis._types`); older ones
  will be renamed as they are touched.
- **Each name has one home.** `Ground` lives in `pylinkage.components`, not
  also in `pylinkage.dyads`. The top-level package re-exports the definition
  path (`Ground`, `Crank`, `RRRDyad`, `Linkage`, …) for convenience; every
  other second location is a deprecated alias, listed below.

The surface is pinned name by name in `tests/test_public_api.py`, so it cannot
change without a diff in that file.

Some public names are **provisional**: exported and documented, but their
shape may still change in a minor release. They are the newer subsystems —
`pylinkage.topology`, `pylinkage.solver`, the co-design family in
`pylinkage.optimization` (`co_optimize`, `CoOptimizationConfig`, …), the
multi-topology synthesis in `pylinkage.synthesis` (`NBarSolution`,
`generalized_synthesis`, …), hierarchical composition in
`pylinkage.hypergraph`, and the CAD export in `pylinkage.visualizer`. A change
to one is announced in the changelog, not through a warning.

## How deprecation works here

pylinkage follows [semantic versioning](https://semver.org/spec/v2.0.0.html).
A public name is never removed without notice:

1. **Announced.** The name keeps working and keeps resolving to the same
   object, but reading it raises a `DeprecationWarning` naming the replacement
   and the release that will drop it.
2. **Removed.** No earlier than the next *major* release.

`DeprecationWarning` is silent by default in Python, so nothing in your output
changes until you go looking. To see them:

```bash
python -W error::DeprecationWarning your_script.py   # fail on any use
python -W default::DeprecationWarning your_script.py # just print them
```

Under pytest, `filterwarnings = ["error::DeprecationWarning"]` in your config
turns each one into a test failure, which is the cheapest way to find out
whether an upgrade will affect you.

## Currently deprecated

| Name | Use instead | Removed in |
|---|---|---|
| `pylinkage.components.Dyad` | `pylinkage.components.Component` | 2.0.0 |
| `pylinkage.components.ConnectedDyad` | `pylinkage.components.ConnectedComponent` | 2.0.0 |
| `pylinkage.dyads.Dyad` | `pylinkage.components.Component` | 2.0.0 |
| `pylinkage.dyads.ConnectedDyad` | `pylinkage.components.ConnectedComponent` | 2.0.0 |
| `pylinkage.synthesis.Dyad` | `pylinkage.synthesis.BurmesterDyad` | 2.0.0 |
| `pylinkage.dyads.Ground` | `pylinkage.components.Ground` | 2.0.0 |
| `pylinkage.dyads.PointTracker` | `pylinkage.components.PointTracker` | 2.0.0 |
| `pylinkage.dyads.Component` | `pylinkage.components.Component` | 2.0.0 |
| `pylinkage.dyads.ConnectedComponent` | `pylinkage.components.ConnectedComponent` | 2.0.0 |
| `pylinkage.dyads.Crank` | `pylinkage.actuators.Crank` | 2.0.0 |
| `pylinkage.dyads.ArcCrank` | `pylinkage.actuators.ArcCrank` | 2.0.0 |
| `pylinkage.dyads.LinearActuator` | `pylinkage.actuators.LinearActuator` | 2.0.0 |
| `pylinkage.dyads.Linkage` | `pylinkage.simulation.Linkage` | 2.0.0 |
| `pylinkage.optimization.Ensemble` | `pylinkage.population.Ensemble` | 2.0.0 |
| `pylinkage.assur.MobilityResult` | `pylinkage.topology.MobilityInfo` | 2.0.0 |
| `pylinkage.assur.StructuralAnalysis` | `pylinkage.topology.compute_mobility()` | 2.0.0 |
| `path_generation(n_orientation_samples=...)` | `orientation_resolution=` | 2.0.0 |

### Why the `dyads` re-exports are going away

`pylinkage.dyads` used to re-export the frame, actuator and container classes
"for convenience", so `pylinkage.dyads.Ground` and `pylinkage.components.Ground`
were both correct and tutorials disagreed on which to use. Under the one-home
rule above, `dyads` defines dyads. The objects are unchanged.

### Why `MobilityResult` and `StructuralAnalysis` are going away

They were exported from `pylinkage.assur` with no function anywhere in the
package that produced them. The mobility analysis that exists is
`pylinkage.topology.compute_mobility()`, which returns a `MobilityInfo`.

### Why the `Dyad` names are going away

Three unrelated classes were reachable as `Dyad`, and only two of them were
dyads at all:

- `pylinkage.assur.Dyad` — an Assur group. A genuine dyad, and **unaffected**;
  it keeps its name.
- `pylinkage.synthesis.Dyad` — a Burmester dyad: a circle point paired with a
  center point. Also a genuine dyad, but a specific kind, now named
  `BurmesterDyad`.
- `pylinkage.components.Dyad` and `pylinkage.dyads.Dyad` — plain aliases of
  `Component`. Never dyads. A `Ground` point is a `Component`, and calling it a
  dyad is simply wrong.

The practical cost was that documentation cross-references could not tell the
three apart, so a link on one class would take you to another. Since
`pylinkage.dyads` annotated its anchor parameters with the alias, the anchor
type on every `RRRDyad`, `RRPDyad`, `PPDyad` and `FixedDyad` pointed at a class
the code never referred to.

### Why `n_orientation_samples` is going away

It never denoted a number of samples. The value was folded into a per-axis grid
resolution through `max(6, round(n_samples ** (1 / free)))`, so the floor
swallowed it: with four precision points, every value from 6 to 216 produced the
same search. Measured on the README's points, `6`, `12`, `36` and `72` all took
the same time and returned the same ten solutions.

`orientation_resolution` is that per-axis number directly, so the cost model is
readable: the grid holds `orientation_resolution ** (n_points - 1)` candidates.
The default of 6 reproduces exactly what the old default did, and a value passed
to `n_orientation_samples` is translated through the old formula, so neither
changes any result.

Lowering the resolution is rarely a good trade. A coarser grid tends to return
*no* solutions rather than fewer -- on three of six test point sets, shrinking it
took the result from ten solutions to zero. Use `max_solutions` to control cost;
that one is monotone.

### Migrating

Renaming the import is the whole change — the objects are unchanged, and the
aliases still point at exactly what they always did:

```python
# Before
from pylinkage.dyads import Dyad, ConnectedDyad
from pylinkage.synthesis import Dyad

# After
from pylinkage.components import Component, ConnectedComponent
from pylinkage.synthesis import BurmesterDyad
```

```python
# Before
from pylinkage.dyads import Ground, Crank, RRRDyad, Linkage

# After -- one home per name, or the top-level shortcut
from pylinkage.components import Ground
from pylinkage.actuators import Crank
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage

from pylinkage import Ground, Crank, RRRDyad, Linkage  # equivalent
```

```python
# Before
from pylinkage.optimization import Ensemble

# After
from pylinkage.population import Ensemble
```

```python
# Before
path_generation(points, n_orientation_samples=36)

# After -- same search, and the cost model is now visible
path_generation(points, orientation_resolution=6)
```

`isinstance` checks keep working through the transition, because the deprecated
name and its replacement are the same object:

```python
>>> from pylinkage.components import Component
>>> import pylinkage.dyads
>>> pylinkage.dyads.Dyad is Component   # emits DeprecationWarning
True
```
