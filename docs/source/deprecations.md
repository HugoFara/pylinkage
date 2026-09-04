# Deprecations

Names that still work but are on their way out, with what to use instead.

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
| `path_generation(n_orientation_samples=...)` | `orientation_resolution=` | 2.0.0 |

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
