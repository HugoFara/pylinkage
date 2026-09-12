# Benchmarks

Reproducible performance figures for pylinkage's public API.

Every number here comes from calling the library the way a user would, through
the documented entry points. Nothing on this page measures a prototype, a
hypothetical optimisation, or a reimplementation kept only in the benchmark
suite.

## Reproducing these numbers

```bash
uv run python benchmarks/run_benchmarks.py
```

The harness prints the tables below as Markdown. Add `--json results.json` for
machine-readable output, or `--quick` to check that the harness runs without
waiting for publication-quality repeat counts.

Each measurement is the **median of 7 repeats**, with warmup runs discarded so
that numba's one-off JIT compilation is not charged to the steady-state figures.
The *spread* column is the full range across repeats as a fraction of the
median, and is reported so you can tell a solid number from a noisy one.

## Environment

These figures were collected on a laptop, not dedicated benchmarking hardware.
Treat the ratios as meaningful and the absolute values as indicative.

| Component | Version |
|---|---|
| pylinkage | 1.1.0 |
| python | 3.11.14 |
| platform | Linux-7.1 |
| cpu | AMD Ryzen 7 7840U |
| numpy | 2.4.6 |
| numba | 0.66.0 |
| scipy | 1.17.1 |

## Solver

Simulating the reference four-bar (two grounds, a crank, and an `RRRDyad`).

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `Linkage.step()` | 325,718 | steps/s | 10.8% | pure-Python generator |
| `Linkage.step_fast()` | 1,675,768 | steps/s | 2.0% | numba JIT, compilation excluded by warmup |
| `step_fast()` speedup | 5.1 | x | — | ratio of the two medians above |

`step_fast()` is the numba-backed path and is roughly **5x faster** on this
mechanism. The gap is what makes optimisation practical: an optimiser run spends
essentially all of its time inside the solver.

The ratio was 6.4x in earlier releases. It narrowed because `step()` itself got
about a quarter faster, not because the solver regressed.

Note that `step_fast()` pays JIT compilation on its first call. That cost is
excluded here because it is paid once per process, but it is the reason a single
short simulation may not feel faster than `step()`.

## PSO throughput

Particle swarm optimisation of the same four-bar, with a cheap fitness function,
so the figure reflects the optimiser and solver rather than the objective.

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `particle_swarm_optimization()` | 20,117 | evaluations/s | 72.3% | 100 particles x 100 iterations |
| Full optimisation run | 0.502 | s | 59.1% | ~10,100 fitness evaluations |

Evaluations are counted at the fitness function itself rather than inferred from
`n_particles * iterations`, so the figure stays correct even if the swarm's
internal call pattern changes.

The spread here is much wider than elsewhere -- it is the least trustworthy
number on this page -- because a half-second run is dominated by CPU frequency
scaling. Read the order of magnitude and nothing finer: **on the order of ten
thousand fitness evaluations per second** on a four-bar, meaning a realistic
optimisation budget is governed by how expensive *your* fitness function is.

## Synthesis

Classical synthesis entry points, timed end to end.

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `function_generation()` | 0.07 | ms | 72.6% | 3 angle pairs (Freudenstein); 1 solution |
| `path_generation()` | 677.35 | ms | 0.7% | 4 precision points, `max_solutions=10`; 10 solutions |
| `motion_generation()` | 0.66 | ms | 3.6% | 3 poses; 10 solutions |

The three differ by four orders of magnitude, and the reason is structural
rather than incidental:

- **`function_generation()`** solves Freudenstein's equation directly. For three
  angle pairs that is a small linear system, hence microseconds.
- **`motion_generation()`** applies Burmester theory to a fixed set of poses. The
  work is bounded by the number of poses.
- **`path_generation()`** has no prescribed timing, so coupler orientation at
  each precision point is a free variable. It runs Burmester synthesis once per
  candidate orientation, over a grid that grows exponentially with the number of
  precision points, which is why it costs **hundreds of milliseconds**.

### Where `path_generation()` spends its time

Cost depends strongly on *which* points you ask for, not just how many. Both of
these are four precision points returning ten solutions:

| Precision points | Time |
|---|---:|
| `[(0,0), (1,1), (2,1), (3,0)]` (the README example) | 164 ms |
| `[(0,1), (1,2), (2,1.5), (3,0)]` (benchmarked above) | 685 ms |

Points that admit solutions early are found early, and the search stops. The
table at the top of this section reports the slower of the two, so treat it as
an upper bound rather than a typical figure.

Profiled on the README's points, the time splits as:

| Stage | Share |
|---|---:|
| Burmester synthesis over the orientation grid | 87% |
| Verifying candidates by simulation | 9% |

The search dominates. Verification used to be 67% of the runtime; it now runs
through the numba solver, which is why it no longer does.

**`max_solutions`** (default 10) is the knob that controls the cost, because the
search stops as soon as it has that many confirmed solutions. Measured on the
README's points:

| `max_solutions` | Time | Solutions |
|---|---:|---:|
| 1 | 87 ms | 1 |
| 5 | 129 ms | 5 |
| 10 (default) | 165 ms | 10 |
| 20 | 550 ms | 20 |
| `None` | 1108 ms | 79 |

If you need `path_generation()` faster — in a loop, or behind an interactive
control — lower `max_solutions`. Asking for one solution instead of ten is
roughly half the cost.

```{warning}
Cost grows **exponentially in the number of precision points**, because the
orientation grid is searched over one free angle per point after the first. Five
points can take several seconds and still return nothing. Three or four points
are the practical range, as the docstring's "best results with 3-5 points"
implies more gently than the timings warrant.
```

```{note}
`n_orientation_samples` is deprecated in favour of `orientation_resolution`,
which is the per-axis grid resolution the old argument was silently folded
into. The grid holds `orientation_resolution ** (n_points - 1)` candidates, so
the exponential above is visible in the signature rather than discovered at
runtime. The default of 6 is unchanged, and old calls are translated, so no
result moves. See [Deprecations](deprecations.md).

Lowering the resolution is rarely the trade it looks like: a coarser grid
returns *no* solutions more often than it returns fewer. On three of six test
point sets, shrinking it took the result from ten solutions to zero. The dense
grid earns its cost. Use `max_solutions` to control the time; that one is
monotone.
```

## What is not benchmarked here

The `benchmarks/` directory also contains `benchmark_geometry.py`,
`benchmark_integrated.py`, and `benchmark_optimizations.py`. Those are
development explorations — they compare candidate implementations, some of
which were never shipped — and their output should not be quoted as pylinkage
performance figures. Only `run_benchmarks.py` and `benchmark_solver.py` measure
the library as released.
