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
| platform | Linux-7.1.12-200.fc44.x86_64-x86_64-with-glibc2.43 |
| cpu | AMD Ryzen 7 7840U w/ Radeon 780M Graphics |
| numpy | 2.4.6 |
| numba | 0.66.0 |
| scipy | 1.17.1 |

## Solver

Simulating the reference four-bar (two grounds, a crank, and an `RRRDyad`).

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `Linkage.step()` | 257,550 | steps/s | 9.8% | pure-Python generator |
| `Linkage.step_fast()` | 1,658,608 | steps/s | 4.6% | numba JIT, compilation excluded by warmup |
| `step_fast()` speedup | 6.4 | x | — | ratio of the two medians above |

`step_fast()` is the numba-backed path and is roughly **6x faster** on this
mechanism. The gap is what makes optimisation practical: an optimiser run spends
essentially all of its time inside the solver.

Note that `step_fast()` pays JIT compilation on its first call. That cost is
excluded here because it is paid once per process, but it is the reason a single
short simulation may not feel faster than `step()`.

## PSO throughput

Particle swarm optimisation of the same four-bar, with a cheap fitness function,
so the figure reflects the optimiser and solver rather than the objective.

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `particle_swarm_optimization()` | 15,814 | evaluations/s | 33.2% | 100 particles x 100 iterations |
| Full optimisation run | 0.639 | s | 29.6% | ~10,100 fitness evaluations |

Evaluations are counted at the fitness function itself rather than inferred from
`n_particles * iterations`, so the figure stays correct even if the swarm's
internal call pattern changes.

The spread here is wider than elsewhere because a sub-second run is sensitive to
CPU frequency scaling. The useful takeaway is the order of magnitude: **roughly
ten thousand fitness evaluations per second** on a four-bar, meaning a realistic
optimisation budget is dominated by how expensive *your* fitness function is.

## Synthesis

Classical synthesis entry points, timed end to end.

| Measurement | Median | Unit | Spread | Notes |
|---|---:|---|---:|---|
| `function_generation()` | 0.09 | ms | 27.8% | 3 angle pairs (Freudenstein); 1 solution |
| `path_generation()` | 1778.63 | ms | 8.2% | 4 precision points, 36 orientation samples; 10 solutions |
| `motion_generation()` | 0.69 | ms | 6.3% | 3 poses; 10 solutions |

The three differ by four orders of magnitude, and the reason is structural
rather than incidental:

- **`function_generation()`** solves Freudenstein's equation directly. For three
  angle pairs that is a small linear system, hence microseconds.
- **`motion_generation()`** applies Burmester theory to a fixed set of poses. The
  work is bounded by the number of poses.
- **`path_generation()`** has no prescribed timing, so coupler orientation at
  each precision point is a free variable. It searches over
  `n_orientation_samples` candidate orientations (36 by default) and runs
  Burmester synthesis for each, which is why it costs **well over a second**.

That last figure is worth knowing before you call `path_generation()` in a loop
or behind an interactive control. If you need it faster, lowering
`n_orientation_samples` trades solution coverage for time roughly linearly.

## What is not benchmarked here

The `benchmarks/` directory also contains `benchmark_geometry.py`,
`benchmark_integrated.py`, and `benchmark_optimizations.py`. Those are
development explorations — they compare candidate implementations, some of
which were never shipped — and their output should not be quoted as pylinkage
performance figures. Only `run_benchmarks.py` and `benchmark_solver.py` measure
the library as released.
