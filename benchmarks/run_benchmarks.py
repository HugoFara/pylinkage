#!/usr/bin/env python3
"""Reproducible benchmarks for the public pylinkage API.

This is the harness behind ``docs/benchmarks.md``. Every figure it reports comes
from calling the public API the way a user would, so the numbers describe
shipped code rather than a prototype or a local reimplementation.

Three suites are measured:

1. **Solver** — ``Linkage.step()`` against the numba-backed ``step_fast()``.
2. **PSO throughput** — fitness evaluations per second through
   ``particle_swarm_optimization()``, counted at the evaluation function itself.
3. **Synthesis** — wall-clock time for ``function_generation()``,
   ``path_generation()`` and ``motion_generation()``.

Run with::

    uv run python benchmarks/run_benchmarks.py

Add ``--json results.json`` to save machine-readable output, and ``--quick`` to
run with reduced repeat counts when smoke-testing the harness itself.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.optimization.particle_swarm import particle_swarm_optimization
from pylinkage.optimization.utils import kinematic_minimization
from pylinkage.simulation import Linkage
from pylinkage.synthesis import (
    Pose,
    function_generation,
    motion_generation,
    path_generation,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Measurement:
    """One benchmarked quantity, summarised across repeats."""

    name: str
    unit: str
    median: float
    minimum: float
    maximum: float
    repeats: int
    note: str = ""

    @property
    def spread(self) -> float:
        """Relative spread across repeats, as a fraction of the median."""
        if self.median == 0:
            return 0.0
        return (self.maximum - self.minimum) / self.median


@dataclass
class Suite:
    """A named group of measurements."""

    name: str
    measurements: list[Measurement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def summarise(
    name: str,
    unit: str,
    samples: list[float],
    note: str = "",
) -> Measurement:
    """Reduce raw per-repeat samples to a reportable measurement."""
    return Measurement(
        name=name,
        unit=unit,
        median=statistics.median(samples),
        minimum=min(samples),
        maximum=max(samples),
        repeats=len(samples),
        note=note,
    )


def time_call(func: Callable[[], Any], repeats: int, warmup: int = 1) -> list[float]:
    """Time a zero-argument callable, returning seconds per repeat.

    The warmup runs are discarded, which matters for anything numba-backed:
    the first call pays JIT compilation that a user only pays once.
    """
    for _ in range(warmup):
        func()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)
    return samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fourbar() -> Linkage:
    """Build the reference four-bar used across the suites."""
    ground1 = Ground(0.0, 0.0, name="O1")
    ground2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=ground1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=ground2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([ground1, ground2, crank, rocker], name="four-bar")


@kinematic_minimization
def _tip_distance(loci: Any, **_: Any) -> float:
    """Cheap, well-behaved fitness: distance of each final locus from origin."""
    tip = [locus[-1] for locus in loci]
    return sum(abs(point[0]) + abs(point[1]) for point in tip)


# ---------------------------------------------------------------------------
# Suite 1: solver
# ---------------------------------------------------------------------------


def bench_solver(repeats: int, batches: int = 500, steps: int = 10) -> Suite:
    """Compare the generator solver against the numba-backed one."""
    linkage = make_fourbar()
    total_steps = batches * steps

    def run_step() -> None:
        for _ in range(batches):
            list(linkage.step(iterations=steps, dt=1.0))

    def run_step_fast() -> None:
        for _ in range(batches):
            linkage.step_fast(iterations=steps, dt=1.0)

    step_times = time_call(run_step, repeats=repeats, warmup=1)
    fast_times = time_call(run_step_fast, repeats=repeats, warmup=2)

    step_rates = [total_steps / t for t in step_times]
    fast_rates = [total_steps / t for t in fast_times]

    suite = Suite("Solver")
    suite.measurements.append(
        summarise("Linkage.step()", "steps/s", step_rates, "pure-Python generator")
    )
    suite.measurements.append(
        summarise(
            "Linkage.step_fast()",
            "steps/s",
            fast_rates,
            "numba JIT, compilation excluded by warmup",
        )
    )
    speedup = statistics.median(fast_rates) / statistics.median(step_rates)
    suite.measurements.append(
        Measurement(
            name="step_fast() speedup",
            unit="x",
            median=speedup,
            minimum=speedup,
            maximum=speedup,
            repeats=repeats,
            note="ratio of the two medians above",
        )
    )
    return suite


# ---------------------------------------------------------------------------
# Suite 2: PSO throughput
# ---------------------------------------------------------------------------


def bench_pso(repeats: int, n_particles: int = 100, iterations: int = 100) -> Suite:
    """Measure fitness evaluations per second through the PSO driver.

    Evaluations are counted at the evaluation function rather than derived from
    ``n_particles * iterations``, so the figure stays honest if the swarm's
    internal call pattern ever changes.
    """
    calls = 0

    def counting_fitness(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        return _tip_distance(*args, **kwargs)

    def run_pso() -> None:
        particle_swarm_optimization(
            eval_func=counting_fitness,
            linkage=make_fourbar(),
            n_particles=n_particles,
            iterations=iterations,
            neighbors=min(17, n_particles),
            order_relation=min,
            verbose=False,
        )

    times = time_call(run_pso, repeats=repeats, warmup=1)

    # calls accumulated over warmup + every repeat; per-run is the fair unit.
    calls_per_run = calls / (repeats + 1)
    rates = [calls_per_run / t for t in times]

    suite = Suite("PSO throughput")
    suite.measurements.append(
        summarise(
            "particle_swarm_optimization()",
            "evaluations/s",
            rates,
            f"{n_particles} particles x {iterations} iterations, four-bar",
        )
    )
    suite.measurements.append(
        summarise(
            "Full optimisation run",
            "s",
            times,
            f"~{calls_per_run:,.0f} fitness evaluations",
        )
    )
    return suite


# ---------------------------------------------------------------------------
# Suite 3: synthesis
# ---------------------------------------------------------------------------


def bench_synthesis(repeats: int) -> Suite:
    """Time each of the three classical synthesis entry points."""
    # These pairs yield a Grashof solution. Several "obvious" choices --
    # including the one in the pylinkage.synthesis docstring -- are rejected and
    # return zero solutions, which would time the rejection path instead.
    angle_pairs = [
        (0.0, 0.0),
        (1.0, 0.6),
        (2.0, 1.1),
    ]
    precision_points = [(0.0, 1.0), (1.0, 2.0), (2.0, 1.5), (3.0, 0.0)]
    poses = [
        Pose(0.0, 0.0, 0.0),
        Pose(1.0, 1.0, math.pi / 6),
        Pose(2.0, 1.0, math.pi / 3),
    ]

    cases: list[tuple[str, Callable[[], Any], str]] = [
        (
            "function_generation()",
            lambda: function_generation(angle_pairs),
            f"{len(angle_pairs)} angle pairs (Freudenstein)",
        ),
        (
            "path_generation()",
            lambda: path_generation(precision_points),
            f"{len(precision_points)} precision points, 36 orientation samples",
        ),
        (
            "motion_generation()",
            lambda: motion_generation(poses),
            f"{len(poses)} poses",
        ),
    ]

    suite = Suite("Synthesis")
    for name, call, note in cases:
        result = call()
        n_solutions = len(getattr(result, "solutions", []))
        times = time_call(call, repeats=repeats, warmup=1)
        suite.measurements.append(
            summarise(
                name,
                "ms",
                [t * 1000 for t in times],
                f"{note}; {n_solutions} solution(s)",
            )
        )
    return suite


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    """Best-effort CPU model name, since platform.processor() is often empty."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _version(package: str) -> str:
    """Installed version of a package, or "not installed"."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def describe_environment() -> dict[str, str]:
    """Capture everything needed to interpret or reproduce the numbers."""
    return {
        "pylinkage": _version("pylinkage"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "numpy": _version("numpy"),
        "numba": _version("numba"),
        "scipy": _version("scipy"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_value(value: float, unit: str) -> str:
    """Render a value with a sensible precision for its unit."""
    if unit in {"steps/s", "evaluations/s"}:
        return f"{value:,.0f}"
    if unit == "x":
        return f"{value:.1f}"
    if unit == "ms":
        return f"{value:.2f}"
    return f"{value:.3f}"


def render_markdown(env: dict[str, str], suites: list[Suite]) -> Iterator[str]:
    """Yield the report as Markdown lines."""
    yield "## Environment"
    yield ""
    yield "| Component | Version |"
    yield "|---|---|"
    for key, value in env.items():
        yield f"| {key} | {value} |"
    yield ""

    for suite in suites:
        yield f"## {suite.name}"
        yield ""
        yield "| Measurement | Median | Unit | Spread | Notes |"
        yield "|---|---:|---|---:|---|"
        for m in suite.measurements:
            yield (
                f"| `{m.name}` | {format_value(m.median, m.unit)} | {m.unit} "
                f"| {m.spread:.1%} | {m.note} |"
            )
        yield ""


def main() -> int:
    """Run every suite and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=None, help="also write results to this JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fewer repeats, for checking the harness rather than publishing numbers",
    )
    args = parser.parse_args()

    repeats = 3 if args.quick else 7

    env = describe_environment()
    suites = [
        bench_solver(repeats=repeats, batches=100 if args.quick else 500),
        bench_pso(repeats=repeats),
        bench_synthesis(repeats=repeats),
    ]

    report = "\n".join(render_markdown(env, suites))
    print(report)

    if args.json is not None:
        payload = {
            "environment": env,
            "suites": [
                {
                    "name": suite.name,
                    "measurements": [asdict(m) for m in suite.measurements],
                }
                for suite in suites
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
