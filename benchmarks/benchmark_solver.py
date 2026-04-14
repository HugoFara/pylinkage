"""Benchmark comparing step() vs step_fast() performance."""

import time

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage


def create_fourbar_linkage() -> Linkage:
    """Create a four-bar linkage for benchmarking."""
    ground1 = Ground(0.0, 0.0, name="ground1")
    ground2 = Ground(3.0, 0.0, name="ground2")
    crank = Crank(
        anchor=ground1, radius=1.0, angular_velocity=0.1, name="crank",
    )
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=ground2,
        distance1=2.0,
        distance2=2.0,
        name="pin",
    )
    return Linkage([ground1, ground2, crank, pin], name="four-bar")


def benchmark_step(linkage: Linkage, iterations: int = 1000, warmup: int = 100) -> float:
    """Benchmark the step() generator method."""
    for _ in range(warmup):
        list(linkage.step(iterations=10, dt=1.0))

    start = time.perf_counter()
    for _ in range(iterations):
        list(linkage.step(iterations=10, dt=1.0))
    end = time.perf_counter()

    total_steps = iterations * 10
    elapsed = end - start
    return total_steps / elapsed


def benchmark_step_fast(
    linkage: Linkage, iterations: int = 1000, warmup: int = 100,
) -> float:
    """Benchmark the step_fast() method."""
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        linkage.step_fast(iterations=10, dt=1.0)

    start = time.perf_counter()
    for _ in range(iterations):
        linkage.step_fast(iterations=10, dt=1.0)
    end = time.perf_counter()

    total_steps = iterations * 10
    elapsed = end - start
    return total_steps / elapsed


def main() -> None:
    """Run the benchmark."""
    print("=" * 60)
    print("Pylinkage Solver Benchmark: step() vs step_fast()")
    print("=" * 60)
    print()

    linkage = create_fourbar_linkage()

    print("Benchmarking step() (generator-based)...")
    step_rate = benchmark_step(linkage, iterations=500, warmup=50)
    print(f"  step() rate: {step_rate:,.0f} steps/sec")
    print()

    print("Benchmarking step_fast() (numba-optimized)...")
    fast_rate = benchmark_step_fast(linkage, iterations=500, warmup=50)
    print(f"  step_fast() rate: {fast_rate:,.0f} steps/sec")
    print()

    speedup = fast_rate / step_rate
    print("-" * 60)
    print(f"Speedup: {speedup:.1f}x faster with step_fast()")
    print("=" * 60)


if __name__ == "__main__":
    main()
