"""Benchmark comparing step() vs step_fast() performance."""

import time

import pylinkage as pl


def create_fourbar_linkage():
    """Create a four-bar linkage for benchmarking."""
    ground1 = pl.Static(0, 0, name="ground1")
    ground2 = pl.Static(3, 0, name="ground2")
    crank = pl.Crank(
        1, 0,
        joint0=ground1,
        distance=1,
        angle=0.1,
        name="crank"
    )
    pin = pl.Revolute(
        2, 1,
        joint0=crank,
        joint1=ground2,
        distance0=2,
        distance1=2,
        name="pin"
    )
    return pl.Linkage(
        joints=(ground1, ground2, crank, pin),
        order=(ground1, ground2, crank, pin),
        name="four-bar"
    )


def benchmark_step(linkage: pl.Linkage, iterations: int = 1000, warmup: int = 100):
    """Benchmark the step() generator method."""
    # Warmup
    for _ in range(warmup):
        list(linkage.step(iterations=10, dt=1.0))

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        list(linkage.step(iterations=10, dt=1.0))
    end = time.perf_counter()

    total_steps = iterations * 10
    elapsed = end - start
    return total_steps / elapsed


def benchmark_step_fast(linkage: pl.Linkage, iterations: int = 1000, warmup: int = 100):
    """Benchmark the step_fast() method."""
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        linkage.step_fast(iterations=10, dt=1.0)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        linkage.step_fast(iterations=10, dt=1.0)
    end = time.perf_counter()

    total_steps = iterations * 10
    elapsed = end - start
    return total_steps / elapsed


def main():
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
