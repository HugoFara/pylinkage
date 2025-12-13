#!/usr/bin/env python3
"""
Integrated benchmarks showing real-world impact of optimizations.

This script:
1. Simulates what a numba-optimized Revolute joint would look like
2. Compares optimization throughput with different approaches
3. Projects real PSO runtime improvements

Run with: uv run python benchmarks/benchmark_integrated.py
"""
import math
import random
import statistics
import time

from numba import njit

import pylinkage as pl

# ============================================================================
# NUMBA-OPTIMIZED JOINT SIMULATION
# ============================================================================

@njit(cache=True)
def circle_intersect_fast(
    x1: float, y1: float, r1: float,
    x2: float, y2: float, r2: float,
) -> tuple:
    """Fast circle intersection returning (n, x1, y1, x2, y2)."""
    dist_x = x2 - x1
    dist_y = y2 - y1
    distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

    if distance > r1 + r2 or distance < abs(r2 - r1):
        return (0, 0.0, 0.0, 0.0, 0.0)

    mid_dist = (r1 * r1 - r2 * r2 + distance * distance) / (2.0 * distance)
    proj_x = x1 + (mid_dist * dist_x) / distance
    proj_y = y1 + (mid_dist * dist_y) / distance

    height_squared = max(0.0, r1 * r1 - mid_dist * mid_dist)
    if height_squared < 1e-12:  # Tangent case
        return (1, proj_x, proj_y, 0.0, 0.0)

    height = math.sqrt(height_squared) / distance
    return (2, proj_x + height * dist_y, proj_y - height * dist_x,
            proj_x - height * dist_y, proj_y + height * dist_x)


@njit(cache=True)
def cyl_to_cart_fast(radius: float, theta: float, ori_x: float, ori_y: float) -> tuple:
    """Fast polar to cartesian."""
    return (radius * math.cos(theta) + ori_x, radius * math.sin(theta) + ori_y)


@njit(cache=True)
def sqr_dist_fast(x1: float, y1: float, x2: float, y2: float) -> float:
    """Fast squared distance."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


@njit(cache=True)
def simulate_fourbar_step_fast(
    crank_x: float, crank_y: float, crank_r: float, crank_angle: float,
    pin_prev_x: float, pin_prev_y: float,
    static_x: float, static_y: float,
    pin_r0: float, pin_r1: float,
    dt: float
) -> tuple:
    """
    Simulate one step of a four-bar linkage.

    Returns: (crank_new_x, crank_new_y, pin_new_x, pin_new_y, success)
    """
    # Update crank position
    new_angle = crank_angle + dt
    crank_new = cyl_to_cart_fast(crank_r, new_angle, 0.0, 0.0)

    # Solve revolute joint (pin)
    result = circle_intersect_fast(
        crank_new[0], crank_new[1], pin_r0,
        static_x, static_y, pin_r1
    )

    if result[0] == 0:
        return (crank_new[0], crank_new[1], 0.0, 0.0, False)

    if result[0] == 1:
        return (crank_new[0], crank_new[1], result[1], result[2], True)

    # Two intersections - pick nearest to previous position
    d1 = sqr_dist_fast(pin_prev_x, pin_prev_y, result[1], result[2])
    d2 = sqr_dist_fast(pin_prev_x, pin_prev_y, result[3], result[4])

    if d1 < d2:
        return (crank_new[0], crank_new[1], result[1], result[2], True)
    else:
        return (crank_new[0], crank_new[1], result[3], result[4], True)


@njit(cache=True)
def simulate_fourbar_cycle_fast(
    crank_r: float, crank_angle: float,
    static_x: float, static_y: float,
    pin_r0: float, pin_r1: float,
    n_steps: int, dt: float
) -> tuple:
    """
    Simulate a full cycle of a four-bar linkage.

    Returns: (final_crank_x, final_crank_y, final_pin_x, final_pin_y, all_successful)
    """
    # Initial positions
    crank_x, crank_y = cyl_to_cart_fast(crank_r, crank_angle, 0.0, 0.0)

    # Solve initial pin position
    result = circle_intersect_fast(crank_x, crank_y, pin_r0, static_x, static_y, pin_r1)
    if result[0] == 0:
        return (0.0, 0.0, 0.0, 0.0, False)

    pin_x = result[1]
    pin_y = result[2]
    current_angle = crank_angle

    # Simulate steps
    for _ in range(n_steps):
        result = simulate_fourbar_step_fast(
            crank_x, crank_y, crank_r, current_angle,
            pin_x, pin_y,
            static_x, static_y,
            pin_r0, pin_r1,
            dt
        )
        if not result[4]:
            return (result[0], result[1], 0.0, 0.0, False)

        crank_x, crank_y = result[0], result[1]
        pin_x, pin_y = result[2], result[3]
        current_angle += dt

    return (crank_x, crank_y, pin_x, pin_y, True)


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def create_fourbar_linkage():
    """Create a standard four-bar linkage."""
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="B")
    pin = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="C")
    return pl.Linkage(joints=(crank, pin), order=(crank, pin), name="Benchmark")


def benchmark_original_linkage(n_cycles: int = 1000) -> dict:
    """Benchmark original pylinkage implementation."""
    linkage = create_fourbar_linkage()
    period = linkage.get_rotation_period()

    # Warmup
    for _ in range(10):
        list(linkage.step(iterations=period))

    times = []
    for _ in range(n_cycles):
        start = time.perf_counter_ns()
        list(linkage.step(iterations=period))
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "mean_us": statistics.mean(times) / 1000,
        "total_ms": sum(times) / 1_000_000,
        "cycles": n_cycles,
        "steps_per_cycle": period,
        "total_steps": n_cycles * period,
    }


def benchmark_numba_simulation(n_cycles: int = 1000) -> dict:
    """Benchmark numba-optimized simulation."""
    # Parameters matching the original four-bar
    crank_r = 1.0
    crank_angle = 0.31
    static_x, static_y = 3.0, 0.0
    pin_r0, pin_r1 = 3.0, 1.0
    n_steps = 20  # Same as original period
    dt = 2 * math.pi / n_steps

    # Warmup (also triggers JIT compilation)
    for _ in range(10):
        simulate_fourbar_cycle_fast(crank_r, crank_angle, static_x, static_y, pin_r0, pin_r1, n_steps, dt)

    times = []
    for _ in range(n_cycles):
        start = time.perf_counter_ns()
        simulate_fourbar_cycle_fast(crank_r, crank_angle, static_x, static_y, pin_r0, pin_r1, n_steps, dt)
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "mean_us": statistics.mean(times) / 1000,
        "total_ms": sum(times) / 1_000_000,
        "cycles": n_cycles,
        "steps_per_cycle": n_steps,
        "total_steps": n_cycles * n_steps,
    }


def benchmark_optimization_scenario(n_evals: int = 1000) -> dict:
    """Benchmark a realistic optimization scenario with original code."""
    linkage = create_fourbar_linkage()
    period = linkage.get_rotation_period()
    init_constraints = tuple(linkage.get_num_constraints())
    init_coords = linkage.get_coords()

    # Generate random constraint variations
    random.seed(42)
    variations = [
        tuple(c * random.uniform(0.8, 1.2) for c in init_constraints)
        for _ in range(n_evals)
    ]

    # Warmup
    for v in variations[:10]:
        linkage.set_num_constraints(v)
        linkage.set_coords(init_coords)
        try:
            list(linkage.step(iterations=period))
        except Exception:
            pass

    linkage.set_num_constraints(init_constraints)
    linkage.set_coords(init_coords)

    # Benchmark
    start = time.perf_counter()
    successful = 0
    for v in variations:
        linkage.set_num_constraints(v)
        linkage.set_coords(init_coords)
        try:
            list(linkage.step(iterations=period))
            successful += 1
        except Exception:
            pass
    total_time = time.perf_counter() - start

    return {
        "total_sec": total_time,
        "evals": n_evals,
        "successful": successful,
        "evals_per_sec": n_evals / total_time,
        "mean_eval_us": total_time * 1_000_000 / n_evals,
    }


def benchmark_numba_optimization_scenario(n_evals: int = 1000) -> dict:
    """Benchmark optimization scenario with numba-optimized code."""
    # Base parameters
    crank_r = 1.0
    crank_angle = 0.31
    static_x, static_y = 3.0, 0.0
    base_pin_r0, base_pin_r1 = 3.0, 1.0
    n_steps = 20
    dt = 2 * math.pi / n_steps

    # Generate variations
    random.seed(42)
    variations = [
        (
            base_pin_r0 * random.uniform(0.8, 1.2),
            base_pin_r1 * random.uniform(0.8, 1.2),
        )
        for _ in range(n_evals)
    ]

    # Warmup
    for pin_r0, pin_r1 in variations[:10]:
        simulate_fourbar_cycle_fast(crank_r, crank_angle, static_x, static_y, pin_r0, pin_r1, n_steps, dt)

    # Benchmark
    start = time.perf_counter()
    successful = 0
    for pin_r0, pin_r1 in variations:
        result = simulate_fourbar_cycle_fast(crank_r, crank_angle, static_x, static_y, pin_r0, pin_r1, n_steps, dt)
        if result[4]:
            successful += 1
    total_time = time.perf_counter() - start

    return {
        "total_sec": total_time,
        "evals": n_evals,
        "successful": successful,
        "evals_per_sec": n_evals / total_time,
        "mean_eval_us": total_time * 1_000_000 / n_evals,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("INTEGRATED OPTIMIZATION BENCHMARKS")
    print("=" * 70)
    print()

    # Force numba compilation
    print("Compiling Numba functions...")
    _ = simulate_fourbar_cycle_fast(1.0, 0.31, 3.0, 0.0, 3.0, 1.0, 20, 0.314)
    print("Done.")
    print()

    # Single cycle comparison
    print("-" * 70)
    print("SINGLE LINKAGE CYCLE COMPARISON (1000 cycles)")
    print("-" * 70)
    print()

    orig_cycle = benchmark_original_linkage(n_cycles=1000)
    numba_cycle = benchmark_numba_simulation(n_cycles=1000)

    print("Original pylinkage:")
    print(f"  Mean cycle time: {orig_cycle['mean_us']:.2f} µs")
    print(f"  Total time: {orig_cycle['total_ms']:.1f} ms")
    print()
    print("Numba-optimized:")
    print(f"  Mean cycle time: {numba_cycle['mean_us']:.2f} µs")
    print(f"  Total time: {numba_cycle['total_ms']:.1f} ms")
    print()
    cycle_speedup = orig_cycle['mean_us'] / numba_cycle['mean_us']
    print(f"Speedup: {cycle_speedup:.1f}x")
    print()

    # Optimization scenario comparison
    print("-" * 70)
    print("OPTIMIZATION SCENARIO COMPARISON (1000 evaluations)")
    print("-" * 70)
    print()

    orig_opt = benchmark_optimization_scenario(n_evals=1000)
    numba_opt = benchmark_numba_optimization_scenario(n_evals=1000)

    print("Original pylinkage:")
    print(f"  Total time: {orig_opt['total_sec']:.3f} sec")
    print(f"  Mean eval time: {orig_opt['mean_eval_us']:.1f} µs")
    print(f"  Throughput: {orig_opt['evals_per_sec']:,.0f} evals/sec")
    print(f"  Successful: {orig_opt['successful']}/{orig_opt['evals']}")
    print()
    print("Numba-optimized:")
    print(f"  Total time: {numba_opt['total_sec']:.3f} sec")
    print(f"  Mean eval time: {numba_opt['mean_eval_us']:.1f} µs")
    print(f"  Throughput: {numba_opt['evals_per_sec']:,.0f} evals/sec")
    print(f"  Successful: {numba_opt['successful']}/{numba_opt['evals']}")
    print()
    opt_speedup = orig_opt['total_sec'] / numba_opt['total_sec']
    print(f"Speedup: {opt_speedup:.1f}x")
    print()

    # PSO projection
    print("-" * 70)
    print("PROJECTED PSO RUNTIME")
    print("-" * 70)
    print()

    pso_configs = [
        (100, 30, "Small (100 iter × 30 particles)"),
        (200, 50, "Medium (200 iter × 50 particles)"),
        (500, 100, "Large (500 iter × 100 particles)"),
    ]

    for n_iter, n_particles, name in pso_configs:
        total_evals = n_iter * n_particles
        orig_time = total_evals * orig_opt['mean_eval_us'] / 1_000_000
        numba_time = total_evals * numba_opt['mean_eval_us'] / 1_000_000
        print(f"{name}:")
        print(f"  Total evaluations: {total_evals:,}")
        print(f"  Original: {orig_time:.2f} sec")
        print(f"  Numba: {numba_time:.2f} sec")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Single cycle speedup:     {cycle_speedup:.1f}x")
    print(f"Optimization speedup:     {opt_speedup:.1f}x")
    print()
    print("Key findings:")
    print(f"  - Numba provides {cycle_speedup:.1f}x speedup for linkage simulation")
    print(f"  - For PSO optimization, expect {opt_speedup:.1f}x faster convergence")
    print()
    print("Integration path:")
    print("  1. Create numba-optimized geometry functions (secants_fast.py)")
    print("  2. Create FastRevolute joint class using numba functions")
    print("  3. Use FastLinkage wrapper for optimization hot path")
    print("  4. Keep original classes for flexibility and debugging")


if __name__ == "__main__":
    main()
