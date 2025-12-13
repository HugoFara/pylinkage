#!/usr/bin/env python3
"""
Benchmarks for pylinkage geometry solvers.

This script measures performance of:
1. Individual geometry functions (circle_intersect, cyl_to_cart, etc.)
2. Full linkage simulation
3. Optimization (PSO-like evaluation loop)

Run with: uv run python benchmarks/benchmark_geometry.py
"""
import math
import random
import statistics
import time
from collections.abc import Callable

# For linkage benchmarks
import pylinkage as pl

# Current implementations
from pylinkage.geometry.core import (
    cyl_to_cart,
    get_nearest_point,
    norm,
    sqr_dist,
)
from pylinkage.geometry.secants import (
    circle_intersect,
    circle_line_from_points_intersection,
)


def benchmark(func: Callable, args_generator: Callable, n_iterations: int = 100_000, warmup: int = 1000) -> dict:
    """
    Benchmark a function with generated arguments.

    Args:
        func: Function to benchmark
        args_generator: Function that returns (args, kwargs) tuple
        n_iterations: Number of iterations
        warmup: Warmup iterations (not counted)

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        args, kwargs = args_generator()
        func(*args, **kwargs)

    # Actual benchmark
    times = []
    for _ in range(n_iterations):
        args, kwargs = args_generator()
        start = time.perf_counter_ns()
        func(*args, **kwargs)
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "stdev_ns": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ns": min(times),
        "max_ns": max(times),
        "total_ms": sum(times) / 1_000_000,
        "iterations": n_iterations,
    }


def format_results(name: str, results: dict) -> str:
    """Format benchmark results for display."""
    return (
        f"{name}:\n"
        f"  Mean: {results['mean_ns']:.1f} ns ({results['mean_ns']/1000:.2f} µs)\n"
        f"  Median: {results['median_ns']:.1f} ns\n"
        f"  Stdev: {results['stdev_ns']:.1f} ns\n"
        f"  Min/Max: {results['min_ns']:.1f} / {results['max_ns']:.1f} ns\n"
        f"  Total: {results['total_ms']:.2f} ms for {results['iterations']:,} iterations\n"
    )


# ============================================================================
# Argument generators for different functions (scalar-based API)
# ============================================================================

def gen_sqr_dist_args():
    """Generate random points for sqr_dist."""
    x1 = random.uniform(-100, 100)
    y1 = random.uniform(-100, 100)
    x2 = random.uniform(-100, 100)
    y2 = random.uniform(-100, 100)
    return (x1, y1, x2, y2), {}


def gen_nearest_point_args():
    """Generate random points for get_nearest_point."""
    ref_x = random.uniform(-100, 100)
    ref_y = random.uniform(-100, 100)
    p1_x = random.uniform(-100, 100)
    p1_y = random.uniform(-100, 100)
    p2_x = random.uniform(-100, 100)
    p2_y = random.uniform(-100, 100)
    return (ref_x, ref_y, p1_x, p1_y, p2_x, p2_y), {}


def gen_norm_args():
    """Generate random vector for norm."""
    x = random.uniform(-100, 100)
    y = random.uniform(-100, 100)
    return (x, y), {}


def gen_cyl_to_cart_args():
    """Generate random polar coordinates."""
    r = random.uniform(0.1, 100)
    theta = random.uniform(0, 2 * math.pi)
    ori_x = random.uniform(-50, 50)
    ori_y = random.uniform(-50, 50)
    return (r, theta, ori_x, ori_y), {}


def gen_circle_intersect_args():
    """Generate two intersecting circles."""
    # Create circles that are likely to intersect
    x1, y1 = random.uniform(-50, 50), random.uniform(-50, 50)
    r1 = random.uniform(1, 10)
    # Second circle center within reasonable distance
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0.5, r1 + 5)
    x2 = x1 + dist * math.cos(angle)
    y2 = y1 + dist * math.sin(angle)
    r2 = random.uniform(1, 10)
    return (x1, y1, r1, x2, y2, r2), {}


def gen_circle_line_args():
    """Generate circle and line that likely intersect."""
    cx, cy = random.uniform(-50, 50), random.uniform(-50, 50)
    r = random.uniform(1, 10)
    # Line passing near the circle
    p1_x = cx + random.uniform(-r, r)
    p1_y = cy + random.uniform(-20, 20)
    p2_x = cx + random.uniform(-r, r)
    p2_y = cy + random.uniform(-20, 20)
    return (cx, cy, r, p1_x, p1_y, p2_x, p2_y), {}


# ============================================================================
# Linkage benchmarks
# ============================================================================

def create_fourbar_linkage():
    """Create a standard four-bar linkage for benchmarking."""
    crank = pl.Crank(
        0, 1,
        joint0=(0, 0),
        angle=0.31, distance=1,
        name="B"
    )
    pin = pl.Revolute(
        3, 2,
        joint0=crank, joint1=(3, 0),
        distance0=3, distance1=1, name="C"
    )
    return pl.Linkage(
        joints=(crank, pin),
        order=(crank, pin),
        name="Benchmark four-bar"
    )


def benchmark_linkage_step(linkage, n_cycles: int = 1000) -> dict:
    """Benchmark the linkage.step() simulation."""
    # Get rotation period
    period = linkage.get_rotation_period()

    # Warmup
    for _ in range(10):
        list(linkage.step(iterations=period))

    # Benchmark
    times = []
    for _ in range(n_cycles):
        start = time.perf_counter_ns()
        coords = list(linkage.step(iterations=period))
        end = time.perf_counter_ns()
        times.append(end - start)

    total_steps = n_cycles * period
    return {
        "mean_cycle_us": statistics.mean(times) / 1000,
        "median_cycle_us": statistics.median(times) / 1000,
        "stdev_cycle_us": statistics.stdev(times) / 1000 if len(times) > 1 else 0,
        "total_ms": sum(times) / 1_000_000,
        "cycles": n_cycles,
        "steps_per_cycle": period,
        "total_steps": total_steps,
        "steps_per_second": total_steps / (sum(times) / 1_000_000_000),
    }


def benchmark_optimization_loop(linkage, n_evaluations: int = 1000) -> dict:
    """Benchmark a typical optimization evaluation loop."""
    period = linkage.get_rotation_period()
    init_constraints = tuple(linkage.get_num_constraints())
    init_coords = linkage.get_coords()

    # Generate random constraint variations
    variations = []
    for _ in range(n_evaluations):
        # Small random variations around initial constraints
        variation = tuple(c * random.uniform(0.8, 1.2) for c in init_constraints)
        variations.append(variation)

    # Warmup
    for i in range(min(10, n_evaluations)):
        linkage.set_num_constraints(variations[i])
        linkage.set_coords(init_coords)
        try:
            list(linkage.step(iterations=period))
        except Exception:
            pass

    # Reset
    linkage.set_num_constraints(init_constraints)
    linkage.set_coords(init_coords)

    # Benchmark
    times = []
    successful = 0
    for variation in variations:
        start = time.perf_counter_ns()
        linkage.set_num_constraints(variation)
        linkage.set_coords(init_coords)
        try:
            coords = list(linkage.step(iterations=period))
            successful += 1
        except Exception:
            pass  # Unbuildable configuration
        end = time.perf_counter_ns()
        times.append(end - start)

    # Reset linkage
    linkage.set_num_constraints(init_constraints)
    linkage.set_coords(init_coords)

    return {
        "mean_eval_us": statistics.mean(times) / 1000,
        "median_eval_us": statistics.median(times) / 1000,
        "total_ms": sum(times) / 1_000_000,
        "evaluations": n_evaluations,
        "successful": successful,
        "evals_per_second": n_evaluations / (sum(times) / 1_000_000_000),
    }


# ============================================================================
# Main benchmark runner
# ============================================================================

def run_geometry_benchmarks(n_iterations: int = 100_000):
    """Run all geometry function benchmarks."""
    print("=" * 70)
    print("GEOMETRY FUNCTION BENCHMARKS (with numba)")
    print(f"({n_iterations:,} iterations each)")
    print("=" * 70)
    print()

    benchmarks = [
        ("sqr_dist", sqr_dist, gen_sqr_dist_args),
        ("get_nearest_point", get_nearest_point, gen_nearest_point_args),
        ("norm", norm, gen_norm_args),
        ("cyl_to_cart", cyl_to_cart, gen_cyl_to_cart_args),
        ("circle_intersect", circle_intersect, gen_circle_intersect_args),
        ("circle_line_from_points_intersection", circle_line_from_points_intersection, gen_circle_line_args),
    ]

    results = {}
    for name, func, gen in benchmarks:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        result = benchmark(func, gen, n_iterations=n_iterations)
        results[name] = result
        print("done")
        print(format_results(name, result))

    return results


def run_linkage_benchmarks(n_cycles: int = 1000, n_evals: int = 500):
    """Run linkage simulation benchmarks."""
    print("=" * 70)
    print("LINKAGE SIMULATION BENCHMARKS (with numba)")
    print("=" * 70)
    print()

    linkage = create_fourbar_linkage()

    print(f"Benchmarking linkage.step() ({n_cycles} cycles)...", end=" ", flush=True)
    step_results = benchmark_linkage_step(linkage, n_cycles=n_cycles)
    print("done")
    print("linkage.step() (full cycle):")
    print(f"  Mean cycle time: {step_results['mean_cycle_us']:.1f} µs")
    print(f"  Median cycle time: {step_results['median_cycle_us']:.1f} µs")
    print(f"  Steps per cycle: {step_results['steps_per_cycle']}")
    print(f"  Total steps: {step_results['total_steps']:,}")
    print(f"  Throughput: {step_results['steps_per_second']:,.0f} steps/sec")
    print()

    print(f"Benchmarking optimization loop ({n_evals} evaluations)...", end=" ", flush=True)
    opt_results = benchmark_optimization_loop(linkage, n_evaluations=n_evals)
    print("done")
    print("Optimization evaluation loop:")
    print(f"  Mean evaluation time: {opt_results['mean_eval_us']:.1f} µs")
    print(f"  Median evaluation time: {opt_results['median_eval_us']:.1f} µs")
    print(f"  Successful builds: {opt_results['successful']}/{opt_results['evaluations']}")
    print(f"  Throughput: {opt_results['evals_per_second']:,.0f} evals/sec")
    print()

    return {"step": step_results, "optimization": opt_results}


def main():
    """Run all benchmarks."""
    random.seed(42)  # Reproducibility

    print()
    print("PYLINKAGE SOLVER BENCHMARKS (NUMBA OPTIMIZED)")
    print("=" * 70)
    print()

    # Geometry benchmarks
    geometry_results = run_geometry_benchmarks(n_iterations=100_000)

    # Linkage benchmarks
    linkage_results = run_linkage_benchmarks(n_cycles=1000, n_evals=500)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Geometry functions (mean time):")
    for name, result in geometry_results.items():
        print(f"  {name}: {result['mean_ns']:.0f} ns ({result['mean_ns']/1000:.2f} µs)")
    print()
    print("Linkage simulation:")
    print(f"  Step cycle: {linkage_results['step']['mean_cycle_us']:.0f} µs")
    print(f"  Optimization eval: {linkage_results['optimization']['mean_eval_us']:.0f} µs")
    print(f"  Steps/sec: {linkage_results['step']['steps_per_second']:,.0f}")
    print(f"  Evals/sec: {linkage_results['optimization']['evals_per_second']:,.0f}")
    print()

    # Estimated PSO time
    pso_iterations = 200
    particles = 50
    total_evals = pso_iterations * particles
    est_pso_time = total_evals * linkage_results['optimization']['mean_eval_us'] / 1_000_000
    print(f"Estimated PSO time ({pso_iterations} iters × {particles} particles = {total_evals:,} evals):")
    print(f"  ~{est_pso_time:.1f} seconds")


if __name__ == "__main__":
    main()
