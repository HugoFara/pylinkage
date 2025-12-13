#!/usr/bin/env python3
"""
Benchmarks comparing original vs optimized geometry solvers.

This script measures performance improvements from:
1. Numba JIT compilation
2. NumPy vectorization

Run with: uv run python benchmarks/benchmark_optimizations.py
"""
import math
import random
import statistics
import time
from collections.abc import Callable

import numpy as np
from numba import njit

# For linkage benchmarks
from pylinkage.geometry.core import (
    cyl_to_cart as cyl_to_cart_orig,
)
from pylinkage.geometry.core import (
    get_nearest_point as get_nearest_point_orig,
)
from pylinkage.geometry.core import (
    norm as norm_orig,
)

# Current implementations
from pylinkage.geometry.core import (
    sqr_dist as sqr_dist_orig,
)
from pylinkage.geometry.secants import (
    circle_intersect as circle_intersect_orig,
)
from pylinkage.geometry.secants import (
    circle_line_from_points_intersection as circle_line_orig,
)

# ============================================================================
# NUMBA OPTIMIZED IMPLEMENTATIONS
# ============================================================================

@njit(cache=True)
def sqr_dist_numba(p1_x: float, p1_y: float, p2_x: float, p2_y: float) -> float:
    """Numba-optimized squared distance."""
    dx = p1_x - p2_x
    dy = p1_y - p2_y
    return dx * dx + dy * dy


@njit(cache=True)
def norm_numba(x: float, y: float) -> float:
    """Numba-optimized norm."""
    return math.sqrt(x * x + y * y)


@njit(cache=True)
def cyl_to_cart_numba(radius: float, theta: float, ori_x: float, ori_y: float) -> tuple:
    """Numba-optimized polar to cartesian conversion."""
    return (radius * math.cos(theta) + ori_x, radius * math.sin(theta) + ori_y)


@njit(cache=True)
def get_nearest_point_numba(
    ref_x: float, ref_y: float,
    p1_x: float, p1_y: float,
    p2_x: float, p2_y: float
) -> tuple:
    """Numba-optimized nearest point selection."""
    d1 = sqr_dist_numba(ref_x, ref_y, p1_x, p1_y)
    d2 = sqr_dist_numba(ref_x, ref_y, p2_x, p2_y)
    if d1 < d2:
        return (p1_x, p1_y)
    return (p2_x, p2_y)


@njit(cache=True)
def circle_intersect_numba(
    x1: float, y1: float, r1: float,
    x2: float, y2: float, r2: float,
    tol: float = 0.0
) -> tuple:
    """
    Numba-optimized circle intersection.

    Returns: (n_intersections, x1, y1, x2, y2)
    where n_intersections is 0, 1, 2, or 3 (same circle)
    For n=0: other values are undefined
    For n=1: (x1, y1) is the intersection
    For n=2: (x1, y1) and (x2, y2) are intersections
    For n=3: same circle case
    """
    dist_x = x2 - x1
    dist_y = y2 - y1
    distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

    # Check for no intersection cases
    if distance > r1 + r2:
        return (0, 0.0, 0.0, 0.0, 0.0)
    if distance < abs(r2 - r1):
        return (0, 0.0, 0.0, 0.0, 0.0)
    if distance <= tol and abs(r1 - r2) <= tol:
        return (3, x1, y1, r1, 0.0)  # Same circle

    # Check for tangent case
    dual = True
    if abs(abs(r1 - distance) - r2) <= tol:
        dual = False

    # Distance from first circle's center to projection point
    mid_dist = (r1 * r1 - r2 * r2 + distance * distance) / (2.0 * distance)

    # Projected point
    proj_x = x1 + (mid_dist * dist_x) / distance
    proj_y = y1 + (mid_dist * dist_y) / distance

    if not dual:
        return (1, proj_x, proj_y, 0.0, 0.0)

    # Two intersections
    height_squared = max(0.0, r1 * r1 - mid_dist * mid_dist)
    height = math.sqrt(height_squared) / distance

    inter1_x = proj_x + height * dist_y
    inter1_y = proj_y - height * dist_x
    inter2_x = proj_x - height * dist_y
    inter2_y = proj_y + height * dist_x

    return (2, inter1_x, inter1_y, inter2_x, inter2_y)


@njit(cache=True)
def circle_line_numba(
    cx: float, cy: float, r: float,
    p1_x: float, p1_y: float,
    p2_x: float, p2_y: float
) -> tuple:
    """
    Numba-optimized circle-line intersection.

    Returns: (n_intersections, x1, y1, x2, y2)
    """
    # Move axis to circle center
    fp_x = p1_x - cx
    fp_y = p1_y - cy
    sp_x = p2_x - cx
    sp_y = p2_y - cy

    dx = sp_x - fp_x
    dy = sp_y - fp_y

    dr2 = dx * dx + dy * dy
    cross = fp_x * sp_y - sp_x * fp_y
    discriminant = r * r * dr2 - cross * cross

    if discriminant < 0:
        return (0, 0.0, 0.0, 0.0, 0.0)

    reduced_x = cross / dr2
    reduced_y = math.sqrt(discriminant) / dr2

    if discriminant == 0:
        # Tangent line
        return (1, reduced_x * dy + cx, -reduced_x * dx + cy, 0.0, 0.0)

    # Two intersections
    sign_dy = 1.0 if dy >= 0 else -1.0
    abs_dy = abs(dy)

    x1 = reduced_x * dy - sign_dy * dx * reduced_y + cx
    y1 = -reduced_x * dx - abs_dy * reduced_y + cy
    x2 = reduced_x * dy + sign_dy * dx * reduced_y + cx
    y2 = -reduced_x * dx + abs_dy * reduced_y + cy

    return (2, x1, y1, x2, y2)


# ============================================================================
# NUMPY VECTORIZED IMPLEMENTATIONS
# ============================================================================

def sqr_dist_numpy(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Vectorized squared distance for arrays of points."""
    diff = points1 - points2
    return np.sum(diff * diff, axis=1)


def cyl_to_cart_numpy(radii: np.ndarray, thetas: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Vectorized polar to cartesian conversion."""
    x = radii * np.cos(thetas) + origins[:, 0]
    y = radii * np.sin(thetas) + origins[:, 1]
    return np.column_stack([x, y])


def circle_intersect_numpy(circles1: np.ndarray, circles2: np.ndarray) -> np.ndarray:
    """
    Vectorized circle intersection for arrays of circles.

    circles1, circles2: Nx3 arrays of (x, y, r)
    Returns: Nx5 array of (n_intersections, x1, y1, x2, y2)
    """
    n = len(circles1)
    results = np.zeros((n, 5))

    x1, y1, r1 = circles1[:, 0], circles1[:, 1], circles1[:, 2]
    x2, y2, r2 = circles2[:, 0], circles2[:, 1], circles2[:, 2]

    dist_x = x2 - x1
    dist_y = y2 - y1
    distance = np.sqrt(dist_x**2 + dist_y**2)

    # Mask for valid intersections
    valid = (distance <= r1 + r2) & (distance >= np.abs(r2 - r1))

    # For valid circles, compute intersections
    mid_dist = np.where(
        distance > 0,
        (r1**2 - r2**2 + distance**2) / (2.0 * distance),
        0
    )

    proj_x = x1 + np.where(distance > 0, (mid_dist * dist_x) / distance, 0)
    proj_y = y1 + np.where(distance > 0, (mid_dist * dist_y) / distance, 0)

    height_squared = np.maximum(0.0, r1**2 - mid_dist**2)
    height = np.where(distance > 0, np.sqrt(height_squared) / distance, 0)

    # Two intersection case (most common)
    results[:, 0] = np.where(valid, 2, 0)
    results[:, 1] = proj_x + height * dist_y
    results[:, 2] = proj_y - height * dist_x
    results[:, 3] = proj_x - height * dist_y
    results[:, 4] = proj_y + height * dist_x

    return results


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def benchmark_single(func: Callable, args: tuple, n_iterations: int = 100_000, warmup: int = 1000) -> dict:
    """Benchmark a function with fixed arguments."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        func(*args)
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "min_ns": min(times),
        "total_ms": sum(times) / 1_000_000,
    }


def benchmark_batch(func: Callable, args: tuple, n_iterations: int = 1000, warmup: int = 100) -> dict:
    """Benchmark a vectorized function with batch arguments."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual benchmark
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter_ns()
        func(*args)
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "min_ns": min(times),
        "total_ms": sum(times) / 1_000_000,
    }


def format_comparison(name: str, orig: dict, new: dict, new_name: str = "optimized") -> str:
    """Format comparison between original and optimized."""
    speedup = orig["mean_ns"] / new["mean_ns"] if new["mean_ns"] > 0 else float("inf")
    return (
        f"{name}:\n"
        f"  Original:  {orig['mean_ns']:8.1f} ns\n"
        f"  {new_name:10s}: {new['mean_ns']:8.1f} ns\n"
        f"  Speedup:   {speedup:8.2f}x\n"
    )


# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================

def run_numba_comparison():
    """Compare original vs numba implementations."""
    print("=" * 70)
    print("NUMBA JIT COMPILATION COMPARISON")
    print("(100,000 iterations each, fixed inputs)")
    print("=" * 70)
    print()

    random.seed(42)
    n_iter = 100_000

    results = []

    # sqr_dist
    p1, p2 = (1.5, 2.3), (4.7, 8.1)
    orig = benchmark_single(lambda: sqr_dist_orig(p1, p2), (), n_iter)
    # For numba, we need to unpack tuples
    numba = benchmark_single(lambda: sqr_dist_numba(p1[0], p1[1], p2[0], p2[1]), (), n_iter)
    print(format_comparison("sqr_dist", orig, numba, "numba"))
    results.append(("sqr_dist", orig, numba))

    # norm
    vec = (3.0, 4.0)
    orig = benchmark_single(lambda: norm_orig(vec), (), n_iter)
    numba = benchmark_single(lambda: norm_numba(vec[0], vec[1]), (), n_iter)
    print(format_comparison("norm", orig, numba, "numba"))
    results.append(("norm", orig, numba))

    # cyl_to_cart
    r, theta, ori = 5.0, 0.785, (1.0, 2.0)
    orig = benchmark_single(lambda: cyl_to_cart_orig(r, theta, ori), (), n_iter)
    numba = benchmark_single(lambda: cyl_to_cart_numba(r, theta, ori[0], ori[1]), (), n_iter)
    print(format_comparison("cyl_to_cart", orig, numba, "numba"))
    results.append(("cyl_to_cart", orig, numba))

    # get_nearest_point
    ref, p1, p2 = (0.0, 0.0), (1.0, 0.0), (5.0, 0.0)
    orig = benchmark_single(lambda: get_nearest_point_orig(ref, p1, p2), (), n_iter)
    numba = benchmark_single(
        lambda: get_nearest_point_numba(ref[0], ref[1], p1[0], p1[1], p2[0], p2[1]), (), n_iter
    )
    print(format_comparison("get_nearest_point", orig, numba, "numba"))
    results.append(("get_nearest_point", orig, numba))

    # circle_intersect (two intersections case)
    c1, c2 = (0.0, 0.0, 5.0), (3.0, 0.0, 4.0)
    orig = benchmark_single(lambda: circle_intersect_orig(c1, c2), (), n_iter)
    numba = benchmark_single(
        lambda: circle_intersect_numba(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]), (), n_iter
    )
    print(format_comparison("circle_intersect", orig, numba, "numba"))
    results.append(("circle_intersect", orig, numba))

    # circle_line_from_points_intersection
    circle = (0.0, 0.0, 5.0)
    p1, p2 = (-10.0, 2.0), (10.0, 2.0)
    orig = benchmark_single(lambda: circle_line_orig(circle, p1, p2), (), n_iter)
    numba = benchmark_single(
        lambda: circle_line_numba(circle[0], circle[1], circle[2], p1[0], p1[1], p2[0], p2[1]), (), n_iter
    )
    print(format_comparison("circle_line_intersection", orig, numba, "numba"))
    results.append(("circle_line", orig, numba))

    return results


def run_numpy_comparison():
    """Compare individual vs vectorized numpy operations."""
    print("=" * 70)
    print("NUMPY VECTORIZATION COMPARISON")
    print("(Batch of 1000 operations)")
    print("=" * 70)
    print()

    random.seed(42)
    batch_size = 1000
    n_iter = 1000

    results = []

    # Generate batch data
    points1 = np.random.uniform(-100, 100, (batch_size, 2))
    points2 = np.random.uniform(-100, 100, (batch_size, 2))

    # sqr_dist batch
    def loop_sqr_dist():
        return [sqr_dist_orig(tuple(p1), tuple(p2)) for p1, p2 in zip(points1, points2)]

    orig = benchmark_batch(loop_sqr_dist, (), n_iter, warmup=10)
    numpy_result = benchmark_batch(lambda: sqr_dist_numpy(points1, points2), (), n_iter, warmup=10)
    print(f"sqr_dist (batch of {batch_size}):")
    print(f"  Loop:     {orig['mean_ns']/1000:8.1f} µs  ({orig['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  NumPy:    {numpy_result['mean_ns']/1000:8.1f} µs  ({numpy_result['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  Speedup:  {orig['mean_ns']/numpy_result['mean_ns']:.2f}x")
    print()
    results.append(("sqr_dist", orig, numpy_result))

    # cyl_to_cart batch
    radii = np.random.uniform(1, 100, batch_size)
    thetas = np.random.uniform(0, 2*np.pi, batch_size)
    origins = np.random.uniform(-50, 50, (batch_size, 2))

    def loop_cyl():
        return [cyl_to_cart_orig(r, t, tuple(o)) for r, t, o in zip(radii, thetas, origins)]

    orig = benchmark_batch(loop_cyl, (), n_iter, warmup=10)
    numpy_result = benchmark_batch(lambda: cyl_to_cart_numpy(radii, thetas, origins), (), n_iter, warmup=10)
    print(f"cyl_to_cart (batch of {batch_size}):")
    print(f"  Loop:     {orig['mean_ns']/1000:8.1f} µs  ({orig['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  NumPy:    {numpy_result['mean_ns']/1000:8.1f} µs  ({numpy_result['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  Speedup:  {orig['mean_ns']/numpy_result['mean_ns']:.2f}x")
    print()
    results.append(("cyl_to_cart", orig, numpy_result))

    # circle_intersect batch
    # Generate intersecting circles
    circles1 = np.zeros((batch_size, 3))
    circles2 = np.zeros((batch_size, 3))
    circles1[:, 0] = np.random.uniform(-50, 50, batch_size)
    circles1[:, 1] = np.random.uniform(-50, 50, batch_size)
    circles1[:, 2] = np.random.uniform(3, 10, batch_size)
    # Make circles2 intersect with circles1
    angles = np.random.uniform(0, 2*np.pi, batch_size)
    dists = np.random.uniform(2, 8, batch_size)
    circles2[:, 0] = circles1[:, 0] + dists * np.cos(angles)
    circles2[:, 1] = circles1[:, 1] + dists * np.sin(angles)
    circles2[:, 2] = np.random.uniform(3, 10, batch_size)

    def loop_circle():
        return [circle_intersect_orig(tuple(c1), tuple(c2)) for c1, c2 in zip(circles1, circles2)]

    orig = benchmark_batch(loop_circle, (), n_iter, warmup=10)
    numpy_result = benchmark_batch(lambda: circle_intersect_numpy(circles1, circles2), (), n_iter, warmup=10)
    print(f"circle_intersect (batch of {batch_size}):")
    print(f"  Loop:     {orig['mean_ns']/1000:8.1f} µs  ({orig['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  NumPy:    {numpy_result['mean_ns']/1000:8.1f} µs  ({numpy_result['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  Speedup:  {orig['mean_ns']/numpy_result['mean_ns']:.2f}x")
    print()
    results.append(("circle_intersect", orig, numpy_result))

    return results


def run_combined_numba_numpy():
    """Test numba with batch processing via numpy arrays."""
    print("=" * 70)
    print("COMBINED NUMBA + NUMPY (Batch Processing with JIT)")
    print("(Batch of 1000 operations)")
    print("=" * 70)
    print()

    batch_size = 1000
    n_iter = 1000

    # Generate intersecting circles
    np.random.seed(42)
    circles1 = np.zeros((batch_size, 3))
    circles2 = np.zeros((batch_size, 3))
    circles1[:, 0] = np.random.uniform(-50, 50, batch_size)
    circles1[:, 1] = np.random.uniform(-50, 50, batch_size)
    circles1[:, 2] = np.random.uniform(3, 10, batch_size)
    angles = np.random.uniform(0, 2*np.pi, batch_size)
    dists = np.random.uniform(2, 8, batch_size)
    circles2[:, 0] = circles1[:, 0] + dists * np.cos(angles)
    circles2[:, 1] = circles1[:, 1] + dists * np.sin(angles)
    circles2[:, 2] = np.random.uniform(3, 10, batch_size)

    # Pure Python loop with original
    def loop_original():
        return [circle_intersect_orig(tuple(c1), tuple(c2)) for c1, c2 in zip(circles1, circles2)]

    # Python loop with numba functions
    def loop_numba():
        results = []
        for c1, c2 in zip(circles1, circles2):
            results.append(circle_intersect_numba(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]))
        return results

    # Vectorized numpy
    def batch_numpy():
        return circle_intersect_numpy(circles1, circles2)

    orig = benchmark_batch(loop_original, (), n_iter, warmup=10)
    numba_result = benchmark_batch(loop_numba, (), n_iter, warmup=10)
    numpy_result = benchmark_batch(batch_numpy, (), n_iter, warmup=10)

    print(f"circle_intersect (batch of {batch_size}):")
    print(f"  Python loop:      {orig['mean_ns']/1000:8.1f} µs  ({orig['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  Numba loop:       {numba_result['mean_ns']/1000:8.1f} µs  ({numba_result['mean_ns']/batch_size:.1f} ns/op)")
    print(f"  NumPy vectorized: {numpy_result['mean_ns']/1000:8.1f} µs  ({numpy_result['mean_ns']/batch_size:.1f} ns/op)")
    print()
    print(f"  Numba speedup vs Python: {orig['mean_ns']/numba_result['mean_ns']:.2f}x")
    print(f"  NumPy speedup vs Python: {orig['mean_ns']/numpy_result['mean_ns']:.2f}x")


def main():
    """Run all comparison benchmarks."""
    print()
    print("PYLINKAGE OPTIMIZATION BENCHMARKS")
    print("=" * 70)
    print()

    # Force numba compilation before benchmarking
    print("Warming up Numba JIT compilation...")
    _ = sqr_dist_numba(1.0, 2.0, 3.0, 4.0)
    _ = norm_numba(3.0, 4.0)
    _ = cyl_to_cart_numba(5.0, 0.5, 1.0, 2.0)
    _ = get_nearest_point_numba(0.0, 0.0, 1.0, 0.0, 5.0, 0.0)
    _ = circle_intersect_numba(0.0, 0.0, 5.0, 3.0, 0.0, 4.0)
    _ = circle_line_numba(0.0, 0.0, 5.0, -10.0, 2.0, 10.0, 2.0)
    print("JIT compilation complete.")
    print()

    numba_results = run_numba_comparison()

    print()
    numpy_results = run_numpy_comparison()

    print()
    run_combined_numba_numpy()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Numba JIT Speedups (single operation):")
    for name, orig, numba in numba_results:
        speedup = orig["mean_ns"] / numba["mean_ns"]
        print(f"  {name}: {speedup:.2f}x faster")

    print()
    print("NumPy Vectorization Speedups (batch of 1000):")
    for name, orig, numpy in numpy_results:
        speedup = orig["mean_ns"] / numpy["mean_ns"]
        print(f"  {name}: {speedup:.2f}x faster")

    print()
    print("Recommendations:")
    print("  - Use Numba for single operations in the hot path (joint.reload())")
    print("  - Use NumPy for batch operations during optimization")
    print("  - For linkage simulation: Numba is best (sequential joint solving)")
    print("  - For optimization: NumPy batching could parallelize evaluations")


if __name__ == "__main__":
    main()
