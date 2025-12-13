#!/usr/bin/env python3
"""
Path Generation Synthesis Demo.

This demo shows how to synthesize a four-bar linkage for path generation
using Burmester theory. Path generation creates a linkage where a point
on the coupler traces a curve through specified precision points.

To run:
    uv run python docs/examples/path_generation_demo.py

Theory:
    Path generation finds linkages where a coupler point passes through
    specified precision points. The coupler orientation at each point is
    a free variable, which is searched over using Burmester theory.

    Burmester's circle point and center point curves identify all possible
    pivot points that allow a body to pass through the precision positions.

Applications:
    - Walking mechanisms (leg tip trajectories)
    - Pick-and-place motions
    - Assembly line transfer mechanisms
    - Drawing machines (pantographs)
    - Film advance mechanisms

References:
    - McCarthy, J.M. "Geometric Design of Linkages" Chapter 7
    - Sandor & Erdman, Chapter 5: "Path, Motion, and Function Generation"
    - Wampler et al., "Complete Solution of the Nine-Point Path Synthesis"
"""

import pylinkage as pl
from pylinkage.synthesis import (
    FourBarSolution,
    grashof_check,
    path_generation,
)


def demo_basic_path_generation():
    """Basic example: synthesize a linkage for 3 precision points.

    Path generation with 3 points gives reliable results.
    With 4+ points, the problem becomes more constrained and
    may not have solutions for arbitrary point configurations.
    """
    print("=" * 60)
    print("Demo 1: Basic Path Generation (3 Precision Points)")
    print("=" * 60)

    # Define the path that the coupler point should trace
    # Using 3 points gives more reliable results
    precision_points = [
        (0.0, 0.0),  # Point 1
        (1.0, 1.0),  # Point 2
        (2.0, 0.0),  # Point 3
    ]

    print("\nPrecision points (x, y):")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    # Perform synthesis
    result = path_generation(
        precision_points,
        max_solutions=5,
        require_grashof=True,
    )

    print(f"\nFound {len(result)} solution(s)")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result:
        # Show the first solution
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nBest solution - Link lengths:")
        print(f"  Crank (a):   {raw.crank_length:.4f}")
        print(f"  Coupler (b): {raw.coupler_length:.4f}")
        print(f"  Rocker (c):  {raw.rocker_length:.4f}")
        print(f"  Ground (d):  {raw.ground_length:.4f}")

        grashof_type = grashof_check(
            raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
        )
        print(f"  Grashof type: {grashof_type.name}")

        print("\nVisualizing synthesized linkage...")
        pl.show_linkage(linkage)


def demo_walking_mechanism():
    """Synthesize a linkage for a walking leg trajectory.

    Walking mechanisms typically need the coupler point (foot)
    to trace a D-shaped or figure-8 path.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Walking Mechanism Path Generation")
    print("=" * 60)

    # D-shaped path for a walking leg
    # Using 3 key points: start, apex, end
    precision_points = [
        (0.0, 0.0),  # Ground contact (start)
        (1.0, 0.8),  # Lifting/swing apex
        (2.0, 0.0),  # Ground contact (end)
    ]

    print("\nWalking path precision points:")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    result = path_generation(
        precision_points,
        max_solutions=10,
        require_grashof=True,
        n_orientation_samples=48,  # More samples for better coverage
    )

    print(f"\nFound {len(result)} walking mechanism solution(s)")

    if result:
        # Show top 3 solutions
        print("\nTop solutions:")
        for i, raw in enumerate(result.raw_solutions[:3], 1):
            grashof_type = grashof_check(
                raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
            )
            print(f"\n  Solution {i}:")
            print(f"    Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
            print(f"    Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")
            print(f"    Type: {grashof_type.name}")

        print("\nVisualizing best walking mechanism...")
        pl.show_linkage(result.solutions[0])


def demo_straight_line_approximation():
    """Synthesize a linkage that approximates a straight line.

    Straight-line mechanisms are historically important and
    practically useful for many applications.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Straight Line Approximation")
    print("=" * 60)

    # Collinear points for straight line
    # Using 3 points for reliable synthesis
    precision_points = [
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ]

    print("\nPrecision points for straight line:")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    # Relax constraints since exact straight line is difficult
    result = path_generation(
        precision_points,
        max_solutions=5,
        require_grashof=False,  # Accept all Grashof types
        n_orientation_samples=72,
    )

    print(f"\nFound {len(result)} straight-line approximation(s)")

    if result.warnings:
        print("Notes:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nBest straight-line approximation:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing straight-line mechanism...")
        pl.show_linkage(linkage)
    else:
        print("No solutions found. Perfect straight lines are challenging.")
        print("Try using only 3 points or adjusting their positions.")


def demo_loop_path():
    """Synthesize a linkage for a looping/figure-8 path.

    The coupler curve of a four-bar is a tricircular sextic,
    capable of producing complex looping shapes.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Loop Path Generation")
    print("=" * 60)

    # Points forming a loop (3 points)
    precision_points = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
    ]

    print("\nLoop path precision points:")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    result = path_generation(
        precision_points,
        max_solutions=5,
        require_grashof=True,
    )

    print(f"\nFound {len(result)} loop mechanism(s)")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nLoop mechanism:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing loop mechanism...")
        pl.show_linkage(linkage)


def demo_constrained_ground_pivots():
    """Path generation with constrained ground pivot positions.

    Sometimes the ground pivot locations are predetermined
    by the physical design constraints.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Constrained Ground Pivots")
    print("=" * 60)

    # Using 3 points for reliable synthesis
    precision_points = [
        (2.0, 3.0),
        (3.0, 3.5),
        (4.0, 3.0),
    ]

    # Fix the ground pivot positions
    ground_pivot_a = (0.0, 0.0)  # Crank pivot
    ground_pivot_d = (6.0, 0.0)  # Rocker pivot

    print("\nPrecision points:")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    print("\nConstrained ground pivots:")
    print(f"  A (crank base):  {ground_pivot_a}")
    print(f"  D (rocker base): {ground_pivot_d}")

    result = path_generation(
        precision_points,
        ground_pivot_a=ground_pivot_a,
        ground_pivot_d=ground_pivot_d,
        max_solutions=5,
        require_grashof=False,
    )

    print(f"\nFound {len(result)} solution(s) with fixed ground")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nSolution with constrained ground:")
        print(f"  Ground pivot A: ({raw.ground_pivot_a[0]:.3f}, {raw.ground_pivot_a[1]:.3f})")
        print(f"  Ground pivot D: ({raw.ground_pivot_d[0]:.3f}, {raw.ground_pivot_d[1]:.3f})")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing constrained mechanism...")
        pl.show_linkage(linkage)
    else:
        print("No solutions found with the given ground constraints.")
        print("Try moving the precision points or relaxing the ground constraints.")


def demo_compare_solutions():
    """Compare multiple solutions for the same path.

    For the same set of precision points, multiple valid
    four-bar linkages may exist with different characteristics.
    """
    print("\n" + "=" * 60)
    print("Demo 6: Comparing Multiple Solutions")
    print("=" * 60)

    # Using 3 points to get multiple solutions
    precision_points = [
        (0.0, 0.0),
        (2.0, 2.0),
        (4.0, 0.0),
    ]

    print("\nPrecision points:")
    for i, (x, y) in enumerate(precision_points, 1):
        print(f"  Point {i}: ({x:.2f}, {y:.2f})")

    result = path_generation(
        precision_points,
        max_solutions=10,
        require_grashof=False,  # Allow all types to see variety
    )

    print(f"\nFound {len(result)} different solutions")

    if result:
        print("\nSolution comparison:")
        print("-" * 70)
        print(f"{'#':<3} {'Crank':<10} {'Coupler':<10} {'Rocker':<10} {'Ground':<10} {'Type':<15}")
        print("-" * 70)

        for i, raw in enumerate(result.raw_solutions, 1):
            grashof_type = grashof_check(
                raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
            )
            print(
                f"{i:<3} {raw.crank_length:<10.3f} {raw.coupler_length:<10.3f} "
                f"{raw.rocker_length:<10.3f} {raw.ground_length:<10.3f} {grashof_type.name:<15}"
            )

        print("-" * 70)

        print("\nVisualizing first solution...")
        pl.show_linkage(result.solutions[0])

        if len(result.solutions) > 1:
            print("\nVisualizing second solution for comparison...")
            pl.show_linkage(result.solutions[1])


def main():
    """Run all path generation demos."""
    print("\n" + "#" * 60)
    print("# PATH GENERATION SYNTHESIS DEMOS")
    print("# Using Burmester Theory")
    print("#" * 60)

    demo_basic_path_generation()
    demo_walking_mechanism()
    demo_straight_line_approximation()
    demo_loop_path()
    demo_constrained_ground_pivots()
    demo_compare_solutions()

    print("\n" + "=" * 60)
    print("Path generation demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
