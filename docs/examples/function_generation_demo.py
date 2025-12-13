#!/usr/bin/env python3
"""
Function Generation Synthesis Demo.

This demo shows how to synthesize a four-bar linkage for function generation
using Freudenstein's equation. Function generation creates a linkage where
the input crank angle produces a desired output rocker angle.

To run:
    uv run python docs/examples/function_generation_demo.py

Theory:
    Freudenstein's equation relates input (theta2) and output (theta4) angles:
        R1*cos(theta4) - R2*cos(theta2) + R3 - cos(theta2 - theta4) = 0

    Given 3 angle pairs, there is a unique solution. With more pairs,
    least-squares fitting is used.

Applications:
    - Mechanical computing devices
    - Coordinated motion mechanisms
    - Transfer functions in machinery
    - Windshield wiper mechanisms

References:
    - Freudenstein, F. "Approximate Synthesis of Four-Bar Linkages" (1955)
    - Sandor & Erdman, Chapter 3: "Analytical Linkage Synthesis"
"""

import math

import pylinkage as pl
from pylinkage.synthesis import (
    FourBarSolution,
    function_generation,
    grashof_check,
    verify_function_generation,
)


def demo_basic_function_generation():
    """Basic example: synthesize a linkage for 3 precision positions.

    With exactly 3 angle pairs, we get a unique exact solution
    (if physically realizable).
    """
    print("=" * 60)
    print("Demo 1: Basic Function Generation (3 Precision Positions)")
    print("=" * 60)

    # Define input/output angle relationship
    # We want the output rocker to move through specific angles
    # when the input crank is at specific angles
    #
    # Note: Not all angle combinations produce physically valid linkages.
    # The Freudenstein equation may yield negative link lengths or
    # non-Grashof configurations for certain inputs.
    angle_pairs = [
        (0.0, 0.0),  # Position 1: both at 0
        (0.3, 0.2),  # Position 2: ~17 deg -> ~11 deg
        (0.6, 0.4),  # Position 3: ~34 deg -> ~23 deg
    ]

    print("\nPrecision positions (crank angle -> rocker angle):")
    for i, (theta2, theta4) in enumerate(angle_pairs, 1):
        print(f"  Position {i}: {math.degrees(theta2):.1f} deg -> {math.degrees(theta4):.1f} deg")

    # Perform synthesis
    result = function_generation(angle_pairs, ground_length=4.0)

    if result:
        print(f"\nFound {len(result)} solution(s)")
        linkage = result.solutions[0]

        # Show link lengths from raw solution
        raw: FourBarSolution = result.raw_solutions[0]
        print("\nLink lengths:")
        print(f"  Crank (a):   {raw.crank_length:.4f}")
        print(f"  Coupler (b): {raw.coupler_length:.4f}")
        print(f"  Rocker (c):  {raw.rocker_length:.4f}")
        print(f"  Ground (d):  {raw.ground_length:.4f}")

        # Check Grashof condition
        grashof_type = grashof_check(
            raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
        )
        print(f"\nGrashof type: {grashof_type.name}")

        # Verify the solution
        is_valid, errors = verify_function_generation(linkage, angle_pairs)
        print(f"\nVerification: {'PASSED' if is_valid else 'FAILED'}")
        for i, err in enumerate(errors, 1):
            print(f"  Position {i} error: {math.degrees(err):.3f} deg")

        # Visualize
        print("\nVisualizing synthesized linkage...")
        pl.show_linkage(linkage)
    else:
        print("No valid solution found")
        for warning in result.warnings:
            print(f"  Warning: {warning}")


def demo_approximate_function_generation():
    """Approximate synthesis with more than 3 precision positions.

    With 4+ angle pairs, we use least-squares fitting to find the
    best approximate solution.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Approximate Function Generation (5 Positions)")
    print("=" * 60)

    # Define 5 precision positions
    # More positions mean tighter constraints - exact match unlikely
    angle_pairs = [
        (0.0, 0.0),
        (0.2, 0.15),
        (0.4, 0.28),
        (0.6, 0.40),
        (0.8, 0.55),
    ]

    print("\nPrecision positions (5 points - overdetermined):")
    for i, (theta2, theta4) in enumerate(angle_pairs, 1):
        print(f"  Position {i}: {math.degrees(theta2):.1f} deg -> {math.degrees(theta4):.1f} deg")

    # Perform synthesis with relaxed Grashof constraint
    result = function_generation(
        angle_pairs,
        ground_length=4.0,
        require_grashof=False,  # Accept non-Grashof solutions
    )

    if result:
        print(f"\nFound {len(result)} solution(s)")
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nLink lengths (least-squares fit):")
        print(f"  Crank (a):   {raw.crank_length:.4f}")
        print(f"  Coupler (b): {raw.coupler_length:.4f}")
        print(f"  Rocker (c):  {raw.rocker_length:.4f}")
        print(f"  Ground (d):  {raw.ground_length:.4f}")

        # Verify - expect some error due to overdetermined system
        is_valid, errors = verify_function_generation(linkage, angle_pairs, tolerance=0.1)
        print(f"\nVerification (tolerance 0.1 rad): {'PASSED' if is_valid else 'FAILED'}")
        for i, err in enumerate(errors, 1):
            print(f"  Position {i} error: {math.degrees(err):.3f} deg")

        print("\nVisualizing approximate solution...")
        pl.show_linkage(linkage)
    else:
        print("No valid solution found")
        for warning in result.warnings:
            print(f"  Warning: {warning}")


def demo_crank_rocker_synthesis():
    """Synthesize specifically a crank-rocker mechanism.

    A crank-rocker allows continuous rotation of the input crank
    while the output rocker oscillates. This is useful for many
    practical applications.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Crank-Rocker Mechanism Synthesis")
    print("=" * 60)

    # Design for a crank-rocker: we need to ensure the
    # shortest link (crank) plus longest link < sum of other two
    # This is the Grashof criterion for crank-rocker

    # Specify angle pairs that span a reasonable range
    # These angles produce a crank-rocker with good proportions
    angle_pairs = [
        (0.0, 0.0),
        (0.5, 0.8),
        (1.0, 1.5),
    ]

    print("\nPrecision positions for crank-rocker:")
    for i, (theta2, theta4) in enumerate(angle_pairs, 1):
        print(f"  Position {i}: {math.degrees(theta2):.1f} deg -> {math.degrees(theta4):.1f} deg")

    result = function_generation(
        angle_pairs,
        ground_length=5.0,
        require_grashof=True,
        require_crank_rocker=True,  # Specifically require crank-rocker type
    )

    if result:
        print(f"\nFound {len(result)} crank-rocker solution(s)")
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        grashof_type = grashof_check(
            raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
        )

        print("\nLink lengths:")
        print(f"  Crank (a):   {raw.crank_length:.4f}")
        print(f"  Coupler (b): {raw.coupler_length:.4f}")
        print(f"  Rocker (c):  {raw.rocker_length:.4f}")
        print(f"  Ground (d):  {raw.ground_length:.4f}")
        print(f"\nGrashof type: {grashof_type.name}")

        # Show the linkage going through a full rotation
        print("\nVisualizing crank-rocker mechanism...")
        pl.show_linkage(linkage)
    else:
        print("No crank-rocker solution found")
        for warning in result.warnings:
            print(f"  Warning: {warning}")
        print("\nTry adjusting the angle pairs to allow a crank-rocker configuration.")


def demo_compare_grashof_types():
    """Compare different Grashof configurations.

    Demonstrates how different angle specifications lead to
    different mechanism types.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Exploring Different Grashof Types")
    print("=" * 60)

    # Different angle specifications leading to different mechanism types
    test_cases = [
        ("Small output swing", [(0, 0), (0.3, 0.2), (0.6, 0.4)]),
        ("Large output swing", [(0, 0), (0.5, 0.8), (1.0, 1.5)]),
        ("Symmetric motion", [(0, 0), (0.5, 0.5), (1.0, 1.0)]),
    ]

    for name, angle_pairs in test_cases:
        print(f"\n--- {name} ---")
        print(
            f"Angles: {[(f'{math.degrees(t2):.1f}', f'{math.degrees(t4):.1f}') for t2, t4 in angle_pairs]}"
        )

        result = function_generation(
            angle_pairs,
            ground_length=4.0,
            require_grashof=False,
        )

        if result:
            raw: FourBarSolution = result.raw_solutions[0]
            grashof_type = grashof_check(
                raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
            )
            print(f"  Grashof type: {grashof_type.name}")
            print(
                f"  Link lengths: a={raw.crank_length:.2f}, b={raw.coupler_length:.2f}, "
                f"c={raw.rocker_length:.2f}, d={raw.ground_length:.2f}"
            )
        else:
            print(f"  No solution: {result.warnings}")


def main():
    """Run all function generation demos."""
    print("\n" + "#" * 60)
    print("# FUNCTION GENERATION SYNTHESIS DEMOS")
    print("# Using Freudenstein's Equation")
    print("#" * 60)

    demo_basic_function_generation()
    demo_approximate_function_generation()
    demo_crank_rocker_synthesis()
    demo_compare_grashof_types()

    print("\n" + "=" * 60)
    print("Function generation demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
