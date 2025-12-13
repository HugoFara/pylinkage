#!/usr/bin/env python3
"""
Four-Bar Linkage from Link Lengths Demo.

This demo shows how to directly create four-bar linkages from link lengths
without going through synthesis, and how to analyze their kinematic properties.

To run:
    uv run python docs/examples/fourbar_from_lengths_demo.py

Theory:
    A four-bar linkage has four links:
    - Ground (d): Fixed frame between ground pivots A and D
    - Crank (a): Input link connected to ground at A
    - Coupler (b): Floating link connecting crank to rocker
    - Rocker (c): Output link connected to ground at D

    Grashof's Law: s + l <= p + q
    Where s = shortest, l = longest, p and q = other two links
    - If satisfied: At least one link can fully rotate
    - If not satisfied: No link can fully rotate (double-rocker)

Grashof Types (when Grashof condition is satisfied):
    - Crank-Rocker: Shortest link is ground or crank
    - Double-Crank: Shortest link is ground (both can rotate)
    - Rocker-Crank: Shortest link is rocker
    - Change-Point: s + l = p + q (singular positions exist)

References:
    - Norton, R.L. "Design of Machinery" Chapter 4
    - Shigley & Uicker, "Theory of Machines and Mechanisms"
"""

import math

import pylinkage as pl
from pylinkage.synthesis import (
    fourbar_from_lengths,
    grashof_check,
    is_crank_rocker,
    is_grashof,
    linkage_to_synthesis_params,
)


def demo_basic_fourbar():
    """Create a basic four-bar linkage from link lengths."""
    print("=" * 60)
    print("Demo 1: Basic Four-Bar Creation")
    print("=" * 60)

    # Classic four-bar proportions
    crank = 1.0
    coupler = 3.5
    rocker = 3.0
    ground = 4.0

    print("\nLink lengths:")
    print(f"  Crank (a):   {crank}")
    print(f"  Coupler (b): {coupler}")
    print(f"  Rocker (c):  {rocker}")
    print(f"  Ground (d):  {ground}")

    # Create the linkage
    linkage = fourbar_from_lengths(
        crank_length=crank,
        coupler_length=coupler,
        rocker_length=rocker,
        ground_length=ground,
        ground_pivot_a=(0.0, 0.0),
        initial_crank_angle=math.pi / 4,  # Start at 45 degrees
    )

    # Analyze
    grashof_type = grashof_check(crank, coupler, rocker, ground)
    print("\nGrashof analysis:")
    print(f"  Type: {grashof_type.name}")
    print(f"  Is Grashof: {is_grashof(crank, coupler, rocker, ground)}")
    print(f"  Is Crank-Rocker: {is_crank_rocker(crank, coupler, rocker, ground)}")

    print("\nVisualizing basic four-bar...")
    pl.show_linkage(linkage)


def demo_grashof_types():
    """Demonstrate all Grashof mechanism types."""
    print("\n" + "=" * 60)
    print("Demo 2: Grashof Mechanism Types")
    print("=" * 60)

    # Different configurations to show each type
    configurations = [
        # (name, crank, coupler, rocker, ground)
        ("Crank-Rocker", 1.0, 4.0, 3.0, 4.0),  # s=crank, can rotate
        ("Double-Crank (Drag-Link)", 2.0, 4.0, 3.0, 1.0),  # s=ground, both rotate
        ("Rocker-Crank", 3.0, 4.0, 1.0, 4.0),  # s=rocker
        ("Double-Rocker", 3.0, 4.0, 3.0, 2.0),  # Non-Grashof
        ("Change-Point", 1.0, 3.0, 2.0, 4.0),  # s+l = p+q
    ]

    for name, a, b, c, d in configurations:
        print(f"\n--- {name} ---")
        print(f"Links: a={a}, b={b}, c={c}, d={d}")

        lengths = sorted([a, b, c, d])
        shortest, mid1, mid2, longest = lengths
        print(f"s={shortest:.1f}, l={longest:.1f}, p={mid1:.1f}, q={mid2:.1f}")
        print(f"s+l = {shortest + longest:.1f}, p+q = {mid1 + mid2:.1f}")

        grashof_type = grashof_check(a, b, c, d)
        print(f"Type: {grashof_type.name}")

        try:
            _linkage = fourbar_from_lengths(
                crank_length=a,
                coupler_length=b,
                rocker_length=c,
                ground_length=d,
            )
            print("Linkage created successfully")
            del _linkage  # Created to verify assembly, not needed further
        except ValueError as e:
            print(f"Cannot create: {e}")


def demo_visualize_grashof_types():
    """Visualize different Grashof types."""
    print("\n" + "=" * 60)
    print("Demo 3: Visualizing Grashof Types")
    print("=" * 60)

    print("\n1. Crank-Rocker (most common industrial mechanism)")
    linkage_cr = fourbar_from_lengths(1.0, 4.0, 3.0, 4.0)
    print("   Crank can rotate fully, rocker oscillates")
    pl.show_linkage(linkage_cr)

    print("\n2. Double-Crank (Drag-Link)")
    linkage_dc = fourbar_from_lengths(2.0, 4.0, 3.0, 1.0)
    print("   Both crank and rocker can rotate fully")
    pl.show_linkage(linkage_dc)

    print("\n3. Double-Rocker (Non-Grashof)")
    linkage_dr = fourbar_from_lengths(3.0, 4.0, 3.0, 2.0)
    print("   Neither crank nor rocker can rotate fully")
    pl.show_linkage(linkage_dr)


def demo_transmission_angle():
    """Analyze transmission angle for mechanism quality."""
    print("\n" + "=" * 60)
    print("Demo 4: Transmission Angle Analysis")
    print("=" * 60)

    print("\nThe transmission angle (mu) affects force transmission quality.")
    print("Ideal range: 40 deg < mu < 140 deg")
    print("Best performance: mu = 90 deg")

    # Good transmission angle design
    linkage = fourbar_from_lengths(
        crank_length=1.0,
        coupler_length=3.0,
        rocker_length=3.0,
        ground_length=4.0,
    )

    # Get parameters back
    params = linkage_to_synthesis_params(linkage)

    print("\nGood design - Link lengths:")
    print(f"  Crank: {params.crank_length:.2f}")
    print(f"  Coupler: {params.coupler_length:.2f}")
    print(f"  Rocker: {params.rocker_length:.2f}")
    print(f"  Ground: {params.ground_length:.2f}")

    # Calculate transmission angle at current position
    # (simplified - full analysis would track through cycle)
    B = params.crank_pivot_b
    C = params.coupler_pivot_c
    D = params.ground_pivot_d

    # Vector from B to C (coupler)
    BC = (C[0] - B[0], C[1] - B[1])
    # Vector from D to C (rocker)
    DC = (C[0] - D[0], C[1] - D[1])

    # Transmission angle
    dot = BC[0] * DC[0] + BC[1] * DC[1]
    mag_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    mag_DC = math.sqrt(DC[0] ** 2 + DC[1] ** 2)

    if mag_BC > 0 and mag_DC > 0:
        cos_mu = dot / (mag_BC * mag_DC)
        cos_mu = max(-1, min(1, cos_mu))  # Clamp for numerical stability
        mu = math.acos(cos_mu)
        print(f"\nTransmission angle at initial position: {math.degrees(mu):.1f} deg")

    print("\nVisualizing mechanism...")
    pl.show_linkage(linkage)


def demo_mechanical_advantage():
    """Demonstrate different mechanical advantage configurations."""
    print("\n" + "=" * 60)
    print("Demo 5: Mechanical Advantage Variations")
    print("=" * 60)

    print("\nMechanical advantage depends on link length ratios.")

    configurations = [
        ("High torque (long rocker)", 1.0, 3.0, 4.0, 4.0),
        ("Balanced", 1.5, 3.0, 2.5, 4.0),
        ("High speed (short rocker)", 2.0, 3.0, 1.5, 4.0),
    ]

    for name, a, b, c, d in configurations:
        print(f"\n{name}:")
        print(f"  Crank/Rocker ratio: {a / c:.2f}")

        grashof_type = grashof_check(a, b, c, d)
        print(f"  Grashof type: {grashof_type.name}")

        try:
            linkage = fourbar_from_lengths(a, b, c, d)
            print("  Visualizing...")
            pl.show_linkage(linkage)
        except ValueError as e:
            print(f"  Cannot assemble: {e}")


def demo_different_initial_angles():
    """Show the same linkage starting at different crank angles."""
    print("\n" + "=" * 60)
    print("Demo 6: Different Initial Crank Angles")
    print("=" * 60)

    crank, coupler, rocker, ground = 1.0, 3.0, 2.5, 3.5

    print(f"\nSame linkage (a={crank}, b={coupler}, c={rocker}, d={ground})")
    print("at different starting positions:\n")

    angles = [0, math.pi / 4, math.pi / 2, math.pi]
    for angle in angles:
        print(f"Initial crank angle: {math.degrees(angle):.0f} deg")
        try:
            linkage = fourbar_from_lengths(
                crank_length=crank,
                coupler_length=coupler,
                rocker_length=rocker,
                ground_length=ground,
                initial_crank_angle=angle,
            )
            pl.show_linkage(linkage)
        except ValueError as e:
            print(f"  Cannot assemble at this angle: {e}")


def demo_extract_parameters():
    """Show how to extract parameters from an existing linkage."""
    print("\n" + "=" * 60)
    print("Demo 7: Extracting Parameters from Linkage")
    print("=" * 60)

    # Create a linkage
    original = fourbar_from_lengths(
        crank_length=1.5,
        coupler_length=3.5,
        rocker_length=2.5,
        ground_length=4.0,
        ground_pivot_a=(1.0, 0.5),
        initial_crank_angle=math.pi / 3,
    )

    print("\nOriginal linkage created with:")
    print("  Crank: 1.5, Coupler: 3.5, Rocker: 2.5, Ground: 4.0")
    print("  Ground pivot A: (1.0, 0.5)")
    print("  Initial angle: 60 deg")

    # Extract parameters
    params = linkage_to_synthesis_params(original)

    print("\nExtracted parameters:")
    print(f"  Ground pivot A: ({params.ground_pivot_a[0]:.3f}, {params.ground_pivot_a[1]:.3f})")
    print(f"  Ground pivot D: ({params.ground_pivot_d[0]:.3f}, {params.ground_pivot_d[1]:.3f})")
    print(f"  Crank pivot B: ({params.crank_pivot_b[0]:.3f}, {params.crank_pivot_b[1]:.3f})")
    print(f"  Coupler pivot C: ({params.coupler_pivot_c[0]:.3f}, {params.coupler_pivot_c[1]:.3f})")
    print(f"  Crank length: {params.crank_length:.3f}")
    print(f"  Coupler length: {params.coupler_length:.3f}")
    print(f"  Rocker length: {params.rocker_length:.3f}")
    print(f"  Ground length: {params.ground_length:.3f}")

    print("\nVisualizing...")
    pl.show_linkage(original)


def demo_classic_linkages():
    """Demonstrate classic four-bar linkage designs."""
    print("\n" + "=" * 60)
    print("Demo 8: Classic Four-Bar Linkages")
    print("=" * 60)

    classics = [
        # Approximate proportions for classic mechanisms
        ("Parallelogram (equal opposite sides)", 2.0, 3.0, 2.0, 3.0),
        ("Antiparallelogram (crossed)", 2.0, 3.0, 2.0, 3.0),  # Same lengths, different config
        ("Isosceles trapezoid", 2.0, 3.0, 2.0, 2.5),
    ]

    for name, a, b, c, d in classics:
        print(f"\n{name}:")
        print(f"  Links: a={a}, b={b}, c={c}, d={d}")

        grashof_type = grashof_check(a, b, c, d)
        print(f"  Grashof type: {grashof_type.name}")

        try:
            linkage = fourbar_from_lengths(a, b, c, d)
            pl.show_linkage(linkage)
        except ValueError as e:
            print(f"  Cannot assemble: {e}")


def main():
    """Run all four-bar from lengths demos."""
    print("\n" + "#" * 60)
    print("# FOUR-BAR LINKAGE FROM LINK LENGTHS DEMOS")
    print("# Direct Construction and Kinematic Analysis")
    print("#" * 60)

    demo_basic_fourbar()
    demo_grashof_types()
    demo_visualize_grashof_types()
    demo_transmission_angle()
    demo_mechanical_advantage()
    demo_different_initial_angles()
    demo_extract_parameters()
    demo_classic_linkages()

    print("\n" + "=" * 60)
    print("Four-bar from lengths demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
