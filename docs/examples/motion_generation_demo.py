#!/usr/bin/env python3
"""
Motion Generation (Rigid Body Guidance) Synthesis Demo.

This demo shows how to synthesize a four-bar linkage for motion generation
using Burmester theory. Motion generation creates a linkage where a body
attached to the coupler passes through specified poses (position + orientation).

To run:
    uv run python docs/examples/motion_generation_demo.py

Theory:
    Motion generation is the most constrained synthesis problem because
    both position AND orientation are specified at each precision position.

    Constraints:
    - 3 poses: Continuous curve of solutions (circular cubic)
    - 4 poses: Up to 6 discrete solutions (Ball's points)
    - 5 poses: Typically 0-2 solutions (over-constrained)
    - 6+ poses: Almost always no solution exists

Applications:
    - Robot end-effector positioning
    - Pick-and-place with orientation control
    - Assembly operations
    - Door/hatch mechanisms
    - Automotive hood linkages

References:
    - McCarthy, J.M. "Geometric Design of Linkages" Chapter 6
    - Sandor & Erdman, Chapter 5: "Path, Motion, and Function Generation"
    - Bottema & Roth, "Theoretical Kinematics"
"""

import math

import pylinkage as pl
from pylinkage.synthesis import (
    FourBarSolution,
    Pose,
    grashof_check,
    motion_generation,
    motion_generation_3_poses,
)


def demo_basic_motion_generation():
    """Basic example: synthesize a linkage for 3 poses.

    With 3 poses, there is a continuous curve of solutions,
    giving us flexibility in selecting favorable linkages.
    """
    print("=" * 60)
    print("Demo 1: Basic Motion Generation (3 Poses)")
    print("=" * 60)

    # Define body poses (position x, y and orientation angle)
    # Each pose specifies where the coupler body should be
    poses = [
        Pose(x=0.0, y=0.0, angle=0.0),  # Pose 1: origin, horizontal
        Pose(x=2.0, y=1.0, angle=math.pi / 6),  # Pose 2: moved right-up, 30 deg rotation
        Pose(x=3.0, y=0.5, angle=math.pi / 4),  # Pose 3: further right, 45 deg rotation
    ]

    print("\nPrecision poses (x, y, angle):")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.angle):.1f} deg)")

    # Perform synthesis
    result = motion_generation(
        poses,
        max_solutions=5,
        require_grashof=True,
    )

    print(f"\nFound {len(result)} solution(s)")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result:
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


def demo_3_pose_curve_sampling():
    """Sample multiple solutions from the 3-pose curve.

    For 3 poses, the solution set forms a continuous curve.
    This demo shows how to sample multiple linkages from that curve.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Sampling the 3-Pose Solution Curve")
    print("=" * 60)

    poses = [
        Pose(x=0.0, y=1.0, angle=0.0),
        Pose(x=1.5, y=1.5, angle=math.pi / 8),
        Pose(x=2.5, y=0.8, angle=math.pi / 6),
    ]

    print("\nPrecision poses:")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.angle):.1f} deg)")

    # Use specialized 3-pose synthesis for better sampling
    result = motion_generation_3_poses(
        poses,
        n_samples=48,
        max_solutions=10,
    )

    print(f"\nSampled {len(result)} solutions from the curve")

    if result:
        print("\nSolution diversity (showing 5):")
        print("-" * 70)
        print(f"{'#':<3} {'Crank':<10} {'Coupler':<10} {'Rocker':<10} {'Ground':<10} {'Type':<15}")
        print("-" * 70)

        for i, raw in enumerate(result.raw_solutions[:5], 1):
            grashof_type = grashof_check(
                raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
            )
            print(
                f"{i:<3} {raw.crank_length:<10.3f} {raw.coupler_length:<10.3f} "
                f"{raw.rocker_length:<10.3f} {raw.ground_length:<10.3f} {grashof_type.name:<15}"
            )

        print("-" * 70)

        print("\nVisualizing first sampled solution...")
        pl.show_linkage(result.solutions[0])


def demo_4_pose_synthesis():
    """Synthesize a linkage for 4 poses (Ball's points).

    With 4 poses, we get discrete solutions (up to 6 Ball's points).
    This is more constrained than 3 poses.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Four-Pose Motion Generation (Ball's Points)")
    print("=" * 60)

    poses = [
        Pose(x=0.0, y=0.0, angle=0.0),
        Pose(x=1.0, y=0.5, angle=0.2),
        Pose(x=2.0, y=0.8, angle=0.4),
        Pose(x=2.5, y=0.3, angle=0.5),
    ]

    print("\nFour precision poses:")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.angle):.1f} deg)")

    result = motion_generation(
        poses,
        max_solutions=10,
        require_grashof=False,  # Accept all types
    )

    print(f"\nFound {len(result)} Ball's point solution(s)")

    if result.warnings:
        print("Notes:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nBest solution:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing 4-pose mechanism...")
        pl.show_linkage(linkage)


def demo_pick_and_place():
    """Synthesize a pick-and-place mechanism.

    Pick-and-place requires the end-effector to move between
    positions while maintaining specific orientations.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Pick-and-Place Mechanism")
    print("=" * 60)

    # Pick-and-place: pick position, intermediate, place position
    poses = [
        Pose(x=0.0, y=2.0, angle=-math.pi / 6),  # Pick: angled down
        Pose(x=2.0, y=3.0, angle=0.0),  # Transfer: horizontal
        Pose(x=4.0, y=2.0, angle=math.pi / 6),  # Place: angled down other side
    ]

    print("\nPick-and-place poses:")
    print(
        f"  Pick:     ({poses[0].x:.1f}, {poses[0].y:.1f}, {math.degrees(poses[0].angle):.0f} deg)"
    )
    print(
        f"  Transfer: ({poses[1].x:.1f}, {poses[1].y:.1f}, {math.degrees(poses[1].angle):.0f} deg)"
    )
    print(
        f"  Place:    ({poses[2].x:.1f}, {poses[2].y:.1f}, {math.degrees(poses[2].angle):.0f} deg)"
    )

    result = motion_generation(
        poses,
        max_solutions=5,
        require_grashof=True,
    )

    print(f"\nFound {len(result)} pick-and-place mechanism(s)")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nPick-and-place linkage:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing pick-and-place mechanism...")
        pl.show_linkage(linkage)


def demo_door_hatch_mechanism():
    """Synthesize a door/hatch opening mechanism.

    Doors and hatches require both translation and rotation
    to open smoothly without interference.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Door/Hatch Opening Mechanism")
    print("=" * 60)

    # Door opening: closed, intermediate, open
    poses = [
        Pose(x=0.0, y=0.0, angle=0.0),  # Closed (vertical)
        Pose(x=0.5, y=1.0, angle=math.pi / 4),  # Partially open
        Pose(x=0.8, y=2.0, angle=math.pi / 2),  # Fully open (horizontal)
    ]

    print("\nDoor opening poses:")
    print(
        f"  Closed:    ({poses[0].x:.1f}, {poses[0].y:.1f}, {math.degrees(poses[0].angle):.0f} deg)"
    )
    print(
        f"  Partial:   ({poses[1].x:.1f}, {poses[1].y:.1f}, {math.degrees(poses[1].angle):.0f} deg)"
    )
    print(
        f"  Full open: ({poses[2].x:.1f}, {poses[2].y:.1f}, {math.degrees(poses[2].angle):.0f} deg)"
    )

    result = motion_generation(
        poses,
        max_solutions=5,
        require_grashof=False,  # Many door mechanisms are non-Grashof
    )

    print(f"\nFound {len(result)} door mechanism(s)")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        grashof_type = grashof_check(
            raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
        )

        print("\nDoor opening linkage:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")
        print(f"  Type: {grashof_type.name}")

        print("\nVisualizing door mechanism...")
        pl.show_linkage(linkage)


def demo_constrained_ground():
    """Motion generation with fixed ground pivot locations.

    In many practical designs, the frame attachment points
    are predetermined by the overall system design.
    """
    print("\n" + "=" * 60)
    print("Demo 6: Motion Generation with Ground Constraints")
    print("=" * 60)

    poses = [
        Pose(x=2.0, y=2.0, angle=0.0),
        Pose(x=3.0, y=2.5, angle=0.3),
        Pose(x=3.5, y=2.0, angle=0.5),
    ]

    # Constrained ground pivots
    ground_a = (0.0, 0.0)
    ground_d = (4.0, 0.0)

    print("\nPrecision poses:")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.angle):.1f} deg)")

    print("\nGround constraints:")
    print(f"  Pivot A: {ground_a}")
    print(f"  Pivot D: {ground_d}")

    result = motion_generation(
        poses,
        ground_pivot_a=ground_a,
        ground_pivot_d=ground_d,
        max_solutions=5,
        require_grashof=False,
    )

    print(f"\nFound {len(result)} solution(s) with fixed ground")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nConstrained solution:")
        print(f"  Pivot A: ({raw.ground_pivot_a[0]:.3f}, {raw.ground_pivot_a[1]:.3f})")
        print(f"  Pivot D: ({raw.ground_pivot_d[0]:.3f}, {raw.ground_pivot_d[1]:.3f})")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing constrained mechanism...")
        pl.show_linkage(linkage)
    else:
        print("No solutions found with the given ground constraints.")
        print("Try adjusting poses or removing ground constraints.")


def demo_5_pose_overconstrained():
    """Demonstrate the challenge of 5-pose synthesis.

    5 poses is typically overconstrained and may not have solutions.
    This demo shows how to handle this case.
    """
    print("\n" + "=" * 60)
    print("Demo 7: Five-Pose Synthesis (Overconstrained)")
    print("=" * 60)

    poses = [
        Pose(x=0.0, y=0.0, angle=0.0),
        Pose(x=0.5, y=0.3, angle=0.15),
        Pose(x=1.0, y=0.5, angle=0.3),
        Pose(x=1.5, y=0.4, angle=0.4),
        Pose(x=2.0, y=0.2, angle=0.5),
    ]

    print("\nFive precision poses:")
    for i, pose in enumerate(poses, 1):
        print(f"  Pose {i}: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.angle):.1f} deg)")

    print("\nNote: 5 poses is highly overconstrained.")
    print("Exact solutions may not exist.")

    result = motion_generation(
        poses,
        max_solutions=5,
        require_grashof=False,
    )

    print(f"\nFound {len(result)} solution(s)")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result:
        linkage = result.solutions[0]
        raw: FourBarSolution = result.raw_solutions[0]

        print("\nBest available solution:")
        print(f"  Crank: {raw.crank_length:.3f}, Coupler: {raw.coupler_length:.3f}")
        print(f"  Rocker: {raw.rocker_length:.3f}, Ground: {raw.ground_length:.3f}")

        print("\nVisualizing (may not pass through all poses exactly)...")
        pl.show_linkage(linkage)
    else:
        print("\nAs expected, no exact solution exists for these 5 poses.")
        print("Options:")
        print("  1. Reduce to 4 or fewer poses")
        print("  2. Adjust pose values to be more compatible")
        print("  3. Use optimization-based synthesis instead")


def main():
    """Run all motion generation demos."""
    print("\n" + "#" * 60)
    print("# MOTION GENERATION (RIGID BODY GUIDANCE) SYNTHESIS DEMOS")
    print("# Using Burmester Theory")
    print("#" * 60)

    demo_basic_motion_generation()
    demo_3_pose_curve_sampling()
    demo_4_pose_synthesis()
    demo_pick_and_place()
    demo_door_hatch_mechanism()
    demo_constrained_ground()
    demo_5_pose_overconstrained()

    print("\n" + "=" * 60)
    print("Motion generation demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
