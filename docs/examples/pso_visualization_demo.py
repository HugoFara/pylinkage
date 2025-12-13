#!/usr/bin/env python3
"""
Demo script for the new PSO visualization approaches.

Demonstrates:
1. Parallel Coordinates Plot - normalized multi-dimensional view
2. Dashboard Layout - comprehensive monitoring view

Run with: uv run python docs/examples/pso_visualization_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

import pylinkage as pl
from pylinkage.visualizer.pso_plots import (
    animate_dashboard,
    animate_parallel_coordinates,
    dashboard_layout,
    parallel_coordinates_plot,
)

# Simulation parameters
LAP_POINTS = 10

# Dimension names and types
DIM_NAMES = (
    "triangle",
    "aperture",
    "femur",
    "rockerL",
    "rockerS",
    "phi",
    "tibia",
    "f",
)

# Specify which dimensions are lengths vs angles
DIM_TYPES = (
    "length",  # triangle
    "angle",   # aperture
    "length",  # femur
    "length",  # rockerL
    "length",  # rockerS
    "angle",   # phi
    "length",  # tibia
    "length",  # f
)

# Starting dimensions
DIMENSIONS = (
    2,           # triangle (AB distance)
    np.pi / 4,   # aperture (angle)
    1.8,         # femur
    2.6,         # rockerL
    1.4,         # rockerS
    np.pi + 0.2, # phi (angle)
    2.5,         # tibia
    1.8,         # f
)

# Bounds for optimization
BOUNDS = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6),
)

# Initial coordinates
INIT_COORD = (
    (0, 0),
    (0, 1),
    (1.41, 1.41),
    (-1.41, 1.41),
    (0, -1),
    (-2.25, 0),
    (2.25, 0),
    (-1.4, -1.2),
    (1.4, -1.2),
    (-2.7, -2.7),
    (2.7, -2.7),
)


def param2dimensions(param=DIMENSIONS, flat=False):
    """Expand dimensions to fit in strider.set_num_constraints."""
    out = (
        (),
        (),
        (param[0], -param[1]),
        (param[0], param[1]),
        (1,),
        (param[2], param[3]),
        (param[2], param[3]),
        (param[4], -param[5]),
        (param[4], param[5]),
        (param[6], param[7]),
        (param[6], param[7]),
    )
    if not flat:
        return out
    flat_dims = []
    for constraint in out[2:]:
        flat_dims.extend(constraint)
    return tuple(flat_dims)


def complete_strider(constraints, prev):
    """Create a strider linkage."""
    linkage = {
        "A": pl.Static(x=0, y=0, name="A"),
        "Y": pl.Static(0, 1, name="Point (0, 1)"),
    }
    linkage["Y"].joint0 = linkage["A"]
    linkage.update(
        {
            "B": pl.Fixed(
                joint0=linkage["A"], joint1=linkage["Y"], name="Frame right (B)"
            ),
            "B_p": pl.Fixed(
                joint0=linkage["A"], joint1=linkage["Y"], name="Frame left (B_p)"
            ),
            "C": pl.Crank(
                joint0=linkage["A"],
                angle=-2 * np.pi / LAP_POINTS,
                name="Crank link (C)",
            ),
        }
    )
    linkage.update(
        {
            "D": pl.Revolute(
                joint0=linkage["B_p"], joint1=linkage["C"], name="Left knee link (D)"
            ),
            "E": pl.Revolute(
                joint0=linkage["B"], joint1=linkage["C"], name="Right knee link (E)"
            ),
        }
    )
    linkage.update(
        {
            "F": pl.Fixed(
                joint0=linkage["C"], joint1=linkage["E"], name="Left ankle link (F)"
            ),
            "G": pl.Fixed(
                joint0=linkage["C"], joint1=linkage["D"], name="Right ankle link (G)"
            ),
        }
    )
    linkage.update(
        {
            "H": pl.Revolute(
                joint0=linkage["D"], joint1=linkage["F"], name="Left foot (H)"
            ),
            "I": pl.Revolute(
                joint0=linkage["E"], joint1=linkage["G"], name="Right foot (I)"
            ),
        }
    )
    strider = pl.Linkage(
        joints=linkage.values(), order=linkage.values(), name="Strider"
    )
    strider.set_coords(prev)
    strider.set_num_constraints(constraints, flat=False)
    return strider


def sym_stride_evaluator(linkage, dimensions, initial_positions):
    """Evaluate the stride length of the linkage."""
    linkage.set_completely(param2dimensions(dimensions, flat=True), initial_positions)
    points = 12
    try:
        loci = tuple(
            map(
                tuple,
                linkage.step(iterations=points, dt=LAP_POINTS / points),
            )
        )
    except pl.UnbuildableError:
        return 0
    foot_locus = tuple(x[-2] for x in loci)
    score = max(k[0] for k in foot_locus) - min(k[0] for k in foot_locus)
    return score


def history_saver(evaluator, history, linkage, dims, pos):
    """Save optimization history."""
    score = evaluator(linkage, dims, pos)
    history.append((score, list(dims), pos))
    return score


def run_quick_optimization(linkage, n_particles=30, n_iterations=20):
    """Run a quick PSO optimization and return formatted history."""
    history = []

    pl.particle_swarm_optimization(
        lambda *x: history_saver(sym_stride_evaluator, history, *x),
        linkage,
        center=DIMENSIONS,
        n_particles=n_particles,
        iters=n_iterations,
        bounds=BOUNDS,
        dimensions=len(DIMENSIONS),
    )

    # Format history into swarms per iteration
    formatted_history = [
        (i, history[i * n_particles : (i + 1) * n_particles])
        for i in range(n_iterations)
    ]

    return formatted_history


def demo_parallel_coordinates_static():
    """Demonstrate static parallel coordinates plot."""
    print("Creating strider linkage...")
    linkage = complete_strider(param2dimensions(), INIT_COORD)

    print("Running quick optimization (30 particles, 20 iterations)...")
    history = run_quick_optimization(linkage, n_particles=30, n_iterations=20)

    print("\n=== Parallel Coordinates Plot ===")
    print("Shows all dimensions normalized to [0,1] range")
    print("- Blue labels = Length parameters")
    print("- Orange labels = Angle parameters")
    print("- Green label = Score")
    print("- Lines colored by score (yellow = best)")

    # Show final iteration
    fig, ax = plt.subplots(figsize=(14, 6))
    parallel_coordinates_plot(
        swarm=history[-1],
        dim_names=DIM_NAMES,
        dim_types=DIM_TYPES,
        bounds=BOUNDS,
        ax=ax,
        highlight_best=3,
    )
    plt.tight_layout()
    plt.savefig("parallel_coordinates_static.png", dpi=150)
    print("Saved: parallel_coordinates_static.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return history


def demo_parallel_coordinates_animated(history):
    """Demonstrate animated parallel coordinates plot."""
    print("\n=== Animated Parallel Coordinates ===")
    print("Watch how particles converge over iterations...")

    animation = animate_parallel_coordinates(
        history=history,
        dim_names=DIM_NAMES,
        dim_types=DIM_TYPES,
        bounds=BOUNDS,
        interval=300,
    )
    plt.pause(8)
    plt.close()

    return animation


def demo_dashboard_static(linkage, history):
    """Demonstrate static dashboard layout."""
    print("\n=== Dashboard Layout ===")
    print("Four-panel view:")
    print("- Top-left: Score history over iterations")
    print("- Top-right: Best linkage visualization")
    print("- Bottom-left: Length parameters distribution (box plot)")
    print("- Bottom-right: Angle parameters (polar plot)")

    # Compute score history
    score_history = [max(agent[0] for agent in swarm[1]) for swarm in history]

    fig = dashboard_layout(
        linkage=linkage,
        swarm=history[-1],
        score_history=score_history,
        dim_names=DIM_NAMES,
        dim_types=DIM_TYPES,
        bounds=BOUNDS,
        dimension_func=lambda dims: param2dimensions(dims, flat=True),
    )
    plt.savefig("dashboard_static.png", dpi=150)
    print("Saved: dashboard_static.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def demo_dashboard_animated(linkage, history):
    """Demonstrate animated dashboard layout."""
    print("\n=== Animated Dashboard ===")
    print("Watch the full optimization process...")

    animation = animate_dashboard(
        linkage=linkage,
        history=history,
        dim_names=DIM_NAMES,
        dim_types=DIM_TYPES,
        bounds=BOUNDS,
        dimension_func=lambda dims: param2dimensions(dims, flat=True),
        interval=400,
    )
    plt.pause(10)
    plt.close()

    return animation


def main():
    """Run all demos."""
    print("=" * 60)
    print("PSO Visualization Demo")
    print("=" * 60)

    # Create linkage
    linkage = complete_strider(param2dimensions(), INIT_COORD)

    # Run optimization and get history
    history = demo_parallel_coordinates_static()

    # Animated parallel coordinates
    demo_parallel_coordinates_animated(history)

    # Recreate linkage (it gets modified during visualization)
    linkage = complete_strider(param2dimensions(), INIT_COORD)

    # Static dashboard
    demo_dashboard_static(linkage, history)

    # Recreate linkage again
    linkage = complete_strider(param2dimensions(), INIT_COORD)

    # Animated dashboard
    demo_dashboard_animated(linkage, history)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
