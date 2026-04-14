#!/usr/bin/env python3
"""
Demo script for the new PSO visualization approaches.

Demonstrates:
1. Parallel Coordinates Plot - normalized multi-dimensional view
2. Dashboard Layout - comprehensive monitoring view

Run with: uv run python docs/examples/pso_visualization_demo.py
"""

import importlib.util
import pathlib as _pathlib

import matplotlib.pyplot as plt

import pylinkage as pl
from pylinkage.visualizer.pso_plots import (
    animate_dashboard,
    animate_parallel_coordinates,
    dashboard_layout,
    parallel_coordinates_plot,
)

# Reuse the strider builder / evaluator migrated to the modern API.
_strider_spec = importlib.util.spec_from_file_location(
    "_strider_demo", _pathlib.Path(__file__).with_name("strider.py"),
)
assert _strider_spec is not None and _strider_spec.loader is not None
_strider = importlib.util.module_from_spec(_strider_spec)
_strider_spec.loader.exec_module(_strider)

BOUNDS = _strider.BOUNDS
DIM_NAMES = _strider.DIM_NAMES
DIMENSIONS = _strider.DIMENSIONS
INIT_COORD = _strider.INIT_COORD
complete_strider = _strider.complete_strider
history_saver = _strider.history_saver
param2dimensions = _strider.param2dimensions
sym_stride_evaluator = _strider.sym_stride_evaluator

DIM_TYPES = (
    "length",  # triangle
    "angle",  # aperture
    "length",  # femur
    "length",  # rockerL
    "length",  # rockerS
    "angle",  # phi
    "length",  # tibia
    "length",  # f
)


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
        (i, history[i * n_particles : (i + 1) * n_particles]) for i in range(n_iterations)
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

    dashboard_layout(
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
