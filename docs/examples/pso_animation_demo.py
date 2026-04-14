#!/usr/bin/env python3
"""
Generate animated PSO visualizations and save them as video files.

Run with: uv run python docs/examples/pso_animation_demo.py
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import pylinkage as pl
from pylinkage.visualizer.pso_plots import (
    dashboard_layout,
    parallel_coordinates_plot,
)

# Reuse the strider builder / evaluator migrated to the modern API.
import importlib.util
import pathlib as _pathlib

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
    "length",
    "angle",
    "length",
    "length",
    "length",
    "angle",
    "length",
    "length",
)


def run_optimization(linkage, n_particles=50, n_iterations=40):
    """Run PSO optimization and return formatted history."""
    print(f"Running PSO with {n_particles} particles for {n_iterations} iterations...")
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


def create_parallel_coordinates_animation(history, output_path="parallel_coords_animation.gif"):
    """Create and save parallel coordinates animation."""
    print("Creating parallel coordinates animation...")

    # Create figure with fixed layout: main axes + colorbar axes
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    def update(frame_idx):
        ax.clear()
        parallel_coordinates_plot(
            swarm=history[frame_idx],
            dim_names=DIM_NAMES,
            dim_types=DIM_TYPES,
            bounds=BOUNDS,
            ax=ax,
            cbar_ax=cbar_ax,
            highlight_best=5,
        )
        return []

    animation = anim.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=250,
        repeat=True,
    )

    print(f"Saving to {output_path}...")
    animation.save(output_path, writer="pillow", fps=4)
    plt.close(fig)
    print(f"Saved: {output_path}")

    return output_path


def create_dashboard_animation(linkage, history, output_path="dashboard_animation.gif"):
    """Create and save dashboard animation."""
    print("Creating dashboard animation...")

    fig = plt.figure(figsize=(14, 10))
    score_history = []

    def update(frame_idx):
        swarm = history[frame_idx]
        _, agents = swarm

        # Update score history
        if agents:
            best_score = max(agent[0] for agent in agents)
            while len(score_history) <= frame_idx:
                score_history.append(best_score)
            score_history[frame_idx] = best_score

        dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history[: frame_idx + 1],
            dim_names=DIM_NAMES,
            dim_types=DIM_TYPES,
            bounds=BOUNDS,
            dimension_func=lambda dims: param2dimensions(dims, flat=True),
            fig=fig,
        )

        return []

    animation = anim.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=300,
        repeat=True,
    )

    print(f"Saving to {output_path}...")
    animation.save(output_path, writer="pillow", fps=3)
    plt.close(fig)
    print(f"Saved: {output_path}")

    return output_path


def main():
    print("=" * 60)
    print("PSO Animation Generator")
    print("=" * 60)

    # Create linkage and run optimization
    linkage = complete_strider(param2dimensions(), INIT_COORD)
    history = run_optimization(linkage, n_particles=50, n_iterations=40)

    # Generate parallel coordinates animation
    create_parallel_coordinates_animation(history, "parallel_coords_animation.gif")

    # Recreate linkage (it gets modified during visualization)
    linkage = complete_strider(param2dimensions(), INIT_COORD)

    # Generate dashboard animation
    create_dashboard_animation(linkage, history, "dashboard_animation.gif")

    print("\n" + "=" * 60)
    print("Animations complete!")
    print("Files created:")
    print("  - parallel_coords_animation.gif")
    print("  - dashboard_animation.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()
