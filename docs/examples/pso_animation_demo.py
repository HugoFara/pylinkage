#!/usr/bin/env python3
"""
Generate animated PSO visualizations and save them as video files.

Run with: uv run python docs/examples/pso_animation_demo.py
"""

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for saving

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import pylinkage as pl
from pylinkage.visualizer.pso_plots import (
    dashboard_layout,
    parallel_coordinates_plot,
)

# Simulation parameters
LAP_POINTS = 10

# Dimension names and types
DIM_NAMES = (
    "triangle", "aperture", "femur", "rockerL",
    "rockerS", "phi", "tibia", "f",
)

DIM_TYPES = (
    "length", "angle", "length", "length",
    "length", "angle", "length", "length",
)

DIMENSIONS = (
    2, np.pi / 4, 1.8, 2.6,
    1.4, np.pi + 0.2, 2.5, 1.8,
)

BOUNDS = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6),
)

INIT_COORD = (
    (0, 0), (0, 1), (1.41, 1.41), (-1.41, 1.41), (0, -1),
    (-2.25, 0), (2.25, 0), (-1.4, -1.2), (1.4, -1.2),
    (-2.7, -2.7), (2.7, -2.7),
)


def param2dimensions(param=DIMENSIONS, flat=False):
    """Expand dimensions to fit in strider.set_num_constraints."""
    out = (
        (), (),
        (param[0], -param[1]), (param[0], param[1]),
        (1,),
        (param[2], param[3]), (param[2], param[3]),
        (param[4], -param[5]), (param[4], param[5]),
        (param[6], param[7]), (param[6], param[7]),
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
    linkage.update({
        "B": pl.Fixed(joint0=linkage["A"], joint1=linkage["Y"], name="Frame right (B)"),
        "B_p": pl.Fixed(joint0=linkage["A"], joint1=linkage["Y"], name="Frame left (B_p)"),
        "C": pl.Crank(joint0=linkage["A"], angle=-2 * np.pi / LAP_POINTS, name="Crank link (C)"),
    })
    linkage.update({
        "D": pl.Revolute(joint0=linkage["B_p"], joint1=linkage["C"], name="Left knee link (D)"),
        "E": pl.Revolute(joint0=linkage["B"], joint1=linkage["C"], name="Right knee link (E)"),
    })
    linkage.update({
        "F": pl.Fixed(joint0=linkage["C"], joint1=linkage["E"], name="Left ankle link (F)"),
        "G": pl.Fixed(joint0=linkage["C"], joint1=linkage["D"], name="Right ankle link (G)"),
    })
    linkage.update({
        "H": pl.Revolute(joint0=linkage["D"], joint1=linkage["F"], name="Left foot (H)"),
        "I": pl.Revolute(joint0=linkage["E"], joint1=linkage["G"], name="Right foot (I)"),
    })
    strider = pl.Linkage(joints=linkage.values(), order=linkage.values(), name="Strider")
    strider.set_coords(prev)
    strider.set_num_constraints(constraints, flat=False)
    return strider


def sym_stride_evaluator(linkage, dimensions, initial_positions):
    """Evaluate the stride length of the linkage."""
    linkage.set_completely(param2dimensions(dimensions, flat=True), initial_positions)
    points = 12
    try:
        loci = tuple(map(tuple, linkage.step(iterations=points, dt=LAP_POINTS / points)))
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
        (i, history[i * n_particles : (i + 1) * n_particles])
        for i in range(n_iterations)
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
    animation.save(output_path, writer='pillow', fps=4)
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
            score_history=score_history[:frame_idx + 1],
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
    animation.save(output_path, writer='pillow', fps=3)
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
