#!/usr/bin/env python3
"""
Advanced visualization for Particle Swarm Optimization.

Provides parallel coordinates plots and dashboard layouts for visualizing
PSO optimization with proper handling of different data types (lengths,
angles, scores).

Created for improved PSO visualization in pylinkage.
"""


from typing import TYPE_CHECKING

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..exceptions import UnbuildableError
from .static import plot_static_linkage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .._types import Coord
    from ..linkage.linkage import Linkage

    # Agent type: (score, dimensions, initial_positions)
    Agent = tuple[float, Sequence[float], Sequence[Coord]]
    # Swarm type: (iteration, list of agents)
    Swarm = tuple[int, Sequence[Agent]]
    # History type: list of swarms
    History = list[Swarm]


def normalize_data(
    data: np.ndarray,
    bounds: "tuple[Sequence[float], Sequence[float]] | None" = None,
) -> np.ndarray:
    """Normalize data to [0, 1] range.

    Args:
        data: Array of shape (n_particles, n_dimensions).
        bounds: Optional (min_bounds, max_bounds) for each dimension.
            If None, uses data min/max.

    Returns:
        Normalized data array.
    """
    if bounds is not None:
        mins = np.array(bounds[0])
        maxs = np.array(bounds[1])
    else:
        mins = data.min(axis=0)
        maxs = data.max(axis=0)

    # Avoid division by zero
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    result: np.ndarray = (data - mins) / ranges
    return result


def parallel_coordinates_plot(
    swarm: "Swarm",
    dim_names: "Sequence[str]",
    dim_types: "Sequence[str] | None" = None,
    bounds: "tuple[Sequence[float], Sequence[float]] | None" = None,
    ax: "Axes | None" = None,
    cbar_ax: "Axes | None" = None,
    cmap: str = "viridis",
    alpha: float = 0.3,
    highlight_best: int = 5,
) -> "Axes":
    """Create a parallel coordinates plot for a PSO swarm.

    Each dimension gets its own vertical axis, normalized to [0, 1].
    Particles are colored by their score (fitness).

    Args:
        swarm: Tuple of (iteration, list_of_agents) where each agent is
            (score, dimensions, initial_positions).
        dim_names: Names for each dimension.
        dim_types: Type of each dimension ('length', 'angle', or 'score').
            Used for axis grouping and labeling.
        bounds: Optional (min_bounds, max_bounds) for normalization.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        cbar_ax: Optional axes for the colorbar. If provided, colorbar is
            drawn there instead of stealing space from ax. Use this for
            animations to prevent layout shifts.
        cmap: Colormap name for score coloring.
        alpha: Line transparency for regular particles.
        highlight_best: Number of best particles to highlight.

    Returns:
        The matplotlib Axes object.
    """
    iteration, agents = swarm

    if not agents:
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        return ax

    # Extract data
    scores = np.array([agent[0] for agent in agents])
    dimensions = np.array([list(agent[1]) for agent in agents])
    n_dims = dimensions.shape[1]

    # Add score as the last column for naming
    all_names = list(dim_names) + ["score"]

    if dim_types is None:
        dim_types = ["length"] * n_dims + ["score"]
    else:
        dim_types = list(dim_types) + ["score"]

    # Normalize dimensions (not score, handle separately)
    if bounds is not None:
        norm_dims = normalize_data(dimensions, bounds)
    else:
        norm_dims = normalize_data(dimensions)

    # Normalize scores
    score_min, score_max = scores.min(), scores.max()
    if score_max - score_min > 0:
        norm_scores = (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(scores) * 0.5

    norm_data = np.column_stack([norm_dims, norm_scores])

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure  # type: ignore[assignment]

    # Setup color mapping based on score
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # X positions for each dimension axis
    x_positions = np.arange(len(all_names))

    # Sort by score to draw best on top
    sorted_indices = np.argsort(scores)

    # Draw lines for each particle
    for idx in sorted_indices[:-highlight_best]:
        color = sm.to_rgba(scores[idx])
        ax.plot(x_positions, norm_data[idx], color=color, alpha=alpha, lw=0.8)

    # Highlight best particles
    for idx in sorted_indices[-highlight_best:]:
        color = sm.to_rgba(scores[idx])
        ax.plot(
            x_positions,
            norm_data[idx],
            color=color,
            alpha=0.9,
            lw=2,
            zorder=10,
        )

    # Draw vertical axis lines
    for i, x in enumerate(x_positions):
        ax.axvline(float(x), color="gray", lw=1, alpha=0.5)

        # Add tick labels showing actual range
        if i < n_dims:
            if bounds is not None:
                low, high = bounds[0][i], bounds[1][i]
            else:
                low, high = dimensions[:, i].min(), dimensions[:, i].max()
        else:
            low, high = score_min, score_max

        # Format based on type
        if dim_types[i] == "angle":
            low_label = f"{np.degrees(low):.0f}deg"
            high_label = f"{np.degrees(high):.0f}deg"
        else:
            low_label = f"{low:.2f}"
            high_label = f"{high:.2f}"

        ax.text(float(x), -0.08, low_label, ha="center", va="top", fontsize=8)
        ax.text(float(x), 1.08, high_label, ha="center", va="bottom", fontsize=8)

    # Color-code axis labels by type
    type_colors = {"length": "blue", "angle": "orange", "score": "green"}
    for i, (name, dtype) in enumerate(zip(all_names, dim_types, strict=True)):
        label_color = type_colors.get(dtype, "black")
        ax.text(
            i,
            -0.18,
            name,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=label_color,
        )

    # Styling
    ax.set_xlim(-0.5, len(all_names) - 0.5)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xticks([])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Iteration {iteration} | Best score: {scores.max():.4f}")

    # Add colorbar
    if cbar_ax is not None:
        # Use dedicated colorbar axes (for animations)
        cbar_ax.clear()
        cbar = fig.colorbar(sm, cax=cbar_ax)
    else:
        # Create colorbar next to main axes (for static plots)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score")

    # Add legend for dimension types
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Length"),
        Line2D([0], [0], color="orange", lw=2, label="Angle"),
        Line2D([0], [0], color="green", lw=2, label="Score"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    return ax


def animate_parallel_coordinates(
    history: "History",
    dim_names: "Sequence[str]",
    dim_types: "Sequence[str] | None" = None,
    bounds: "tuple[Sequence[float], Sequence[float]] | None" = None,
    interval: int = 200,
    cmap: str = "viridis",
    save_path: str | None = None,
) -> anim.FuncAnimation:
    """Animate parallel coordinates plot over optimization history.

    Args:
        history: List of swarms, one per iteration.
        dim_names: Names for each dimension.
        dim_types: Type of each dimension ('length', 'angle', or 'score').
        bounds: Optional (min_bounds, max_bounds) for normalization.
        interval: Delay between frames in milliseconds.
        cmap: Colormap name.
        save_path: If provided, save animation to this path.

    Returns:
        The animation object.
    """
    # Create figure with fixed layout: main axes + colorbar axes
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    def update(frame_idx: int) -> None:
        ax.clear()
        parallel_coordinates_plot(
            history[frame_idx],
            dim_names=dim_names,
            dim_types=dim_types,
            bounds=bounds,
            ax=ax,
            cbar_ax=cbar_ax,
            cmap=cmap,
        )

    animation = anim.FuncAnimation(
        fig,
        update,  # type: ignore[arg-type]
        frames=len(history),
        interval=interval,
        repeat=True,
    )

    if save_path:
        writer = anim.FFMpegWriter(fps=24, bitrate=1800)
        animation.save(save_path, writer=writer)

    plt.show(block=False)
    return animation


def dashboard_layout(
    linkage: "Linkage",
    swarm: "Swarm",
    score_history: "Sequence[float]",
    dim_names: "Sequence[str]",
    dim_types: "Sequence[str] | None" = None,
    bounds: "tuple[Sequence[float], Sequence[float]] | None" = None,
    dimension_func: "Callable[[np.ndarray], Sequence[float]] | None" = None,
    fig: "Figure | None" = None,
) -> "Figure":
    """Create a dashboard layout for PSO visualization.

    Layout:
    +-----------------+------------------+
    | Score History   | Best Linkage     |
    | (line plot)     | (static plot)    |
    +-----------------+------------------+
    | Length Params   | Angle Params     |
    | (box plot)      | (polar/circular) |
    +-----------------+------------------+

    Args:
        linkage: The linkage being optimized (will be modified).
        swarm: Current swarm state.
        score_history: History of best scores per iteration.
        dim_names: Names for each dimension.
        dim_types: Type of each dimension ('length' or 'angle').
        bounds: Optional (min_bounds, max_bounds).
        dimension_func: Optional function to transform dimensions.
        fig: Existing figure to use.

    Returns:
        The matplotlib Figure object.
    """
    iteration, agents = swarm

    if not agents:
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        return fig

    # Extract data
    scores = np.array([agent[0] for agent in agents])
    dimensions = np.array([list(agent[1]) for agent in agents])
    n_dims = dimensions.shape[1]

    # Default dim_types if not provided
    if dim_types is None:
        dim_types = ["length"] * n_dims

    # Separate length and angle dimensions
    length_indices = [i for i, t in enumerate(dim_types) if t == "length"]
    angle_indices = [i for i, t in enumerate(dim_types) if t == "angle"]

    # Create figure
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    fig.clear()

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # === Panel 1: Score History (top-left) ===
    ax_score = fig.add_subplot(gs[0, 0])
    ax_score.plot(score_history, "b-", lw=2, label="Best score")
    ax_score.scatter(
        len(score_history) - 1, score_history[-1], color="red", s=100, zorder=5
    )
    ax_score.set_xlabel("Iteration")
    ax_score.set_ylabel("Score")
    ax_score.set_title("Optimization Progress")
    ax_score.grid(True, alpha=0.3)
    ax_score.legend()

    # Add current stats
    if len(score_history) > 1:
        improvement = score_history[-1] - score_history[0]
        ax_score.text(
            0.02,
            0.98,
            f"Improvement: {improvement:+.4f}",
            transform=ax_score.transAxes,
            va="top",
            fontsize=9,
        )

    # === Panel 2: Best Linkage Visualization (top-right) ===
    ax_linkage = fig.add_subplot(gs[0, 1])

    # Find best agent
    best_idx = int(np.argmax(scores))
    best_agent = agents[best_idx]

    # Configure and simulate linkage
    if dimension_func is None:
        linkage.set_num_constraints(best_agent[1])
    else:
        linkage.set_num_constraints(dimension_func(np.array(best_agent[1])))
    linkage.set_coords(best_agent[2])

    try:
        loci = tuple(tuple(pos) for pos in linkage.step(iterations=50, dt=1))
        plot_static_linkage(linkage, ax_linkage, loci)  # type: ignore[arg-type]
        ax_linkage.set_title(f"Best Linkage (score: {best_agent[0]:.4f})")
    except UnbuildableError:
        ax_linkage.text(
            0.5, 0.5, "Unbuildable", ha="center", va="center", fontsize=14
        )
        ax_linkage.set_title("Best Linkage (unbuildable)")

    ax_linkage.set_aspect("equal")

    # === Panel 3: Length Parameters (bottom-left) ===
    ax_lengths = fig.add_subplot(gs[1, 0])

    if length_indices:
        length_data = dimensions[:, length_indices]
        length_names = [dim_names[i] for i in length_indices]

        # Box plot for lengths
        bp = ax_lengths.boxplot(
            length_data, tick_labels=length_names, patch_artist=True
        )

        # Color boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        # Add scatter for best agent
        best_lengths = [best_agent[1][i] for i in length_indices]
        ax_lengths.scatter(
            range(1, len(length_indices) + 1),
            best_lengths,
            color="red",
            s=100,
            zorder=5,
            label="Best",
            marker="*",
        )

        # Add bounds if available
        if bounds is not None:
            for j, i in enumerate(length_indices):
                ax_lengths.hlines(
                    bounds[0][i],
                    j + 0.7,
                    j + 1.3,
                    colors="gray",
                    linestyles="dashed",
                    alpha=0.5,
                )
                ax_lengths.hlines(
                    bounds[1][i],
                    j + 0.7,
                    j + 1.3,
                    colors="gray",
                    linestyles="dashed",
                    alpha=0.5,
                )

        ax_lengths.set_ylabel("Value")
        ax_lengths.legend(loc="upper right")
    else:
        ax_lengths.text(0.5, 0.5, "No length parameters", ha="center", va="center")

    ax_lengths.set_title("Length Parameters Distribution")
    ax_lengths.tick_params(axis="x", rotation=45)

    # === Panel 4: Angle Parameters (bottom-right) ===
    ax_angles = fig.add_subplot(gs[1, 1], projection="polar")

    if angle_indices:
        angle_names = [dim_names[i] for i in angle_indices]
        n_angles = len(angle_indices)

        # Create angular positions for each angle parameter
        theta_positions = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Plot each particle's angles
        for agent in agents:
            angles = [agent[1][i] for i in angle_indices]
            # Normalize angles to [0, 1] for radial display
            if bounds is not None:
                norm_angles = [
                    (a - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
                    for a, i in zip(angles, angle_indices, strict=True)
                ]
            else:
                norm_angles = [a / (2 * np.pi) for a in angles]

            ax_angles.plot(
                theta_positions,
                norm_angles,
                "o-",
                alpha=0.1,
                color="blue",
                markersize=3,
            )

        # Highlight best agent
        best_angles = [best_agent[1][i] for i in angle_indices]
        if bounds is not None:
            best_norm_angles = [
                (a - bounds[0][i]) / (bounds[1][i] - bounds[0][i])
                for a, i in zip(best_angles, angle_indices, strict=True)
            ]
        else:
            best_norm_angles = [a / (2 * np.pi) for a in best_angles]

        ax_angles.plot(
            theta_positions,
            best_norm_angles,
            "o-",
            color="red",
            lw=2,
            markersize=8,
            label="Best",
        )

        # Set labels
        ax_angles.set_xticks(theta_positions)
        ax_angles.set_xticklabels(angle_names)
        ax_angles.set_ylim(0, 1)
        ax_angles.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_angles.set_yticklabels(["25%", "50%", "75%", "100%"])
        ax_angles.legend(loc="upper right")
    else:
        ax_angles.text(
            0,
            0.5,
            "No angle parameters",
            ha="center",
            va="center",
            transform=ax_angles.transAxes,
        )

    ax_angles.set_title("Angle Parameters (normalized)")

    # Main title
    fig.suptitle(
        f"PSO Dashboard - Iteration {iteration} | "
        f"Particles: {len(agents)} | Best: {scores.max():.4f}",
        fontsize=14,
        fontweight="bold",
    )

    return fig


def animate_dashboard(
    linkage: "Linkage",
    history: "History",
    dim_names: "Sequence[str]",
    dim_types: "Sequence[str] | None" = None,
    bounds: "tuple[Sequence[float], Sequence[float]] | None" = None,
    dimension_func: "Callable[[np.ndarray], Sequence[float]] | None" = None,
    interval: int = 500,
    save_path: str | None = None,
) -> anim.FuncAnimation:
    """Animate the dashboard layout over optimization history.

    Args:
        linkage: The linkage being optimized.
        history: List of swarms, one per iteration.
        dim_names: Names for each dimension.
        dim_types: Type of each dimension ('length' or 'angle').
        bounds: Optional (min_bounds, max_bounds).
        dimension_func: Optional function to transform dimensions.
        interval: Delay between frames in milliseconds.
        save_path: If provided, save animation to this path.

    Returns:
        The animation object.
    """
    fig = plt.figure(figsize=(14, 10))

    # Track best score history
    score_history: list[float] = []

    def update(frame_idx: int) -> None:
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
            dim_names=dim_names,
            dim_types=dim_types,
            bounds=bounds,
            dimension_func=dimension_func,
            fig=fig,
        )

    animation = anim.FuncAnimation(
        fig,
        update,  # type: ignore[arg-type]
        frames=len(history),
        interval=interval,
        repeat=True,
    )

    if save_path:
        writer = anim.FFMpegWriter(fps=24, bitrate=1800)
        animation.save(save_path, writer=writer)

    plt.show(block=False)
    return animation
