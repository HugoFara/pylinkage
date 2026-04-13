"""
Static (non-animated) linkage visualization.

Supports both legacy ``pylinkage.linkage.Linkage`` (joints API) and modern
``pylinkage.simulation.Linkage`` (components API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core import get_components, get_parent_pairs, resolve_component
from .symbols import LINK_COLORS, is_ground_joint

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes

    from .._types import Coord


def plot_static_linkage(
    linkage: Any,
    axis: Axes,
    loci: Iterable[tuple[Coord, ...]],
    locus_highlights: list[list[Coord]] | None = None,
    show_legend: bool = False,
    *,
    show_labels: bool = True,
    show_loci: bool = True,
    n_ghosts: int = 0,
    title: str | None = None,
) -> None:
    """Plot a linkage at one position with joint trajectories.

    Draws the mechanism bars at the initial position (bold, colored),
    joint trajectory paths (faded), ground pivot markers, and joint
    labels.  Optionally draws "ghost" mechanism outlines at evenly
    spaced frames through the cycle.

    Works with both the legacy ``Linkage`` (joints) and the modern
    ``SimLinkage`` (components) APIs.

    Args:
        linkage: The linkage to draw (legacy or modern).
        axis: Matplotlib axes to draw on.
        loci: Sequence of frames, each frame a tuple of (x, y) per joint.
        locus_highlights: Optional list of coordinate lists to scatter.
        show_legend: Add a legend with joint names.
        show_labels: Annotate each joint with its name.
        show_loci: Draw joint trajectory paths.
        n_ghosts: Number of ghost mechanism outlines to draw through
            the cycle (0 = none).
        title: Optional axes title.
    """
    loci_list = list(loci)
    components = get_components(linkage)
    n_joints = len(components)

    # --- Build connection list: (parent_idx, child_idx, link_index) ---
    connections: list[tuple[int, int]] = []
    for j, comp in enumerate(components):
        for parent in get_parent_pairs(comp):
            p = resolve_component(parent, components)
            if p is not None:
                connections.append((p, j))

    # --- Draw joint trajectories (faded) ---
    if show_loci and loci_list:
        for j in range(n_joints):
            if is_ground_joint(components[j]):
                continue
            xs = [frame[j][0] for frame in loci_list if frame[j][0] is not None]
            ys = [frame[j][1] for frame in loci_list if frame[j][1] is not None]
            if xs:
                color = LINK_COLORS[j % len(LINK_COLORS)]
                axis.plot(xs, ys, "-", color=color, linewidth=1, alpha=0.35)

    # --- Draw ghost mechanism outlines ---
    if n_ghosts > 0 and loci_list:
        ghost_indices = [
            int(i * len(loci_list) / n_ghosts)
            for i in range(n_ghosts)
        ]
        for gi, fi in enumerate(ghost_indices):
            alpha = 0.12 + 0.13 * (gi / max(n_ghosts - 1, 1))
            frame = loci_list[fi]
            for p, j in connections:
                if frame[p][0] is None or frame[j][0] is None:
                    continue
                axis.plot(
                    [frame[p][0], frame[j][0]],
                    [frame[p][1], frame[j][1]],
                    "-", color="#777777", linewidth=1.5, alpha=alpha,
                )

    # --- Draw mechanism bars at initial position ---
    if loci_list:
        frame0 = loci_list[0]
        for link_idx, (p, j) in enumerate(connections):
            if frame0[p][0] is None or frame0[j][0] is None:
                continue
            color = LINK_COLORS[link_idx % len(LINK_COLORS)]
            axis.plot(
                [frame0[p][0], frame0[j][0]],
                [frame0[p][1], frame0[j][1]],
                "-o", color=color, linewidth=3, markersize=5, zorder=4,
                label=components[j].name if show_legend else None,
            )

        # --- Draw ground pivot markers ---
        for j, comp in enumerate(components):
            if is_ground_joint(comp) and frame0[j][0] is not None:
                axis.plot(
                    frame0[j][0], frame0[j][1],
                    "ks", markersize=10, zorder=5,
                )

        # --- Draw joint labels ---
        if show_labels:
            for j, comp in enumerate(components):
                if frame0[j][0] is None:
                    continue
                name = getattr(comp, "name", None)
                if name:
                    axis.annotate(
                        name,
                        (frame0[j][0], frame0[j][1]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        fontweight="bold",
                    )

    # --- Highlight points ---
    if locus_highlights:
        for locus in locus_highlights:
            axis.scatter(
                [coord[0] for coord in locus],
                [coord[1] for coord in locus],
            )

    # --- Formatting ---
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    if title:
        axis.set_title(title, fontsize=11, fontweight="bold")
    if show_legend:
        if not title:
            axis.set_title("Static representation")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend(fontsize=8)
