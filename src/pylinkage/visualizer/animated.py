#!/usr/bin/env python3
"""
The visualizer module makes visualization of linkages easy using matplotlib.

Created on Mon Jun 14, 12:13:58 2021.

@author: HugoFara
"""


from typing import TYPE_CHECKING, Any

import matplotlib.animation as anim
import matplotlib.pyplot as plt

from ..exceptions import UnbuildableError
from ..joints import Crank, Static
from ..linkage.analysis import movement_bounding_box
from .core import _get_color
from .static import plot_static_linkage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from .._types import Coord
    from ..linkage.linkage import Linkage

# List of animations
ANIMATIONS: list[Any] = []


def update_animated_plot(
    linkage: "Linkage",
    index: int,
    images: "list[Line2D]",
    loci: "Sequence[tuple[Coord, ...]]",
) -> "list[Line2D]":
    """Modify im, instead of recreating it to make the animation run faster.

    Args:
        linkage: The linkage being animated.
        index: Frame index.
        images: Artist to be modified.
        loci: list of loci.

    Returns:
        Updated version of images.
    """
    image = iter(images)
    locus = loci[index]
    for j, pos in enumerate(locus):
        joint = linkage.joints[j]
        # Draw a link to the first parent if it exists
        if joint.joint0 is None:
            continue
        if isinstance(joint.joint0, Static):
            par_locus = joint.joint0.coord()
        else:
            par_locus = locus[linkage.joints.index(joint.joint0)]
        im = next(image)
        im.set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])  # type: ignore[arg-type]
        # Then second parent
        if isinstance(joint, (Crank, Static)):
            continue
        if isinstance(joint.joint1, Static):
            par_locus = joint.joint1.coord()
        else:
            par_locus = locus[linkage.joints.index(joint.joint1)]
        im = next(image)
        im.set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])  # type: ignore[arg-type]
    return images


def plot_kinematic_linkage(
    linkage: "Linkage",
    fig: "Figure",
    axis: "Axes",
    loci: "Sequence[tuple[Coord, ...]]",
    frames: int = 100,
    interval: float = 40,
) -> anim.FuncAnimation:
    """Plot a linkage with an animation.

    Args:
        linkage: The linkage to animate.
        fig: Figure to support the axes.
        axis: The subplot to draw on.
        loci: list of list of coordinates.
        frames: Number of frames to draw the linkage on.
        interval: Delay between frames in milliseconds.

    Returns:
        The animation object.
    """
    axis.set_aspect('equal')
    axis.set_title("Animation")

    images: list[Line2D] = []
    for joint in linkage.joints:
        for parent in (joint.joint0, joint.joint1):
            if parent is not None:
                images.append(axis.plot(
                    [], [], c=_get_color(joint),
                    animated=isinstance(joint, Static)
                )[0])

    animation = anim.FuncAnimation(
        fig=fig,
        func=lambda index: update_animated_plot(
            linkage, index % len(loci), images, loci
        ),
        frames=frames,
        blit=True,
        interval=interval,
        repeat=True
    )
    return animation


def show_linkage(
    linkage: "Linkage",
    save: bool = False,
    prev: "Sequence[Coord] | None" = None,
    loci: "Sequence[tuple[Coord, ...]] | None" = None,
    points: int = 100,
    iteration_factor: float = 1,
    title: str | None = None,
    duration: float = 5,
    fps: int = 24,
) -> anim.FuncAnimation:
    """Display results as an animated drawing.

    Args:
        linkage: The Linkage you want to draw.
        save: To save the animation.
        prev: Previous coordinates to use for linkage.
        loci: list of loci.
        points: Number of points to draw for a crank revolution.
            Useless when loci are set.
        iteration_factor: A simple way to subdivide the movement. The real
            number of points will be points * iteration_factor.
        title: Figure title. Defaults to str(len(ani)).
        duration: Animation duration (in seconds).
        fps: Number of frames per second for the output video.

    Returns:
        The animation object.
    """
    if title is None:
        title = str(len(ANIMATIONS))
    # Define initial positions
    linkage.rebuild(prev)
    if loci is None:
        loci = tuple(
            tuple(pos) for pos in linkage.step(  # type: ignore[arg-type]
                iterations=int(points * iteration_factor),
                dt=1 / iteration_factor
            )
        )

    fig = plt.figure("Result " + title, figsize=(14, 7))
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    linkage_bb = movement_bounding_box(loci)
    # We introduce a relative padding of 20%
    padding = (
        (linkage_bb[2] - linkage_bb[0]) ** 2
        + (linkage_bb[3] - linkage_bb[1]) ** 2
    ) ** .5 * .2
    for axis in (ax1, ax2):
        axis.set_xlim(linkage_bb[3] - padding, linkage_bb[1] + padding)
        axis.set_ylim(linkage_bb[0] - padding, linkage_bb[2] + padding)

    plot_static_linkage(linkage, ax1, loci, show_legend=True)
    animation = plot_kinematic_linkage(
        linkage, fig, ax2, loci, interval=1000 / fps
    )
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(duration)
    plt.close()
    if save:
        writer = anim.FFMpegWriter(fps=fps, bitrate=3600)
        animation.save(f"Kinematic {linkage.name}.mp4", writer=writer)
    return animation


def swarm_tiled_repr(
    linkage: "Linkage",
    swarm: "tuple[int, Sequence[tuple[float, np.ndarray, Sequence[Coord]]]]",
    fig: "Figure",
    axes: "np.ndarray",
    dimension_func: "Callable[[np.ndarray], Sequence[float]] | None" = None,
    points: int = 12,
    iteration_factor: float = 1,
) -> None:
    """Show all the linkages in a swarm in tiled mode.

    Args:
        linkage: The original Linkage that will be MODIFIED.
        swarm: Tuple of (iteration_number, list_of_agents) where each agent is
            (score, dimensions, initial_positions).
        fig: Figure to support the axes.
        axes: The subplot to draw on.
        points: Number of steps to use for each Linkage.
        iteration_factor: A simple way to subdivide the movement. The real
            number of points will be points * iteration_factor.
        dimension_func: If you want a special formatting of dimensions from
            agents before passing them to the linkage.
    """
    fig.suptitle(f"Iteration: {swarm[0]}, best score: {max(agent[0] for agent in swarm[1])}")
    for i, agent in enumerate(swarm[1]):
        dimensions = agent[1]
        if dimension_func is None:
            linkage.set_num_constraints(dimensions)
        else:
            linkage.set_num_constraints(dimension_func(dimensions))
        linkage.set_coords(agent[2])
        try:
            loci = tuple(
                tuple(pos) for pos in linkage.step(
                    iterations=int(points * iteration_factor),
                    dt=1 / iteration_factor
                )
            )
        except UnbuildableError:
            continue
        ax = axes.flatten()[i]
        ax.clear()
        plot_static_linkage(linkage, ax, loci)  # type: ignore[arg-type]
