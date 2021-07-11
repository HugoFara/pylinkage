#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The visualizer module makes visualisation of linkages easy using matplotlib.

Created on Mon Jun 14 12:13:58 2021.

@author: HugoFara
"""
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pylinkage.geometry import bounding_box

from .exceptions import UnbuildableError
from .linkage import Crank, Fixed, Static, Pivot

# List of animations
ANIMATIONS = []

# Colors to use for plotting
COLOR_SWITCHER = {
    Static: 'k',
    Crank: 'g',
    Fixed: 'r',
    Pivot: 'b'
}

def _get_color(joint):
    """Search in COLOR_SWITCHER for the corresponding color."""
    for joint_type, color in COLOR_SWITCHER.items():
        if isinstance(joint, joint_type):
            return color
    return ''

def plot_static_linkage(linkage, axis, loci, locus_highlights=None,
                        show_legend=False):
    """
    Plot a linkage without movement.

    Parameters
    ----------
    linkage : Linkage
        The linkage you want to see.
    axis : Artist
        The graph we should draw on.
    loci : sequence
        List of list of coordinates. They will be plotted.
    locus_highlights : list, optional
        If a list, shoud be a list of list of coordinates you want to see
        highlighted. The default is None.
    show_legend : bool, optional
        To add an automatic legend to the graph. The default is False.

    Returns
    -------
    None.

    """
    axis.set_aspect('equal')
    axis.grid(True)
    # Plot loci
    for i, joint in enumerate(linkage.joints):
        axis.plot(tuple(j[i][0] for j in loci), tuple(j[i][1] for j in loci))

    # The plot linkage in intial position
    # It as imporant to use separate loops, because we would have bad
    # formatted legend otherwisee
    for i, joint in enumerate(linkage.joints):
        # Then the linkage in initial position
        # Draw link to first parent if it exists
        if joint.joint0 is None:
            continue
        pos = joint.coord()
        par_pos = joint.joint0.coord()
        plot_kwargs = {
            "c": _get_color(joint),
            "linewidth": .3
        }
        axis.plot(
            [par_pos[0], pos[0]], [par_pos[1], pos[1]],
            **plot_kwargs
        )
        # Then second parent
        if isinstance(joint, (Crank, Static)):
            continue
        par_pos = joint.joint1.coord()
        axis.plot(
            [par_pos[0], pos[0]], [par_pos[1], pos[1]],
            **plot_kwargs
        )

    # Highlight for specific loci
    if locus_highlights:
        for locus in locus_highlights:
            axis.scatter(tuple(coord[0] for coord in locus),
                         tuple(coord[1] for coord in i))

    if show_legend:
        axis.set_title("Static representation")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend(tuple(i.name for i in linkage.joints))

def update_animated_plot(linkage, index, images, loci):
    """
    Modify im, instead of recreating it to make the animation run faster.

    Parameters
    ----------
    linkage : TYPE
        DESCRIPTION.
    index : int
        Frame index.
    images : list of images Artists
        Artist to be modified.
    loci : list
        list of locuses.

    Returns
    -------
    im : list of images Artists
        Updated version.

    """
    image = iter(images)
    locus = loci[index]
    for j, pos in enumerate(locus):
        joint = linkage.joints[j]
        # Draw link to first parent if it exists
        if joint.joint0 is None:
            continue
        if isinstance(joint.joint0, Static):
            par_locus = joint.joint0.coord()
        else:
            par_locus = locus[linkage.joints.index(joint.joint0)]
        next(image).set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
        # Then second parent
        if isinstance(joint, (Crank, Static)):
            continue
        if isinstance(joint.joint1, Static):
            par_locus = joint.joint1.coord()
        else:
            par_locus = locus[linkage.joints.index(joint.joint1)]
        next(image).set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
    return images


def plot_kinematic_linkage(
        linkage,
        fig,
        axis,
        loci,
        frames=100,
        interval=40
):
    """
    Plot a linkage with an animation.

    Parameters
    ----------
    linkage : pylinkage.linkage.Linkage
        DESCRIPTION.
    fig : matplotlib.figure.Figure
        Figure to support the axes.
    axis : matplotlib.axes._subplots.AxesSubplot
        The subplot to draw on.
    loci : list
        list of list of coordinates.
    frames : int, optional
        Number of frames to draw the linkage on. The default is 100.
    interval : float, optional
        Delay between frames in milliseconds. The default is 40 (24 fps).

    Returns
    -------
    None.

    """
    axis.set_aspect('equal')
    axis.set_title("Animation")

    images = []
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
        frames=frames, blit=True, interval=interval, repeat=True,
        save_count=frames)
    return animation

def movement_bounding_bow(loci):
    """Return the general bounding box of a group of loci."""
    bb = (float('inf'), -float('inf'), -float('inf'), float('inf'))
    for locus in loci:
        new_bb = bounding_box(locus)
        bb = (
            min(new_bb[0], bb[0]), max(new_bb[1], bb[1]),
            max(new_bb[2], bb[2]), min(new_bb[3], bb[3])
        )
    return bb

def show_linkage(
        linkage,
        save=False,
        prev=None,
        loci=None,
        points=100,
        iteration_factor=1,
        title=str(len(ANIMATIONS)),
        duration=5,
        fps=24
):
    """
    Display results as an animated drawing.

    Parameters
    ----------
    linkage : pylinkage.linkage.Linkage
        The Linkage you want to draw.
    save : bool, optional
        To save the animation. The default is False.
    prev : list, optional
        Previous coordinates to use for linkage. The default is None.
    loci : list, optional
        list of loci. The default is None.
    points : int, optional
        Number of point to draw for a crank revolution.
        Useless when loci is set.
        The default is 100.
    iteration_factor : float, optional
        A simple way to subdivide the movement. The real number of points
        will be points * iteration_factor. The default is 1.
    title : str, optional
        Figure title. The default is str(len(ani)).
    duration : float, optional
        Animation duration (in seconds). The default is 5.
    fps : float, optional
        Number of frame per second for the output video. The default is 24.

    Returns
    -------
    None.

    """
    # Define intial positions
    linkage.rebuild(prev)
    if loci is None:
        loci = tuple(tuple(i) for i in linkage.step(
            iterations=points * iteration_factor, dt=1 / iteration_factor))

    fig = plt.figure("Result " + title, figsize=(14, 7))
    fig.clear()

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    linkage_bb = movement_bounding_bow(loci)
    # We introduce a relative padding of 20%
    padding = (
        (linkage_bb[2] -  linkage_bb[0]) ** 2
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
    # Save as global variable
    ANIMATIONS.append(animation)
    return animation


def swarm_tiled_repr(
        linkage,
        swarm,
        fig,
        axes,
        dimension_func=None,
        points=12,
        iteration_factor=1):
    """
    Show all the linkages in a swarm in tiled mode.

    Parameters
    ----------
    linkage : pylinkage.linkage.Linkage
        The original Linkage that will be MODIFIED.
    swarm : list
        Sequence of 3 elements: agents, interation number and initial
        positions.
    fig : matplotlib.figure.Figure
        Figure to support the axes.
    axes : matplotlib.axes._subplots.AxesSubplot
        The subplot to draw on.
    points : int, optional
        Number of steps to use for each Linkage. The default is 12.
    iteration_factor : float, optional
        A simple way to subdivide the movement. The real number of points
        will be points * iteration_factor. The default is 1.
    dimension_func : callable, optional
        If you want a special formatting of dimensions from agents before
        passing them to the linkage.

    Returns
    -------
    None.

    """
    # fig.suptitle("Iteration: {}, agents: {}".format(swarm[1], len(agents)))
    for i, dimensions in enumerate(swarm):
        if dimension_func is None:
            linkage.set_num_constraints(dimensions)
        else:
            linkage.set_num_constraints(dimension_func(dimensions))
        axes.flatten()[i].clear()
        try:
            loci = tuple(
                tuple(pos) for pos in linkage.step(
                    iterations=points * iteration_factor,
                    dt=1 / iteration_factor
                    ))
        except UnbuildableError:
            pass
        else:
            plot_static_linkage(linkage, axes.flatten()[i], loci)
