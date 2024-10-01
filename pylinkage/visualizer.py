#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The visualizer module makes visualization of linkages easy using matplotlib.

Created on Mon Jun 14, 12:13:58 2021.

@author: HugoFara
"""
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .utility import movement_bounding_box
from .interface.exceptions import UnbuildableError
from .joints import Crank, Static
from .visualizer import plot_static_linkage
from .visualizer.core import _get_color

# List of animations
ANIMATIONS = []


def update_animated_plot(linkage, index, images, loci):
    """Modify im, instead of recreating it to make the animation run faster.

    :param linkage: DESCRIPTION.
    :type linkage: TYPE
    :param index: Frame index.
    :type index: int
    :param images: Artist to be modified.
    :type images: list of images Artists
    :param loci: list of loci.
    :type loci: list

    :returns: Updated version
    :rtype: list[Artists]
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
    """Plot a linkage with an animation.

    :param linkage: DESCRIPTION.
    :type linkage: pylinkage.linkage.Linkage
    :param fig: Figure to support the axes.
    :type fig: matplotlib.figure.Figure
    :param axis: The subplot to draw on.
    :type axis: matplotlib.axes._subplots.AxesSubplot
    :param loci: list of list of coordinates.
    :type loci: list
    :param frames: Number of frames to draw the linkage on. The default is 100.
    :type frames: int
    :param interval: Delay between frames in milliseconds. The default is 40 (24 fps).
    :type interval: float


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
        frames=frames,
        blit=True,
        interval=interval,
        repeat=True
    )
    return animation


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
    """Display results as an animated drawing.

    :param linkage: The Linkage you want to draw.
    :type linkage: pylinkage.linkage.Linkage
    :param save: To save the animation. The default is False.
    :type save: bool
    :param prev: Previous coordinates to use for linkage. The default is None.
    :type prev: list | tuple
    :param loci: list of loci. The default is None.
    :type loci: list
    :param points: Number of points to draw for a crank revolution.
        Useless when loci are set.
        The default is 100.
    :type points: int
    :param iteration_factor: A simple way to subdivide the movement. The real number of points
        will be points * iteration_factor. The default is 1.
    :type iteration_factor: float
    :param title: Figure title. The default is str(len(ani)).
    :type title: str
    :param duration: Animation duration (in seconds). The default is 5.
    :type duration: float
    :param fps: Number of frames per second for the output video.
        The default is 24.
    :type fps: int


    """
    # Define initial positions
    linkage.rebuild(prev)
    if loci is None:
        loci = tuple(
            map(
                tuple,
                linkage.step(
                    iterations=points * iteration_factor,
                    dt=1 / iteration_factor
                )
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
    linkage,
    swarm,
    fig,
    axes,
    dimension_func=None,
    points=12,
    iteration_factor=1
):
    """Show all the linkages in a swarm in tiled mode.

    :param linkage: The original Linkage that will be MODIFIED.
    :type linkage: pylinkage.linkage.Linkage
    :param swarm: Sequence of list of 3 elements: for each iteration, for each agent, (score, dimensions and initial
        positions).
    :type swarm: list
    :param fig: Figure to support the axes.
    :type fig: matplotlib.figure.Figure
    :param axes: The subplot to draw on.
    :type axes: matplotlib.axes._subplots.AxesSubplot
    :param points: Number of steps to use for each Linkage. The default is 12.
    :type points: int
    :param iteration_factor: A simple way to subdivide the movement. The real number of points
        will be points * iteration_factor. The default is 1.
    :type iteration_factor: float
    :param dimension_func: If you want a special formatting of dimensions from agents before
        passing them to the linkage. (Default value = None)
    :type dimension_func: callable, optional


    """
    fig.suptitle("Iteration: {}, best score: {}".format(swarm[0], max(agent[0] for agent in swarm[1])))
    for i, agent in enumerate(swarm[1]):
        dimensions = agent[1]
        if dimension_func is None:
            linkage.set_num_constraints(dimensions)
        else:
            linkage.set_num_constraints(dimension_func(dimensions))
        linkage.set_coords(agent[2])
        try:
            loci = tuple(
                map(
                    tuple,
                    linkage.step(
                        iterations=points * iteration_factor,
                        dt=1 / iteration_factor
                    )
                )
            )
        except UnbuildableError:
            continue
        axes.flatten()[i].clear()
        plot_static_linkage(linkage, axes.flatten()[i], loci)
