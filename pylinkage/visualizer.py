#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:13:58 2021.

@author: HugoFara

This module makes visualisation of linkages easy using matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .exceptions import UnbuildableError
from .linkage import Crank, Fixed, Static, Pivot

# List of animations
animations = []


def plot_static_linkage(linkage, axis, locii, locus_highlights=None,
                        show_legend=False):
    """
    Plot a linkage without movement.

    Parameters
    ----------
    linkage : Linkage
        The linkage you want to see.
    axis : Artist
        The graph we should draw on.
    locii : sequence
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
    for i in range(len(linkage.joints)):
        axis.plot(tuple(j[i][0] for j in locii), tuple(j[i][1] for j in locii))
    if locus_highlights:
        for locus in locus_highlights:
            axis.scatter(tuple(coord[0] for coord in locus),
                         tuple(coord[1] for coord in i))
    if show_legend:
        axis.set_title("Individual joint locus")
        axis.set_xlabel("Points abscisses")
        axis.set_ylabel("Ordinates")
        axis.legend(tuple(i.name for i in linkage.joints[:11]))
        axis.set_xlim(min(min(i[0] for i in m) for m in locii) - 4)


def update_animated_plot(linkage, index, images, locii):
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
    locii : list
        list of locuses.

    Returns
    -------
    im : list of images Artists
        Updated version.

    """
    image = iter(images)
    locus = locii[index]
    for j, pos in enumerate(locus):
        joint = linkage.joints[j]
        # Draw link to first parent if it exists
        if joint.joint0 is None:
            continue
        par_locus = locus[linkage.joints.index(joint.joint0)]
        next(image).set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
        # Then second parent
        if isinstance(joint, (Crank, Static)):
            continue
        par_locus = locus[linkage.joints.index(joint.joint1)]
        next(image).set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
    return images


def plot_kinematic_linkage(linkage, fig, axis, locii, frames=None,
                           interval=40):
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
    locii : list
        list of list of coordinates.
    frames : int, optional
        Number of frames to draw the linkage on. The default is None.
    interval : float, optional
        Delay between frames in milliseconds. The default is 40 (24 fps).

    Returns
    -------
    None.

    """
    axis.set_aspect('equal')
    axis.set_title("Animation")

    images = []
    for j in linkage.joints:
        if isinstance(j, Static):
            # We will draw a fictive line with closest neighbor
            if j.joint0 is not None:
                images.append(axis.plot([], [], c='k', animated=False)[0])
        elif isinstance(j, Crank):
            # Crank has one parent only
            images.append(axis.plot([], [], c='g', animated=True)[0])
        elif isinstance(j, Fixed):
            images.append(axis.plot([], [], c='r', animated=True)[0])
            images.append(axis.plot([], [], c='r', animated=True)[0])
        elif isinstance(j, Pivot):
            images.append(axis.plot([], [], c='b', animated=True)[0])
            images.append(axis.plot([], [], c='b', animated=True)[0])

    padding = .5
    axis.set_xlim(min((min((i[0] for i in m)) for m in locii)) - padding,
                  max((max((i[0] for i in m)) for m in locii)) + padding)
    axis.set_ylim(min((min((i[1] for i in m)) for m in locii)),
                  max((max((i[1] for i in m)) for m in locii)) + padding)
    animation = anim.FuncAnimation(
        fig=fig,
        func=lambda index: update_animated_plot(linkage, index % len(locii),
                                                images, locii),
        frames=frames, blit=True, interval=interval, repeat=True,
        save_count=frames)
    return animation


def show_linkage(linkage, save=False, prev=None, locii=None, points=100,
                 iteration_factor=1, title=str(len(animations)), duration=5,
                 fps=24):
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
    locii : list, optional
        list of locii. The default is None.
    points : int, optional
        Number of point to draw for a crank revolution.
        Useless when locii is set
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
    if locii is None:
        locii = tuple(tuple(i) for i in linkage.step(
            iterations=points * iteration_factor, dt=1 / iteration_factor))

    fig = plt.figure("Result " + title, figsize=(14, 7))
    fig.clear()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_static_linkage(linkage, ax1, locii, show_legend=True)
    ax2 = fig.add_subplot(1, 2, 2)
    animation = plot_kinematic_linkage(linkage, fig, ax2, locii,
                                       interval=1000 / fps)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(duration)
    plt.close()
    if save:
        writer = anim.FFMpegWriter(fps=fps, bitrate=3600)
        animation.save("Kinamtic linkage animation.mp4", writer=writer)
    # Save as global variable
    animations.append(animation)
    return animation


def swarm_tiled_repr(linkage, swarm, fig, axes, dimension_func=None,
                     points=12, iteration_factor=1):
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
            locii = tuple(
                tuple(pos) for pos in linkage.step(
                    iterations=points * iteration_factor,
                    dt=1 / iteration_factor
                    ))
        except UnbuildableError:
            pass
        else:
            plot_static_linkage(linkage, axes.flatten()[i], locii)
