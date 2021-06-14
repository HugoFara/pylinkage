#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:13:58 2021

@author: HugoFara

This module makes visualisation of linkages easy using matplotlib.
"""
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .linkage import Crank, Fixed, Static, Pivot

# List of animations
ani = []


def plot_static_linkage(linkage, ax, locii, locus_highlights=None,
                        show_legend=False):
    """
    Plot a linkage without movement.

    Parameters
    ----------
    linkage : Linkage
        The linkage you want to see.
    ax : Artist
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
    ax.set_aspect('equal')
    ax.grid(True)
    for i in range(len(linkage.joints)):
        ax.plot(tuple(j[i][0] for j in locii), tuple(j[i][1] for j in locii))
    if locus_highlights:
        for locus in locus_highlights:
            ax.scatter(tuple(coord[0] for coord in locus),
                       tuple(coord[1] for coord in i))
    if show_legend:
        ax.set_title("Individual joint locus")
        ax.set_xlabel("Points abscisses")
        ax.set_ylabel("Ordinates")
        ax.legend(tuple(i.name for i in linkage.joints[:11]))
        ax.set_xlim(min(min(i[0] for i in m) for m in locii) - 4)


def update_animated_plot(linkage, index, im, locii):
    """
    Modify im to make the animation run faster.

    Parameters
    ----------
    linkage : TYPE
        DESCRIPTION.
    index : int
        Frame index.
    im : list of images Artists
        Artist to be modified.
    locii : list
        list of locuses.

    Returns
    -------
    im : list of images Artists
        Updated version.

    """
    a = 0
    locus = locii[index]
    for j, pos in enumerate(locus):
        joint = linkage.joints[j]
        # Draw link to first parent if it exists
        if joint.joint0 is None:
            continue
        par_locus = locus[linkage.joints.index(joint.joint0)]
        im[a].set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
        a += 1
        # Then second parent
        if isinstance(joint, (Crank, Static)):
            continue
        par_locus = locus[linkage.joints.index(joint.joint1)]
        im[a].set_data([par_locus[0], pos[0]], [par_locus[1], pos[1]])
        a += 1
    return im


def plot_animated_linkage(linkage, fig, ax, locii, frames=None, interval=.04):
    """
    Plot a linkage with an animation.

    Parameters
    ----------
    linkage : Linkage
        DESCRIPTION.
    fig : matplotlib.figure.Figure
        Figure to support the axes.
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot to draw on.
    locii : list
        list of list of coordinates.
    frames : int, optional
        Number of frames to draw the linkage on. The default is None.
    interval : float, optional
        Minimal amount of time between two frames. The default is .04 (24 fps).

    Returns
    -------
    None.

    """
    ax.set_aspect('equal')
    ax.set_title("Animation")

    im = []
    for j in linkage.joints:
        if isinstance(j, Static):
            # We will draw a fictive line with closest neighbor
            if j.joint0 is not None:
                im.append(ax.plot([], [], c='k', animated=False)[0])
        elif isinstance(j, Crank):
            # Crank has one parent only
            im.append(ax.plot([], [], c='g', animated=True)[0])
        elif isinstance(j, Fixed):
            im.append(ax.plot([], [], c='r', animated=True)[0])
            im.append(ax.plot([], [], c='r', animated=True)[0])
        elif isinstance(j, Pivot):
            im.append(ax.plot([], [], c='b', animated=True)[0])
            im.append(ax.plot([], [], c='b', animated=True)[0])

    padding = .5
    ax.set_xlim(min((min((i[0] for i in m)) for m in locii)) - padding,
                max((max((i[0] for i in m)) for m in locii)) + padding)
    ax.set_ylim(min((min((i[1] for i in m)) for m in locii)),
                max((max((i[1] for i in m)) for m in locii)) + padding)
    ani.append(anim.FuncAnimation(
        fig=fig,
        func=lambda index: update_animated_plot(linkage, index, im, locii),
        frames=frames, blit=True, interval=interval, repeat=True,
        save_count=frames))


def show_results(linkage, save=False, prev=None, L=None, title=str(len(ani)),
                 duration=5, points=100, fps=24):
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
    L : list, optional
        list of locii. The default is None.
    title : str, optional
        Figure title. The default is str(len(ani)).
    duration : float, optional
        Animation duration (in seconds). The default is 5.
    points : int, optional
        Number of point to draw for a crank revolution. The default is 100.
    fps : float, optional
        Number of frame per second for the output video. The default is 24.

    Returns
    -------
    None.

    """
    # Define intial positions
    linkage.rebuild(prev)
    if L is None:
        factor = int(12 / points) + 1
        L = tuple(tuple(i) for i in linkage.step(
            iterations=points * factor, dt=1 / factor))

    fig = plt.figure("Result " + title, figsize=(14, 7))
    fig.clear()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_static_linkage(linkage, ax1, L, show_legend=True)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_animated_linkage(linkage, fig, ax2, L)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(duration)
    plt.close()
    if save:
        writer = anim.FFMpegWriter(fps=fps, bitrate=3600)
        ani[-1].save("Linkage animated.mp4", writer=writer)
