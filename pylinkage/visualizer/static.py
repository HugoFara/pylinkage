"""
Static (not animated) visualization.
"""

from .core import _get_color
from ..joints import (Fixed, Revolute, Linear)
from ..interface import Pivot


def plot_static_linkage(
        linkage, axis, loci, locus_highlights=None,
        show_legend=False
):
    """Plot a linkage without movement.

    :param linkage: The linkage you want to see.
    :type linkage: Linkage
    :param axis: The graph we should draw on.
    :type axis: Artist
    :param loci: List of list of coordinates. They will be plotted.
    :type loci: Iterable
    :param locus_highlights: If a list, should be a list of list of coordinates you want to see
        highlighted. The default is None.
    :type locus_highlights: list
    :param show_legend: To add an automatic legend to the graph. The default is False.
    :type show_legend: bool


    """
    axis.set_aspect('equal')
    axis.grid(True)
    # Plot loci
    for i, joint in enumerate(linkage.joints):
        axis.plot(tuple(j[i][0] for j in loci), tuple(j[i][1] for j in loci))

    # The plot linkage in initial positioning
    # It as important to use separate loops, because we would have bad
    # formatted legend otherwise
    for i, joint in enumerate(linkage.joints):
        # Then the linkage in initial position

        # Draw a link to the first parent if it exists
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
        if isinstance(joint, (Fixed, Pivot, Revolute)):
            par_pos = joint.joint1.coord()
            axis.plot(
                [par_pos[0], pos[0]], [par_pos[1], pos[1]],
                **plot_kwargs
            )
        elif isinstance(joint, Linear):
            # Different ordering
            par_pos = joint.joint2.coord()
            other_pos = joint.joint1.coord()
            axis.plot(
                [par_pos[0], other_pos[0]], [par_pos[1], other_pos[1]],
                **plot_kwargs
            )

    # Highlight for specific loci
    if locus_highlights:
        for locus in locus_highlights:
            axis.scatter(
                tuple(coord[0] for coord in locus),
                tuple(coord[1] for coord in locus)
            )

    if show_legend:
        axis.set_title("Static representation")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.legend(tuple(i.name for i in linkage.joints))
