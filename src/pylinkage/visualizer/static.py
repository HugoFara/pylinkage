"""
Static (not animated) visualization.
"""


from typing import TYPE_CHECKING

from ..joints import Fixed, Prismatic, Revolute
from ..joints.revolute import Pivot
from .core import _get_color

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes

    from .._types import Coord
    from ..linkage.linkage import Linkage


def plot_static_linkage(
    linkage: "Linkage",
    axis: "Axes",
    loci: "Iterable[tuple[Coord, ...]]",
    locus_highlights: "list[list[Coord]] | None" = None,
    show_legend: bool = False,
) -> None:
    """Plot a linkage without movement.

    Args:
        linkage: The linkage you want to see.
        axis: The graph we should draw on.
        loci: List of list of coordinates. They will be plotted.
        locus_highlights: If a list, should be a list of list of coordinates you
            want to see highlighted.
        show_legend: To add an automatic legend to the graph.
    """
    axis.set_aspect('equal')
    axis.grid(True)
    # Plot loci
    for i, _joint in enumerate(linkage.joints):
        axis.plot(tuple(j[i][0] for j in loci), tuple(j[i][1] for j in loci))

    # The plot linkage in initial positioning
    # It as important to use separate loops, because we would have bad
    # formatted legend otherwise
    for _i, joint in enumerate(linkage.joints):
        # Then the linkage in initial position

        # Draw a link to the first parent if it exists
        if joint.joint0 is None:
            continue
        pos = joint.coord()
        par_pos = joint.joint0.coord()
        axis.plot(
            [par_pos[0], pos[0]], [par_pos[1], pos[1]],  # type: ignore[arg-type]
            c=_get_color(joint), linewidth=.3
        )
        # Then second parent
        if isinstance(joint, (Fixed, Pivot, Revolute)) and joint.joint1 is not None:
            par_pos = joint.joint1.coord()
            axis.plot(
                [par_pos[0], pos[0]], [par_pos[1], pos[1]],  # type: ignore[arg-type]
                c=_get_color(joint), linewidth=.3
            )
        elif isinstance(joint, Prismatic) and joint.joint1 is not None and joint.joint2 is not None:
            # Different ordering
            par_pos = joint.joint2.coord()
            other_pos = joint.joint1.coord()
            axis.plot(
                [par_pos[0], other_pos[0]], [par_pos[1], other_pos[1]],  # type: ignore[arg-type]
                c=_get_color(joint), linewidth=.3
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
