"""
Analysis tools for linkages.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import BoundingBox, Coord, JointPositions
from ..exceptions import UnbuildableError

if TYPE_CHECKING:
    from .linkage import Linkage


def kinematic_default_test(
    func: Callable[..., float],
    error_penalty: float,
) -> "Callable[[Linkage, Iterable[float], JointPositions | None], float]":
    """Standard run for any linkage before a complete fitness evaluation.

    This decorator makes a kinematic simulation, before passing the loci to the
    decorated function.

    :param func: Fitness function to be decorated.
    :param error_penalty: Penalty value for unbuildable linkage. Common values include
            float('inf') and 0.
    """

    def wrapper(
        linkage: "Linkage",
        params: Iterable[float],
        init_pos: JointPositions | None = None,
    ) -> float:
        """Decorated function.

        :param linkage: The linkage to optimize.
        :param params: Geometric constraints to pass to linkage.set_num_constraints.
        :param init_pos: List of initial positions for the joints. If None it will be
            redefined at each successful iteration. (Default value = None).

        :return: Fitness score.
        """
        if init_pos is not None:
            linkage.set_coords(init_pos)
        linkage.set_num_constraints(params)
        try:
            points = 12
            n = linkage.get_rotation_period()
            # Complete revolution with 12 points
            tuple(tuple(i) for i in linkage.step(iterations=points + 1, dt=n / points))
            # Again with n points, and at least 12 iterations
            n = 96
            factor = int(points / n) + 1
            loci = tuple(tuple(i) for i in linkage.step(iterations=n * factor, dt=1 / factor))
        except UnbuildableError:
            return error_penalty
        else:
            # We redefine the initial position if requested
            actual_init_pos = init_pos if init_pos is not None else linkage.get_coords()
            return func(linkage=linkage, params=params, init_pos=actual_init_pos, loci=loci)

    return wrapper


def extract_trajectory(
    loci: "Sequence[Sequence[Coord | tuple[Any, Any]]]",
    joint: "int | str | Any" = -1,
    linkage: "Any | None" = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Extract the (x, y) path of a single joint from simulation loci.

    Frames where the joint position is ``None`` (unbuildable configuration) are
    silently skipped.

    :param loci: Sequence of frames as produced by ``Linkage.step()`` or
        ``Mechanism.step()``. Each frame is a sequence of ``(x, y)`` tuples,
        one per joint/component in the linkage's iteration order.
    :param joint: Which joint's trajectory to extract. Can be:

        - an integer index into each frame (default ``-1`` = last joint),
        - a joint/component name (requires ``linkage``),
        - a joint/component instance (requires ``linkage``).

    :param linkage: The ``Linkage`` or ``Mechanism`` the loci come from.
        Required when ``joint`` is a name or instance.

    :returns: Pair ``(xs, ys)`` of ``numpy.ndarray`` with the same length.
        Empty arrays if every frame is unbuildable.
    """
    if isinstance(joint, int):
        index = joint
    else:
        if linkage is None:
            msg = "linkage is required when joint is not an integer index"
            raise ValueError(msg)
        index = _resolve_joint_index(linkage, joint)

    xs: list[float] = []
    ys: list[float] = []
    for frame in loci:
        point = frame[index]
        if point[0] is None or point[1] is None:
            continue
        xs.append(point[0])
        ys.append(point[1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _resolve_joint_index(linkage: "Any", joint: "Any") -> int:
    """Return the frame index of ``joint`` within ``linkage``'s iteration order."""
    candidates = (
        getattr(linkage, "joints", None)
        or getattr(linkage, "components", None)
        or ()
    )
    for i, candidate in enumerate(candidates):
        if candidate is joint or getattr(candidate, "name", None) == joint:
            return i
    msg = f"joint {joint!r} not found in linkage"
    raise ValueError(msg)


def bounding_box(locus: Iterable[Coord]) -> BoundingBox:
    """Compute the bounding box of a locus.

    :param locus: A list of points or any iterable with the same structure.

    :returns: Bounding box as (y_min, x_max, y_max, x_min).
    """
    y_min = float("inf")
    x_min = float("inf")
    y_max = -float("inf")
    x_max = -float("inf")
    for point in locus:
        y_min = min(y_min, point[1])
        x_min = min(x_min, point[0])
        y_max = max(y_max, point[1])
        x_max = max(x_max, point[0])
    return y_min, x_max, y_max, x_min


def movement_bounding_box(loci: Iterable[Iterable[Coord]]) -> BoundingBox:
    """
    Bounding box for a group of loci.

    :param loci: Iterable of loci (sequences of coordinates).

    :returns: Bounding box as (y_min, x_max, y_max, x_min).
    """
    bb: BoundingBox = (float("inf"), -float("inf"), -float("inf"), float("inf"))
    for locus in loci:
        new_bb = bounding_box(locus)
        bb = (
            min(new_bb[0], bb[0]),
            max(new_bb[1], bb[1]),
            max(new_bb[2], bb[2]),
            min(new_bb[3], bb[3]),
        )
    return bb
