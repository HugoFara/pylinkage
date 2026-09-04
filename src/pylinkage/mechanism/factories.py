"""Factory functions for common planar mechanisms.

These helpers build standard parametric mechanisms with a single call,
so users don't need to repeat the
``MechanismBuilder() .add_ground_link() .add_driver_link() .connect() ...``
boilerplate (or hand-roll a circle-circle intersection) every time they
want a four-bar or slider-crank.
"""

from __future__ import annotations

import math

from .builder import MechanismBuilder
from .mechanism import Mechanism


def fourbar(
    crank: float,
    coupler: float,
    rocker: float,
    ground: float,
    omega: float = math.tau / 100,
    initial_angle: float = 0.0,
    branch: int = 1,
    name: str = "fourbar",
) -> Mechanism:
    """Build a four-bar Mechanism from link lengths.

    Ground pivots are placed at ``A = (0, 0)`` and ``D = (ground, 0)``.
    The crank rotates about ``A`` and the rocker oscillates about ``D``.

    Args:
        crank: Crank length (link a, ``A``-``B``).
        coupler: Coupler length (link b, ``B``-``C``).
        rocker: Rocker length (link c, ``C``-``D``).
        ground: Ground link length (link d, ``A``-``D``).
        omega: Driver angular velocity, in rad/step.
        initial_angle: Starting crank angle, in radians.
        branch: ``0`` or ``1`` — which of the two circle-circle
            intersections to pick for the coupler-rocker joint. Branch
            ``1`` (default) is the upper (positive y) configuration,
            matching ``synthesis.fourbar_from_lengths``.
        name: Name of the resulting mechanism.

    Returns:
        An assembled :class:`Mechanism`.

    Raises:
        pylinkage.exceptions.UnbuildableError: if the link lengths
            cannot form a closed loop at ``initial_angle``.

    Example:
        >>> from pylinkage.mechanism import fourbar
        >>> mech = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        >>> loci = list(mech.step())
    """
    builder = MechanismBuilder(name=name)
    builder.add_ground_link("ground", ports={"A": (0.0, 0.0), "D": (ground, 0.0)})
    builder.add_driver_link(
        "crank", length=crank, motor_port="A",
        omega=omega, initial_angle=initial_angle,
    )
    builder.add_link("coupler", length=coupler)
    builder.add_link("rocker", length=rocker)
    builder.connect("crank.tip", "coupler.0")
    builder.connect("coupler.1", "rocker.0")
    builder.connect("rocker.1", "ground.D")
    builder.set_branch("coupler.1", branch)
    return builder.build()


def slider_crank(
    crank: float,
    rod: float,
    omega: float = math.tau / 100,
    initial_angle: float = 0.0,
    slide_through: tuple[float, float] = (0.0, 0.0),
    slide_direction: tuple[float, float] = (1.0, 0.0),
    name: str = "slider-crank",
) -> Mechanism:
    """Build a slider-crank Mechanism.

    A crank of length ``crank`` rotates about the origin and drives a rod
    of length ``rod`` whose far end slides along the line through
    ``slide_through`` in direction ``slide_direction``.

    Args:
        crank: Crank length.
        rod: Connecting rod length.
        omega: Driver angular velocity, in rad/step.
        initial_angle: Starting crank angle, in radians.
        slide_through: A point the slide axis passes through.
        slide_direction: Direction vector of the slide axis.
        name: Name of the resulting mechanism.

    Returns:
        An assembled :class:`Mechanism`.

    Example:
        >>> from pylinkage.mechanism import slider_crank
        >>> mech = slider_crank(crank=1.0, rod=3.0)
        >>> loci = list(mech.step())
    """
    builder = MechanismBuilder(name=name)
    builder.add_ground_link("ground", ports={"O": (0.0, 0.0)})
    builder.add_driver_link(
        "crank", length=crank, motor_port="O",
        omega=omega, initial_angle=initial_angle,
    )
    builder.add_link("rod", length=rod)
    builder.add_slide_axis("rail", through=slide_through, direction=slide_direction)
    builder.connect("crank.tip", "rod.0")
    builder.connect_prismatic("rod.1", "rail")
    return builder.build()
