"""``Linkage.step_fast()`` must agree with ``Linkage.step()``, dyad by dyad.

Before this suite existed, ``linkage_to_solver_data`` typed every dyad as
``JOINT_REVOLUTE`` and read ``distance1``/``distance2`` off it. Only ``RRRDyad``
has those attributes, so a ``FixedDyad`` reached the solver as two zero-radius
circles and produced a trajectory of ``NaN`` -- silently, with no exception. The
bridge tests covered the ``RRRDyad`` path, and nothing compared the two
simulation paths for anything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, PPDyad, RRPDyad, RRRDyad
from pylinkage.simulation import Linkage

ITERATIONS = 12


def make_fourbar() -> Linkage:
    """Crank plus an RRR dyad: the only case the solver ever handled."""
    anchor = Ground(0.0, 0.0, name="A")
    frame = Ground(3.0, 0.0, name="D")
    crank = Crank(anchor=anchor, radius=1.0, angular_velocity=0.31, name="B")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=frame, distance1=3.0, distance2=1.5, name="C"
    )
    return Linkage([anchor, frame, crank, rocker], name="fourbar")


def make_fourbar_with_coupler_point() -> Linkage:
    """A four-bar carrying a coupler point, which is what synthesis emits."""
    anchor = Ground(0.0, 0.0, name="A")
    frame = Ground(3.0, 0.0, name="D")
    crank = Crank(anchor=anchor, radius=1.0, angular_velocity=0.31, name="B")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=frame, distance1=3.0, distance2=1.5, name="C"
    )
    coupler_point = FixedDyad(
        anchor1=crank.output, anchor2=rocker, distance=1.4, angle=-0.37, name="P"
    )
    return Linkage([anchor, frame, crank, rocker, coupler_point], name="coupler")


def make_crank_slider() -> Linkage:
    """A slider on a horizontal rail.

    ``distance`` exceeds the largest crank-tip-to-rail distance (3.0), so the
    circle meets the line at every crank angle and the mechanism stays
    buildable through a full revolution.
    """
    anchor = Ground(0.0, 0.0, name="A")
    rail_start = Ground(0.0, -2.0, name="L1")
    rail_end = Ground(5.0, -2.0, name="L2")
    crank = Crank(anchor=anchor, radius=1.0, angular_velocity=0.31, name="B")
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=rail_start,
        line_anchor2=rail_end,
        distance=3.5,
        name="S",
    )
    return Linkage([anchor, rail_start, rail_end, crank, slider], name="slider")


def trajectory_via_step(linkage: Linkage) -> np.ndarray:
    """Collect ``step()`` output in the same shape ``step_fast()`` returns."""
    return np.array(
        [list(positions) for positions in linkage.step(iterations=ITERATIONS)],
        dtype=float,
    )


BUILDERS = [
    pytest.param(make_fourbar, id="RRRDyad"),
    pytest.param(make_fourbar_with_coupler_point, id="FixedDyad"),
    pytest.param(make_crank_slider, id="RRPDyad"),
]


@pytest.mark.parametrize("builder", BUILDERS)
def test_step_fast_matches_step(builder):
    """The two simulation paths agree to floating-point exactness.

    Both call the same ``solve_*`` functions, so anything other than an exact
    match means the conversion handed the solver different numbers.
    """
    slow = trajectory_via_step(builder())
    fast = builder().step_fast(iterations=ITERATIONS)

    assert fast.shape == slow.shape
    np.testing.assert_allclose(fast, slow, rtol=0, atol=0)


@pytest.mark.parametrize("builder", BUILDERS)
def test_step_fast_produces_no_nan(builder):
    """A NaN trajectory is the failure mode this suite exists to catch."""
    fast = builder().step_fast(iterations=ITERATIONS)
    assert np.isfinite(fast).all()


def test_unsupported_dyad_raises_instead_of_returning_nan():
    """``PPDyad`` has four anchors, one more than the solver can carry.

    It must say so rather than emit a wrong joint type, which is what produced
    silent NaN before.
    """
    a = Ground(0.0, 0.0, name="A")
    b = Ground(4.0, 0.0, name="B")
    c = Ground(0.0, 1.0, name="C")
    d = Ground(4.0, 3.0, name="D")
    crank = Crank(anchor=a, radius=1.0, angular_velocity=0.31, name="crank")
    intersection = PPDyad(
        line1_anchor1=crank.output,
        line1_anchor2=b,
        line2_anchor1=c,
        line2_anchor2=d,
        name="X",
    )
    linkage = Linkage([a, b, c, d, crank, intersection], name="pp")

    with pytest.raises(NotImplementedError, match="PPDyad"):
        linkage.step_fast(iterations=3)


def test_unsupported_dyad_error_names_the_alternative():
    """The message has to tell the caller what to do instead."""
    a = Ground(0.0, 0.0)
    b = Ground(4.0, 0.0)
    c = Ground(0.0, 1.0)
    d = Ground(4.0, 3.0)
    crank = Crank(anchor=a, radius=1.0, angular_velocity=0.31)
    linkage = Linkage(
        [
            a,
            b,
            c,
            d,
            crank,
            PPDyad(
                line1_anchor1=crank.output,
                line1_anchor2=b,
                line2_anchor1=c,
                line2_anchor2=d,
            ),
        ]
    )

    with pytest.raises(NotImplementedError, match=r"step\(\)"):
        linkage.step_fast(iterations=3)
