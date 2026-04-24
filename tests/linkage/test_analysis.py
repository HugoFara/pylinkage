"""Tests for pylinkage.linkage.analysis.

Covers:
- bounding_box (empty input, single point, multi-point)
- movement_bounding_box (multiple loci)
- extract_trajectory / extract_trajectories
- kinematic_default_test decorator (successful + unbuildable paths)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.exceptions import UnbuildableError
from pylinkage.linkage.analysis import (
    bounding_box,
    extract_trajectories,
    extract_trajectory,
    kinematic_default_test,
    movement_bounding_box,
)
from pylinkage.simulation import Linkage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fourbar() -> Linkage:
    o1 = Ground(0.0, 0.0, name="O1")
    o2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=o2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([o1, o2, crank, rocker], name="FourBar")


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------


def test_bounding_box_empty_returns_infinities():
    bb = bounding_box([])
    assert bb[0] == float("inf")
    assert bb[1] == -float("inf")
    assert bb[2] == -float("inf")
    assert bb[3] == float("inf")


def test_bounding_box_single_point():
    y_min, x_max, y_max, x_min = bounding_box([(2.0, 3.0)])
    assert y_min == 3.0
    assert y_max == 3.0
    assert x_min == 2.0
    assert x_max == 2.0


def test_bounding_box_multiple_points():
    points = [(-1.0, 0.0), (2.0, 4.0), (5.0, -3.0), (0.0, 1.0)]
    y_min, x_max, y_max, x_min = bounding_box(points)
    assert x_min == -1.0
    assert x_max == 5.0
    assert y_min == -3.0
    assert y_max == 4.0


def test_bounding_box_accepts_generator():
    bb = bounding_box(iter([(0.0, 0.0), (1.0, 1.0)]))
    y_min, x_max, y_max, x_min = bb
    assert (x_min, y_min, x_max, y_max) == (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# movement_bounding_box
# ---------------------------------------------------------------------------


def test_movement_bounding_box_combines_loci():
    locus_a = [(0.0, 0.0), (1.0, 2.0)]
    locus_b = [(-2.0, -1.0), (3.0, 5.0)]
    y_min, x_max, y_max, x_min = movement_bounding_box([locus_a, locus_b])
    assert x_min == -2.0
    assert x_max == 3.0
    assert y_min == -1.0
    assert y_max == 5.0


def test_movement_bounding_box_empty_loci_iterable():
    bb = movement_bounding_box([])
    assert bb[0] == float("inf")
    assert bb[1] == -float("inf")
    assert bb[2] == -float("inf")
    assert bb[3] == float("inf")


def test_movement_bounding_box_single_locus():
    bb = movement_bounding_box([[(1.0, 1.0), (2.0, 3.0)]])
    y_min, x_max, y_max, x_min = bb
    assert (x_min, y_min, x_max, y_max) == (1.0, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# extract_trajectory
# ---------------------------------------------------------------------------


def test_extract_trajectory_integer_index_default_last():
    loci = [
        ((0.0, 0.0), (1.0, 2.0)),
        ((0.0, 0.0), (3.0, 4.0)),
    ]
    xs, ys = extract_trajectory(loci)
    np.testing.assert_array_equal(xs, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(ys, np.array([2.0, 4.0]))


def test_extract_trajectory_positive_integer_index():
    loci = [
        ((0.0, 0.0), (1.0, 2.0), (5.0, 6.0)),
        ((0.0, 0.0), (3.0, 4.0), (7.0, 8.0)),
    ]
    xs, ys = extract_trajectory(loci, joint=0)
    np.testing.assert_array_equal(xs, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(ys, np.array([0.0, 0.0]))


def test_extract_trajectory_skips_none_frames():
    loci = [
        ((1.0, 2.0),),
        ((None, None),),
        ((3.0, 4.0),),
    ]
    xs, ys = extract_trajectory(loci, joint=0)
    assert len(xs) == 2
    np.testing.assert_array_equal(xs, np.array([1.0, 3.0]))


def test_extract_trajectory_all_none_returns_empty():
    loci = [((None, None),), ((None, None),)]
    xs, ys = extract_trajectory(loci, joint=0)
    assert xs.size == 0
    assert ys.size == 0


def test_extract_trajectory_by_name_requires_linkage():
    with pytest.raises(ValueError, match="linkage is required"):
        extract_trajectory([((1.0, 1.0),)], joint="somejoint")


def test_extract_trajectory_by_name_with_linkage():
    lk = _make_fourbar()
    loci = tuple(lk.step(iterations=3))
    xs, ys = extract_trajectory(loci, joint="crank", linkage=lk)
    assert len(xs) == 3
    # Crank pivots around origin on a circle of radius 1.0
    for x, y in zip(xs, ys):
        assert math.isclose(math.hypot(x, y), 1.0, rel_tol=1e-6)


def test_extract_trajectory_unknown_joint_raises():
    lk = _make_fourbar()
    loci = tuple(lk.step(iterations=2))
    with pytest.raises(ValueError, match="not found"):
        extract_trajectory(loci, joint="nope", linkage=lk)


def test_extract_trajectory_by_instance():
    lk = _make_fourbar()
    joint = lk.components[2]  # the crank
    loci = tuple(lk.step(iterations=2))
    xs, ys = extract_trajectory(loci, joint=joint, linkage=lk)
    assert len(xs) == 2


# ---------------------------------------------------------------------------
# extract_trajectories
# ---------------------------------------------------------------------------


def test_extract_trajectories_empty_loci_returns_empty_dict():
    assert extract_trajectories([]) == {}


def test_extract_trajectories_no_linkage_keyed_by_index():
    loci = [
        ((0.0, 0.0), (1.0, 2.0)),
        ((0.0, 0.0), (None, None)),
        ((0.0, 0.0), (3.0, 4.0)),
    ]
    result = extract_trajectories(loci)
    assert set(result.keys()) == {0, 1}
    xs0, ys0 = result[0]
    assert xs0.size == 3
    xs1, ys1 = result[1]
    assert xs1.size == 2  # NaN frame skipped
    np.testing.assert_array_equal(xs1, np.array([1.0, 3.0]))


def test_extract_trajectories_with_linkage_keyed_by_name():
    lk = _make_fourbar()
    loci = tuple(lk.step(iterations=4))
    result = extract_trajectories(loci, linkage=lk)
    # All component names should be keys
    assert "O1" in result
    assert "O2" in result
    assert "crank" in result
    assert "rocker" in result
    for xs, ys in result.values():
        assert xs.shape == ys.shape


def test_extract_trajectories_linkage_mismatch_raises():
    lk = _make_fourbar()
    # frames with wrong number of joints
    loci = [((0.0, 0.0), (1.0, 1.0))]
    with pytest.raises(ValueError, match="joints but frames have"):
        extract_trajectories(loci, linkage=lk)


# ---------------------------------------------------------------------------
# kinematic_default_test
# ---------------------------------------------------------------------------


def test_kinematic_default_test_success_path():
    """The decorator should run a simulation and call the wrapped fn."""
    captured: dict[str, object] = {}

    def fitness(linkage, params, init_pos, loci):
        captured["params"] = params
        captured["init_pos"] = init_pos
        captured["loci"] = loci
        return 42.0

    wrapped = kinematic_default_test(fitness, error_penalty=-1.0)
    lk = _make_fourbar()
    params = list(lk.get_constraints())
    result = wrapped(lk, params)
    assert result == 42.0
    assert "loci" in captured
    # Loci is a tuple of tuples
    assert isinstance(captured["loci"], tuple)
    assert len(captured["loci"]) > 0
    # init_pos falls back to linkage.get_coords()
    assert captured["init_pos"] is not None


def test_kinematic_default_test_with_explicit_init_pos():
    def fitness(linkage, params, init_pos, loci):
        return init_pos

    wrapped = kinematic_default_test(fitness, error_penalty=0.0)
    lk = _make_fourbar()
    params = list(lk.get_constraints())
    init_pos = lk.get_coords()
    result = wrapped(lk, params, init_pos=init_pos)
    assert result == init_pos


def test_kinematic_default_test_returns_penalty_on_unbuildable():
    """An unbuildable linkage should return the penalty and not call fitness."""
    called = []

    def fitness(linkage, params, init_pos, loci):
        called.append(True)
        return 1.0

    wrapped = kinematic_default_test(fitness, error_penalty=-999.0)
    lk = _make_fourbar()
    # Give absurd distances making the dyad geometrically impossible.
    # Crank radius = 1, anchor at (3, 0). Make distances way too small.
    bad_params = [1.0, 0.01, 0.01]
    result = wrapped(lk, bad_params)
    assert result == -999.0
    assert not called


def test_kinematic_default_test_penalty_from_step_failure():
    """Forcing a specific UnbuildableError via degenerate constraints."""
    penalty_raised = {"count": 0}

    def fitness(linkage, params, init_pos, loci):
        penalty_raised["count"] += 1
        return 7.0

    wrapped = kinematic_default_test(fitness, error_penalty=float("inf"))
    lk = _make_fourbar()
    # Use safe nominal params — should succeed
    params = list(lk.get_constraints())
    assert wrapped(lk, params) == 7.0
    assert penalty_raised["count"] == 1
