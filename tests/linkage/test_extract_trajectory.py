"""Tests for extract_trajectory."""

import math

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.linkage.analysis import extract_trajectories, extract_trajectory


def _square_loci():
    """Three frames, two joints each; last joint traces a square corner."""
    return [
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (0.0, 1.0)),
    ]


class TestExtractTrajectory:
    def test_default_last_joint(self):
        xs, ys = extract_trajectory(_square_loci())
        assert np.array_equal(xs, [1.0, 1.0, 0.0])
        assert np.array_equal(ys, [0.0, 1.0, 1.0])

    def test_integer_index(self):
        xs, ys = extract_trajectory(_square_loci(), joint=0)
        assert np.array_equal(xs, [0.0, 0.0, 0.0])
        assert np.array_equal(ys, [0.0, 0.0, 0.0])

    def test_skips_none_frames(self):
        loci = [
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (None, None)),
            ((0.0, 0.0), (0.0, 1.0)),
        ]
        xs, ys = extract_trajectory(loci)
        assert np.array_equal(xs, [1.0, 0.0])
        assert np.array_equal(ys, [0.0, 1.0])

    def test_all_none_returns_empty(self):
        loci = [((0.0, 0.0), (None, None))] * 3
        xs, ys = extract_trajectory(loci)
        assert xs.size == 0 and ys.size == 0

    def test_returns_numpy_arrays(self):
        xs, ys = extract_trajectory(_square_loci())
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert xs.dtype == np.float64

    def test_by_name_requires_linkage(self):
        with pytest.raises(ValueError, match="linkage is required"):
            extract_trajectory(_square_loci(), joint="C")

    def test_by_name_with_linkage(self):
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=math.tau / 20, distance=1, name="B")
        pin = pl.Revolute(
            3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="C"
        )
        linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin], name="fourbar")
        loci = list(linkage.step(iterations=20))

        xs_name, ys_name = extract_trajectory(loci, joint="C", linkage=linkage)
        xs_obj, ys_obj = extract_trajectory(loci, joint=pin, linkage=linkage)
        xs_idx, ys_idx = extract_trajectory(loci, joint=-1)

        assert np.array_equal(xs_name, xs_idx)
        assert np.array_equal(ys_name, ys_idx)
        assert np.array_equal(xs_obj, xs_idx)
        assert np.array_equal(ys_obj, ys_idx)

    def test_unknown_name_raises(self):
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.1, distance=1, name="B")
        pin = pl.Revolute(
            3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="C"
        )
        linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin])
        loci = list(linkage.step(iterations=4))
        with pytest.raises(ValueError, match="not found"):
            extract_trajectory(loci, joint="nonexistent", linkage=linkage)

    def test_top_level_export(self):
        assert pl.extract_trajectory is extract_trajectory


class TestExtractTrajectories:
    def _square_loci(self):
        return [
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.0, 0.0), (0.0, 1.0)),
        ]

    def test_keys_by_index_without_linkage(self):
        t = extract_trajectories(self._square_loci())
        assert set(t.keys()) == {0, 1}

    def test_trajectories_have_correct_values(self):
        t = extract_trajectories(self._square_loci())
        xs0, ys0 = t[0]
        xs1, ys1 = t[1]
        assert np.array_equal(xs0, [0.0, 0.0, 0.0])
        assert np.array_equal(xs1, [1.0, 1.0, 0.0])
        assert np.array_equal(ys1, [0.0, 1.0, 1.0])

    def test_skips_none_per_joint(self):
        loci = [
            ((0.0, 0.0), (1.0, 0.0)),
            ((None, None), (1.0, 1.0)),     # joint 0 bad, joint 1 ok
            ((0.0, 0.0), (None, None)),     # joint 1 bad, joint 0 ok
        ]
        t = extract_trajectories(loci)
        assert t[0][0].size == 2
        assert t[1][0].size == 2
        assert np.array_equal(t[0][0], [0.0, 0.0])
        assert np.array_equal(t[1][0], [1.0, 1.0])

    def test_empty_loci(self):
        assert extract_trajectories([]) == {}

    def test_keys_by_joint_name_with_linkage(self):
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
        pin = pl.Revolute(
            3, 2, joint0=crank, joint1=(3, 0), distance0=3.0, distance1=1.0, name="C",
        )
        linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin])
        loci = list(linkage.step(iterations=12))

        t = extract_trajectories(loci, linkage=linkage)
        assert set(t.keys()) == {"B", "C"}
        # C's trajectory should match extract_trajectory(..., joint=-1)
        xs_expected, _ = extract_trajectory(loci, joint=-1)
        assert np.array_equal(t["C"][0], xs_expected)

    def test_mismatched_linkage_raises(self):
        linkage = type("Fake", (), {"joints": [object(), object(), object()]})()
        with pytest.raises(ValueError, match="has 3 joints but frames"):
            extract_trajectories(self._square_loci(), linkage=linkage)

    def test_top_level_export(self):
        assert pl.extract_trajectories is extract_trajectories
