from __future__ import annotations

import math

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRPDyad, RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.core import (
    COLOR_SWITCHER,
    _get_color,
    build_connections,
    get_components,
    get_parent_pairs,
    is_prismatic_like,
    is_revolute_like,
    is_static_like,
    resolve_component,
)


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar"), [O1, O2, crank, rocker]


def _slider_crank():
    O1 = Ground(0.0, 0.0)
    L1 = Ground(0.0, -1.0)
    L2 = Ground(4.0, -1.0)
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=L1,
        line_anchor2=L2,
        distance=2.0,
    )
    return Linkage([O1, L1, L2, crank, slider]), [O1, L1, L2, crank, slider]


class TestIsStaticLike:
    def test_ground(self):
        assert is_static_like(Ground(0, 0))

    def test_non_ground(self):
        g = Ground(0, 0)
        c = Crank(anchor=g, radius=1.0, angular_velocity=0.1)
        assert not is_static_like(c)


class TestIsRevoluteLike:
    def test_rrr(self):
        a = Ground(0, 0)
        b = Ground(2, 0)
        c = Crank(anchor=a, radius=0.5, angular_velocity=0.1)
        r = RRRDyad(anchor1=c.output, anchor2=b, distance1=1.5, distance2=1.5)
        assert is_revolute_like(r)

    def test_ground_not_revolute(self):
        assert not is_revolute_like(Ground(0, 0))


class TestIsPrismaticLike:
    def test_rrp(self):
        _, parts = _slider_crank()
        assert is_prismatic_like(parts[-1])

    def test_ground_not_prismatic(self):
        assert not is_prismatic_like(Ground(0, 0))


class TestGetColor:
    def test_known(self):
        assert _get_color(Ground(0, 0)) == "k"

    def test_unknown(self):
        class Weird:
            pass

        assert _get_color(Weird()) == ""

    def test_color_switcher_has_entries(self):
        assert "Ground" in COLOR_SWITCHER
        assert "Crank" in COLOR_SWITCHER


class TestGetComponents:
    def test_returns_components_list(self):
        lk, parts = _fourbar()
        comps = get_components(lk)
        assert len(comps) == len(parts)


class TestGetParentPairs:
    def test_crank_has_one_parent(self):
        g = Ground(0, 0)
        c = Crank(anchor=g, radius=1.0, angular_velocity=0.1)
        parents = get_parent_pairs(c)
        assert len(parents) == 1

    def test_rrr_has_two_parents(self):
        a = Ground(0, 0)
        b = Ground(2, 0)
        c = Crank(anchor=a, radius=0.5, angular_velocity=0.1)
        r = RRRDyad(anchor1=c.output, anchor2=b, distance1=1.5, distance2=1.5)
        parents = get_parent_pairs(r)
        assert len(parents) == 2

    def test_ground_has_no_parents(self):
        assert get_parent_pairs(Ground(0, 0)) == []

    def test_legacy_joint_attrs(self):
        class LegacyRevolute:
            def __init__(self):
                self.joint0 = "p0"
                self.joint1 = "p1"

        # Not revolute_like by class name -> only joint0 counted
        parents = get_parent_pairs(LegacyRevolute())
        assert parents == ["p0"]


class TestResolveComponent:
    def test_direct_membership(self):
        g = Ground(0, 0)
        comps = [g, Ground(1, 1)]
        assert resolve_component(g, comps) == 0

    def test_anchor_proxy(self):
        a = Ground(0, 0)
        c = Crank(anchor=a, radius=1.0, angular_velocity=0.1)
        # c.output has _parent pointing at c
        assert resolve_component(c.output, [a, c]) == 1

    def test_not_found_returns_none(self):
        assert resolve_component(Ground(0, 0), []) is None


class TestBuildConnections:
    def test_fourbar(self):
        lk, parts = _fourbar()
        pairs = build_connections(lk, parts)
        assert len(pairs) >= 2
        # every index should be within range
        for p, j in pairs:
            assert 0 <= p < len(parts)
            assert 0 <= j < len(parts)

    def test_slider_crank(self):
        lk, parts = _slider_crank()
        pairs = build_connections(lk, parts)
        assert pairs

    def test_mechanism_branch(self):
        # Build a minimal Mechanism-like object
        class FakeLink:
            def __init__(self, joints):
                self.joints = joints

        a, b, c = Ground(0, 0), Ground(1, 1), Ground(2, 0)
        comps = [a, b, c]

        class FakeMech:
            links = [FakeLink([a, b, c]), FakeLink([a]), FakeLink([])]

        pairs = build_connections(FakeMech(), comps)
        # ternary triangle -> 3 pairs
        assert len(pairs) == 3
