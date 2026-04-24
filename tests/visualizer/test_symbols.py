from __future__ import annotations

import math

import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRPDyad, RRRDyad
from pylinkage.visualizer.symbols import (
    LINK_COLORS,
    SYMBOL_SPECS,
    LinkStyle,
    SymbolSpec,
    SymbolType,
    get_link_color,
    get_symbol_spec,
    is_ground_joint,
)


class TestSymbolType:
    def test_all_members_exist(self):
        for name in ["GROUND", "REVOLUTE", "CRANK", "SLIDER", "FIXED", "LINEAR"]:
            assert hasattr(SymbolType, name)

    def test_members_are_unique(self):
        values = [t.value for t in SymbolType]
        assert len(values) == len(set(values))


class TestLinkStyle:
    def test_members(self):
        assert LinkStyle.BAR is not LinkStyle.BONE
        assert LinkStyle.BONE is not LinkStyle.LINE


class TestSymbolSpec:
    def test_frozen_dataclass(self):
        spec = SymbolSpec(SymbolType.GROUND, "#000000", 1.0, (0.0, 0.0))
        with pytest.raises(Exception):
            spec.color = "#ffffff"  # type: ignore[misc]

    def test_fields(self):
        spec = SymbolSpec(SymbolType.CRANK, "#abcdef", 1.2, (0.5, -0.5))
        assert spec.symbol_type is SymbolType.CRANK
        assert spec.color == "#abcdef"
        assert spec.size == 1.2
        assert spec.label_offset == (0.5, -0.5)


class TestSymbolSpecs:
    def test_covers_major_classes(self):
        for name in ["Ground", "Crank", "RRRDyad", "FixedDyad", "RRPDyad", "PPDyad"]:
            assert name in SYMBOL_SPECS

    def test_all_specs_valid(self):
        for spec in SYMBOL_SPECS.values():
            assert isinstance(spec, SymbolSpec)
            assert spec.color.startswith("#")


class TestGetSymbolSpec:
    def test_known_ground(self):
        g = Ground(0, 0)
        spec = get_symbol_spec(g)
        assert spec.symbol_type is SymbolType.GROUND

    def test_known_crank(self):
        g = Ground(0, 0)
        c = Crank(anchor=g, radius=1.0, angular_velocity=0.1)
        spec = get_symbol_spec(c)
        assert spec.symbol_type is SymbolType.CRANK

    def test_known_rrr(self):
        a = Ground(0, 0)
        b = Ground(2, 0)
        c = Crank(anchor=a, radius=0.5, angular_velocity=0.1)
        r = RRRDyad(anchor1=c.output, anchor2=b, distance1=1.5, distance2=1.5)
        spec = get_symbol_spec(r)
        assert spec.symbol_type is SymbolType.REVOLUTE

    def test_unknown_fallback(self):
        class Strange:
            pass

        spec = get_symbol_spec(Strange())
        assert spec.symbol_type is SymbolType.REVOLUTE
        assert spec.color == "#666666"


class TestGetLinkColor:
    def test_cycle(self):
        n = len(LINK_COLORS)
        assert get_link_color(0) == get_link_color(n)
        assert get_link_color(1) == get_link_color(n + 1)

    def test_all_returned(self):
        seen = {get_link_color(i) for i in range(len(LINK_COLORS))}
        assert len(seen) == len(LINK_COLORS)

    def test_negative_index(self):
        # Python % handles negatives; just shouldn't error
        assert get_link_color(-1) in LINK_COLORS


class TestIsGroundJoint:
    def test_ground_instance_is_ground(self):
        assert is_ground_joint(Ground(0, 0)) is True

    def test_crank_is_not_ground(self):
        g = Ground(0, 0)
        c = Crank(anchor=g, radius=1.0, angular_velocity=0.1)
        assert is_ground_joint(c) is False

    def test_fixeddyad_not_ground(self):
        g1 = Ground(0, 0)
        g2 = Ground(2, 0)
        c = Crank(anchor=g1, radius=0.5, angular_velocity=0.1)
        r = RRRDyad(anchor1=c.output, anchor2=g2, distance1=1.5, distance2=1.5)
        fd = FixedDyad(anchor1=c.output, anchor2=r, distance=1.0, angle=math.pi / 4)
        assert is_ground_joint(fd) is False

    def test_static_without_parent_is_ground(self):
        class Static:
            def __init__(self):
                self.joint0 = None

        assert is_ground_joint(Static()) is True

    def test_static_with_parent_not_ground(self):
        class Static:
            def __init__(self, parent):
                self.joint0 = parent

        assert is_ground_joint(Static(object())) is False
