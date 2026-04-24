from __future__ import annotations

import math

import drawsvg as draw
import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRPDyad, RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.drawsvg_viz import (
    _draw_acceleration_arrow,
    _draw_crank_joint,
    _draw_dimension,
    _draw_ground_symbol,
    _draw_link,
    _draw_revolute_joint,
    _draw_slider_joint,
    _draw_velocity_arrow,
    _world_to_canvas,
    plot_linkage_svg,
    plot_linkage_svg_with_velocity,
    save_linkage_svg,
    save_linkage_svg_with_velocity,
)
from pylinkage.visualizer.symbols import LinkStyle


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar"), crank


def _slider_crank():
    O1 = Ground(0.0, 0.0, name="O1")
    L1 = Ground(0.0, -1.0, name="L1")
    L2 = Ground(4.0, -1.0, name="L2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=L1,
        line_anchor2=L2,
        distance=2.0,
        name="slider",
    )
    return Linkage([O1, L1, L2, crank, slider], name="SlideCrank"), crank


def _get_loci(lk, n=10):
    return [tuple(frame) for frame in lk.step(iterations=n)]


class TestWorldToCanvas:
    def test_basic(self):
        cx, cy = _world_to_canvas(1.0, 2.0, 500, 50, 100)
        assert cx == 150.0
        assert cy == 500 - (2.0 * 50 + 100)


class TestDrawGroundSymbol:
    def test_adds_elements(self):
        d = draw.Drawing(200, 200)
        _draw_ground_symbol(d, 100, 100)
        assert len(d.elements) > 0


class TestDrawRevoluteJoint:
    def test_adds(self):
        d = draw.Drawing(200, 200)
        _draw_revolute_joint(d, 50, 50)
        assert len(d.elements) == 2


class TestDrawCrankJoint:
    def test_adds(self):
        d = draw.Drawing(200, 200)
        _draw_crank_joint(d, 100, 100, angle=math.pi / 4)
        assert len(d.elements) > 0


class TestDrawSliderJoint:
    def test_basic(self):
        d = draw.Drawing(200, 200)
        _draw_slider_joint(d, 100, 100, angle=0.5)
        assert len(d.elements) > 0


class TestDrawLink:
    def test_bar_style(self):
        d = draw.Drawing(200, 200)
        _draw_link(d, 0, 0, 100, 50, style=LinkStyle.BAR)
        assert len(d.elements) >= 1

    def test_bone_style(self):
        d = draw.Drawing(200, 200)
        _draw_link(d, 0, 0, 100, 50, style=LinkStyle.BONE)
        assert len(d.elements) >= 1

    def test_line_style(self):
        d = draw.Drawing(200, 200)
        _draw_link(d, 0, 0, 100, 50, style=LinkStyle.LINE)
        assert len(d.elements) == 1


class TestDrawDimension:
    def test_basic(self):
        d = draw.Drawing(200, 200)
        _draw_dimension(d, 0, 0, 100, 50, "10.00")
        assert len(d.elements) > 0

    def test_zero_length_skipped(self):
        d = draw.Drawing(200, 200)
        _draw_dimension(d, 50, 50, 50, 50, "0")
        assert len(d.elements) == 0


class TestDrawVelocityArrow:
    def test_basic(self):
        d = draw.Drawing(200, 200)
        _draw_velocity_arrow(d, 50, 50, 30, 40)
        assert len(d.elements) > 0

    def test_zero_magnitude(self):
        d = draw.Drawing(200, 200)
        _draw_velocity_arrow(d, 50, 50, 0, 0)
        assert len(d.elements) == 0


class TestDrawAccelerationArrow:
    def test_basic(self):
        d = draw.Drawing(200, 200)
        _draw_acceleration_arrow(d, 50, 50, 30, 40)
        assert len(d.elements) > 0

    def test_zero(self):
        d = draw.Drawing(200, 200)
        _draw_acceleration_arrow(d, 50, 50, 0, 0)
        assert len(d.elements) == 0


class TestPlotLinkageSvg:
    def test_basic(self):
        lk, _ = _fourbar()
        d = plot_linkage_svg(lk)
        assert isinstance(d, draw.Drawing)

    def test_with_loci(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci)
        assert len(d.elements) > 0

    def test_with_title(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, title="Fourbar")
        assert len(d.elements) > 0

    def test_show_dimensions(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, show_dimensions=True)
        assert len(d.elements) > 0

    def test_bone_style(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, link_style="bone")
        assert len(d.elements) > 0

    def test_line_style(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, link_style="line")
        assert len(d.elements) > 0

    def test_unknown_style_falls_back(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, link_style="unknown")  # type: ignore[arg-type]
        assert len(d.elements) > 0

    def test_no_loci_drawing(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, show_loci=False)
        assert len(d.elements) > 0

    def test_no_labels(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci, show_labels=False)
        assert len(d.elements) > 0

    def test_empty_loci_raises(self):
        lk, _ = _fourbar()
        with pytest.raises(ValueError):
            plot_linkage_svg(lk, loci=[])

    def test_slider_crank(self):
        lk, _ = _slider_crank()
        loci = _get_loci(lk)
        d = plot_linkage_svg(lk, loci=loci)
        assert len(d.elements) > 0


class TestSaveLinkageSvg:
    def test_basic(self, tmp_path):
        lk, _ = _fourbar()
        path = str(tmp_path / "out.svg")
        save_linkage_svg(lk, path)
        contents = (tmp_path / "out.svg").read_text()
        assert "<svg" in contents

    def test_with_loci_kwargs(self, tmp_path):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        path = str(tmp_path / "out2.svg")
        save_linkage_svg(lk, path, loci=loci, title="T", show_dimensions=True)
        assert (tmp_path / "out2.svg").exists()


class TestPlotLinkageSvgWithVelocity:
    def test_basic(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0])
        assert isinstance(d, draw.Drawing)

    def test_custom_scale(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(
            lk, pos[0], vel[0], velocity_scale=0.2, title="KN"
        )
        assert len(d.elements) > 0

    def test_bone_style(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0], link_style="bone")
        assert len(d.elements) > 0

    def test_line_style(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0], link_style="line")
        assert len(d.elements) > 0

    def test_no_labels(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0], show_labels=False)
        assert len(d.elements) > 0

    def test_no_skip_static(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0], skip_static=False)
        assert len(d.elements) > 0

    def test_list_inputs(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        # pass as lists of tuples to exercise asarray conversion
        positions = [(float(p[0]), float(p[1])) for p in pos[0]]
        velocities = [(float(v[0]), float(v[1])) for v in vel[0]]
        d = plot_linkage_svg_with_velocity(lk, positions, velocities)
        assert len(d.elements) > 0

    def test_slider_crank(self):
        lk, crank = _slider_crank()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        d = plot_linkage_svg_with_velocity(lk, pos[0], vel[0])
        assert len(d.elements) > 0


class _LegacyBase:
    def __init__(self, x, y, name="", joint0=None, joint1=None):
        self._x = x
        self._y = y
        self.name = name
        self.joint0 = joint0
        self.joint1 = joint1

    def coord(self):
        return (self._x, self._y)


class Static(_LegacyBase):
    def __init__(self, x, y, name=""):
        super().__init__(x, y, name=name)


class Revolute(_LegacyBase):
    def __init__(self, x, y, name, j0, j1):
        super().__init__(x, y, name=name, joint0=j0, joint1=j1)


class _LegacyCrankBase(_LegacyBase):
    def __init__(self, x, y, name, j0):
        super().__init__(x, y, name=name, joint0=j0)


_LegacyCrankBase.__name__ = "Crank"


class Prismatic(_LegacyBase):
    def __init__(self, x, y, name, j0, j1, j2):
        super().__init__(x, y, name=name, joint0=j0, joint1=j1)
        self.joint2 = j2

    def coord(self):
        return (self._x, self._y)


_LegacyStatic = Static
_LegacyRevolute = Revolute
_LegacyCrank = _LegacyCrankBase
_LegacyLinear = Prismatic


class _LegacyLinkage:
    def __init__(self, joints, name="legacy"):
        self.joints = joints
        self.name = name

    def step(self, iterations=None, dt=1.0):
        # Produce a handful of frames just by perturbing coords.
        for _ in range(10):
            yield tuple(j.coord() for j in self.joints)


class TestPlotSvgLegacyJoints:
    def test_revolute_link(self):
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        s2 = _LegacyStatic(3.0, 0.0, name="S2")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        r = _LegacyRevolute(1.5, 1.0, "R", c, s2)
        lk = _LegacyLinkage([s1, s2, c, r])
        d = plot_linkage_svg(lk, show_dimensions=True)
        assert isinstance(d, draw.Drawing)

    def test_prismatic_path(self):
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        s2 = _LegacyStatic(4.0, 0.0, name="S2")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        lin = _LegacyLinear(2.0, 0.0, "L", c, s1, s2)
        lk = _LegacyLinkage([s1, s2, c, lin])
        d = plot_linkage_svg(lk)
        assert isinstance(d, draw.Drawing)

    def test_implicit_static_parent(self):
        external_static = _LegacyStatic(5.0, 5.0, name="ext")
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        r = _LegacyRevolute(1.5, 1.0, "R", c, external_static)
        # Note: external_static NOT in lk.joints → triggers get_position fallback
        lk = _LegacyLinkage([s1, c, r])
        d = plot_linkage_svg(lk)
        assert isinstance(d, draw.Drawing)


class TestPlotSvgWithVelocityLegacyJoints:
    def test_revolute_link(self):
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        s2 = _LegacyStatic(3.0, 0.0, name="S2")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        r = _LegacyRevolute(1.5, 1.0, "R", c, s2)
        lk = _LegacyLinkage([s1, s2, c, r])
        positions = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [1.5, 1.0]])
        velocities = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.2], [0.2, 0.3]])
        d = plot_linkage_svg_with_velocity(lk, positions, velocities)
        assert isinstance(d, draw.Drawing)

    def test_implicit_static_parent(self):
        external = _LegacyStatic(5.0, 5.0, name="ext")
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        r = _LegacyRevolute(1.5, 1.0, "R", c, external)
        lk = _LegacyLinkage([s1, c, r])
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 1.0]])
        velocities = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]])
        d = plot_linkage_svg_with_velocity(lk, positions, velocities)
        assert isinstance(d, draw.Drawing)

    def test_nan_coord_skips(self):
        bad = _LegacyStatic(None, None, name="bad")  # type: ignore[arg-type]
        s1 = _LegacyStatic(0.0, 0.0, name="S1")
        c = _LegacyCrank(1.0, 0.0, "C", s1)
        r = _LegacyRevolute(1.5, 1.0, "R", c, bad)
        lk = _LegacyLinkage([s1, c, r])
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 1.0]])
        velocities = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]])
        d = plot_linkage_svg_with_velocity(lk, positions, velocities)
        assert isinstance(d, draw.Drawing)


class TestSaveLinkageSvgWithVelocity:
    def test_basic(self, tmp_path):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        path = str(tmp_path / "kin.svg")
        save_linkage_svg_with_velocity(lk, path, pos[0], vel[0])
        assert (tmp_path / "kin.svg").exists()

    def test_with_kwargs(self, tmp_path):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        pos, vel, _ = lk.step_fast_with_kinematics(iterations=5)
        path = str(tmp_path / "kin2.svg")
        save_linkage_svg_with_velocity(
            lk, path, pos[0], vel[0], title="T", velocity_scale=0.3
        )
        contents = (tmp_path / "kin2.svg").read_text()
        assert "<svg" in contents
