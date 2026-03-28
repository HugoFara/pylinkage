"""Tests for the drawsvg visualization module."""

import math
import os
import tempfile

import drawsvg as draw
import numpy as np
import pytest

from pylinkage.joints import Crank, Static
from pylinkage.joints.revolute import Pivot
from pylinkage.linkage import Linkage
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

# --------------- fixtures ---------------

@pytest.fixture
def fourbar():
    """Create a simple four-bar linkage and run one step to get loci."""
    A = Static(0, 0, name="A")
    B = Crank(0, 1, joint0=A, distance=1, angle=0.31, name="B")
    C = Static(2, 0, name="C")
    D = Pivot(1.5, 1.5, joint0=B, joint1=C, distance0=2, distance1=2, name="D")
    linkage = Linkage(joints=[A, B, C, D], order=[B, D], name="FourBar")
    loci = list(linkage.step())
    return linkage, loci


@pytest.fixture
def single_frame_loci(fourbar):
    """Return only the first frame of loci."""
    _, loci = fourbar
    return [loci[0]]


# --------------- helper function tests ---------------

class TestWorldToCanvas:
    def test_basic_conversion(self):
        cx, cy = _world_to_canvas(1.0, 2.0, height=500, scale=80, padding=100)
        assert cx == 1.0 * 80 + 100
        assert cy == 500 - (2.0 * 80 + 100)

    def test_origin(self):
        cx, cy = _world_to_canvas(0, 0, height=400, scale=100, padding=50)
        assert cx == 50
        assert cy == 400 - 50


class TestDrawGroundSymbol:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_ground_symbol(d, 100, 100)
        assert len(d.elements) > initial

    def test_custom_color(self):
        d = draw.Drawing(200, 200)
        _draw_ground_symbol(d, 100, 100, color="#FF0000")
        # Should not raise


class TestDrawRevoluteJoint:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_revolute_joint(d, 50, 50)
        assert len(d.elements) > initial

    def test_custom_params(self):
        d = draw.Drawing(200, 200)
        _draw_revolute_joint(d, 50, 50, radius=20, color="#00FF00")


class TestDrawCrankJoint:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_crank_joint(d, 100, 100)
        assert len(d.elements) > initial

    def test_with_angle(self):
        d = draw.Drawing(200, 200)
        _draw_crank_joint(d, 100, 100, angle=math.pi / 4)


class TestDrawSliderJoint:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_slider_joint(d, 100, 100)
        assert len(d.elements) > initial

    def test_with_angle(self):
        d = draw.Drawing(200, 200)
        _draw_slider_joint(d, 100, 100, angle=math.pi / 3)


class TestDrawLink:
    def test_bar_style(self):
        d = draw.Drawing(300, 300)
        initial = len(d.elements)
        _draw_link(d, 50, 50, 200, 200, style=LinkStyle.BAR)
        assert len(d.elements) > initial

    def test_bone_style(self):
        d = draw.Drawing(300, 300)
        initial = len(d.elements)
        _draw_link(d, 50, 50, 200, 200, style=LinkStyle.BONE)
        assert len(d.elements) > initial

    def test_line_style(self):
        d = draw.Drawing(300, 300)
        initial = len(d.elements)
        _draw_link(d, 50, 50, 200, 200, style=LinkStyle.LINE)
        assert len(d.elements) > initial


class TestDrawDimension:
    def test_creates_elements(self):
        d = draw.Drawing(300, 300)
        initial = len(d.elements)
        _draw_dimension(d, 50, 50, 200, 50, "1.50")
        assert len(d.elements) > initial

    def test_zero_length_skipped(self):
        d = draw.Drawing(300, 300)
        initial = len(d.elements)
        _draw_dimension(d, 100, 100, 100, 100, "0.00")
        # Should not add elements for zero-length dimension
        assert len(d.elements) == initial


class TestDrawVelocityArrow:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_velocity_arrow(d, 100, 100, 50, 30)
        assert len(d.elements) > initial

    def test_zero_velocity_skipped(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_velocity_arrow(d, 100, 100, 0, 0)
        assert len(d.elements) == initial

    def test_with_scale(self):
        d = draw.Drawing(200, 200)
        _draw_velocity_arrow(d, 100, 100, 50, 30, scale=2.0)


class TestDrawAccelerationArrow:
    def test_creates_elements(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_acceleration_arrow(d, 100, 100, 40, 20)
        assert len(d.elements) > initial

    def test_zero_acceleration_skipped(self):
        d = draw.Drawing(200, 200)
        initial = len(d.elements)
        _draw_acceleration_arrow(d, 100, 100, 0, 0)
        assert len(d.elements) == initial


# --------------- main function tests ---------------

class TestPlotLinkageSvg:
    def test_returns_drawing(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci)
        assert isinstance(result, draw.Drawing)

    def test_auto_simulation(self, fourbar):
        linkage, _ = fourbar
        result = plot_linkage_svg(linkage)
        assert isinstance(result, draw.Drawing)

    def test_empty_loci_raises(self, fourbar):
        linkage, _ = fourbar
        with pytest.raises(ValueError, match="No loci data"):
            plot_linkage_svg(linkage, loci=[])

    def test_single_frame(self, fourbar, single_frame_loci):
        linkage, _ = fourbar
        result = plot_linkage_svg(linkage, loci=single_frame_loci)
        assert isinstance(result, draw.Drawing)

    def test_show_dimensions(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, show_dimensions=True)
        assert isinstance(result, draw.Drawing)

    def test_hide_loci(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, show_loci=False)
        assert isinstance(result, draw.Drawing)

    def test_hide_labels(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, show_labels=False)
        assert isinstance(result, draw.Drawing)

    def test_link_style_bone(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, link_style="bone")
        assert isinstance(result, draw.Drawing)

    def test_link_style_line(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, link_style="line")
        assert isinstance(result, draw.Drawing)

    def test_custom_scale_and_padding(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, scale=120, padding=50)
        assert isinstance(result, draw.Drawing)

    def test_with_title(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci, title="My Linkage")
        assert isinstance(result, draw.Drawing)

    def test_svg_output_is_string(self, fourbar):
        linkage, loci = fourbar
        result = plot_linkage_svg(linkage, loci)
        svg_str = result.as_svg()
        assert isinstance(svg_str, str)
        assert "<svg" in svg_str


class TestSaveLinkageSvg:
    def test_save_to_file(self, fourbar):
        linkage, loci = fourbar
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            save_linkage_svg(linkage, path, loci=loci)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<svg" in content
        finally:
            os.unlink(path)

    def test_save_with_kwargs(self, fourbar):
        linkage, loci = fourbar
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            save_linkage_svg(linkage, path, loci=loci, show_dimensions=True, link_style="line")
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestPlotLinkageSvgWithVelocity:
    def _make_positions_velocities(self, fourbar):
        linkage, _ = fourbar
        positions = np.array([[0, 0], [0.95, 0.31], [2, 0], [1.2, 1.8]], dtype=np.float64)
        velocities = np.array([[0, 0], [3.1, 9.5], [0, 0], [-2.0, 1.5]], dtype=np.float64)
        return linkage, positions, velocities

    def test_returns_drawing(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(linkage, positions, velocities)
        assert isinstance(result, draw.Drawing)

    def test_custom_velocity_scale(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(
            linkage, positions, velocities, velocity_scale=0.5
        )
        assert isinstance(result, draw.Drawing)

    def test_skip_static_false(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(
            linkage, positions, velocities, skip_static=False
        )
        assert isinstance(result, draw.Drawing)

    def test_with_title(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(
            linkage, positions, velocities, title="Velocity Diagram"
        )
        assert isinstance(result, draw.Drawing)

    def test_hide_labels(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(
            linkage, positions, velocities, show_labels=False
        )
        assert isinstance(result, draw.Drawing)

    def test_link_style_bone(self, fourbar):
        linkage, positions, velocities = self._make_positions_velocities(fourbar)
        result = plot_linkage_svg_with_velocity(
            linkage, positions, velocities, link_style="bone"
        )
        assert isinstance(result, draw.Drawing)

    def test_nan_velocity_handled(self, fourbar):
        linkage, positions, _ = self._make_positions_velocities(fourbar)
        velocities = np.array(
            [[np.nan, np.nan], [3.1, 9.5], [np.nan, np.nan], [-2.0, 1.5]],
            dtype=np.float64,
        )
        result = plot_linkage_svg_with_velocity(linkage, positions, velocities)
        assert isinstance(result, draw.Drawing)


class TestSaveLinkageSvgWithVelocity:
    def test_save_to_file(self, fourbar):
        linkage, _ = fourbar
        positions = np.array([[0, 0], [0.95, 0.31], [2, 0], [1.2, 1.8]], dtype=np.float64)
        velocities = np.array([[0, 0], [3.1, 9.5], [0, 0], [-2.0, 1.5]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            save_linkage_svg_with_velocity(linkage, path, positions, velocities)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<svg" in content
        finally:
            os.unlink(path)
