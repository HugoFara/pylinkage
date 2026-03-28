"""Tests for the plotly visualization module."""

import pytest
import plotly.graph_objects as go

from pylinkage.joints import Static, Crank
from pylinkage.joints.revolute import Pivot
from pylinkage.linkage import Linkage
from pylinkage.visualizer.plotly_viz import (
    _get_plotly_marker,
    animate_linkage_plotly,
    plot_linkage_plotly,
)
from pylinkage.visualizer.symbols import SymbolType


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

class TestGetPlotlyMarker:
    def test_ground(self):
        m = _get_plotly_marker(SymbolType.GROUND)
        assert m["symbol"] == "triangle-down"
        assert "size" in m

    def test_crank(self):
        m = _get_plotly_marker(SymbolType.CRANK)
        assert m["symbol"] == "circle"
        assert m["size"] == 16

    def test_slider(self):
        m = _get_plotly_marker(SymbolType.SLIDER)
        assert m["symbol"] == "square"

    def test_revolute(self):
        m = _get_plotly_marker(SymbolType.REVOLUTE)
        assert m["symbol"] == "circle"
        assert m["size"] == 14

    def test_fixed(self):
        m = _get_plotly_marker(SymbolType.FIXED)
        assert m["symbol"] == "circle"


# --------------- plot_linkage_plotly tests ---------------

class TestPlotLinkagePlotly:
    def test_returns_figure(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci)
        assert isinstance(fig, go.Figure)

    def test_auto_simulation(self, fourbar):
        linkage, _ = fourbar
        fig = plot_linkage_plotly(linkage)
        assert isinstance(fig, go.Figure)

    def test_empty_loci_raises(self, fourbar):
        linkage, _ = fourbar
        with pytest.raises(ValueError, match="No loci data"):
            plot_linkage_plotly(linkage, loci=[])

    def test_single_frame_no_loci_paths(self, fourbar, single_frame_loci):
        linkage, _ = fourbar
        fig = plot_linkage_plotly(linkage, loci=single_frame_loci)
        assert isinstance(fig, go.Figure)

    def test_show_dimensions(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci, show_dimensions=True)
        assert isinstance(fig, go.Figure)
        # Dimensions should add annotations
        assert len(fig.layout.annotations) > 0

    def test_hide_loci(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci, show_loci=False)
        assert isinstance(fig, go.Figure)

    def test_hide_labels(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci, show_labels=False)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_default_title_from_linkage_name(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci)
        assert fig.layout.title.text == "FourBar"

    def test_custom_dimensions(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci, width=1200, height=900)
        assert fig.layout.width == 1200
        assert fig.layout.height == 900

    def test_has_traces(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci)
        # Should have at least link traces + joint traces
        assert len(fig.data) > 0

    def test_equal_axis_scaling(self, fourbar):
        linkage, loci = fourbar
        fig = plot_linkage_plotly(linkage, loci)
        assert fig.layout.xaxis.scaleanchor == "y"


# --------------- animate_linkage_plotly tests ---------------

class TestAnimateLinkagePlotly:
    def test_returns_figure(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci)
        assert isinstance(fig, go.Figure)

    def test_auto_simulation(self, fourbar):
        linkage, _ = fourbar
        fig = animate_linkage_plotly(linkage)
        assert isinstance(fig, go.Figure)

    def test_empty_loci_raises(self, fourbar):
        linkage, _ = fourbar
        with pytest.raises(ValueError, match="No loci data"):
            animate_linkage_plotly(linkage, loci=[])

    def test_has_frames(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci)
        assert len(fig.frames) > 0
        assert len(fig.frames) == len(loci)

    def test_has_play_pause_buttons(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci)
        menus = fig.layout.updatemenus
        assert len(menus) > 0
        buttons = menus[0].buttons
        labels = [b.label for b in buttons]
        assert "Play" in labels
        assert "Pause" in labels

    def test_has_slider(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci)
        assert len(fig.layout.sliders) > 0

    def test_hide_loci(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci, show_loci=False)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci, title="Animated")
        assert fig.layout.title.text == "Animated"

    def test_custom_frame_duration(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci, frame_duration=100)
        assert isinstance(fig, go.Figure)

    def test_custom_size(self, fourbar):
        linkage, loci = fourbar
        fig = animate_linkage_plotly(linkage, loci, width=640, height=480)
        assert fig.layout.width == 640
        assert fig.layout.height == 480

    def test_single_frame(self, fourbar, single_frame_loci):
        linkage, _ = fourbar
        fig = animate_linkage_plotly(linkage, loci=single_frame_loci)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 1
