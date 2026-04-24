from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import plotly.graph_objects as go
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.plotly_viz import (
    _get_components,
    _get_parent_pairs,
    _get_plotly_marker,
    _resolve_position,
    animate_linkage_plotly,
    interactive_linkage_plotly,
    plot_linkage_plotly,
    plot_linkage_plotly_with_velocity,
)
from pylinkage.visualizer.symbols import SymbolType


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar"), crank


def _get_loci(lk, n=10):
    return [tuple(frame) for frame in lk.step(iterations=n)]


class _CrankAdapter:
    def __init__(self, real):
        self._real = real
        self.omega = 5.0
        self.name = real.name

    def __getattr__(self, item):
        return getattr(self._real, item)


_CrankAdapter.__name__ = "Crank"


class _Wrap:
    """Wraps a simulation.Linkage for kinematics with velocity."""

    def __init__(self, sim, crank):
        self._sim = sim
        self.name = sim.name
        comps = list(sim.components)
        idx = comps.index(crank)
        comps[idx] = _CrankAdapter(crank)
        self._components = comps

    @property
    def components(self):
        return self._components

    def step(self, *a, **k):
        return self._sim.step(*a, **k)

    def step_fast_with_kinematics(self, iterations=None, dt=1.0):
        return self._sim.step_fast_with_kinematics(iterations=iterations or 10, dt=dt)


class TestGetPlotlyMarker:
    def test_ground(self):
        m = _get_plotly_marker(SymbolType.GROUND)
        assert m["symbol"] == "triangle-down"

    def test_crank(self):
        m = _get_plotly_marker(SymbolType.CRANK)
        assert m["symbol"] == "circle"

    def test_slider(self):
        m = _get_plotly_marker(SymbolType.SLIDER)
        assert m["symbol"] == "square"

    def test_revolute(self):
        m = _get_plotly_marker(SymbolType.REVOLUTE)
        assert m["symbol"] == "circle"


class TestHelpers:
    def test_get_components_delegate(self):
        lk, _ = _fourbar()
        comps = _get_components(lk)
        assert len(comps) == len(lk.components)

    def test_get_parent_pairs_delegate(self):
        _, crank = _fourbar()
        parents = _get_parent_pairs(crank)
        assert len(parents) == 1

    def test_resolve_position_from_map(self):
        g = Ground(1.0, 2.0)
        pos = _resolve_position(g, {id(g): (5.0, 6.0)})
        assert pos == (5.0, 6.0)

    def test_resolve_position_fallback_position(self):
        g = Ground(1.0, 2.0)
        pos = _resolve_position(g, {})
        assert pos[0] == 1.0

    def test_resolve_position_coord_fallback(self):
        class CoordOnly:
            def coord(self):
                return (7.0, 8.0)

        pos = _resolve_position(CoordOnly(), {})
        assert pos == (7.0, 8.0)

    def test_resolve_position_no_attrs(self):
        class Bare:
            pass

        pos = _resolve_position(Bare(), {})
        assert pos == (0.0, 0.0)


class TestPlotLinkagePlotly:
    def test_basic(self):
        lk, _ = _fourbar()
        fig = plot_linkage_plotly(lk)
        assert isinstance(fig, go.Figure)
        assert fig.data

    def test_with_loci(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = plot_linkage_plotly(lk, loci=loci)
        assert isinstance(fig, go.Figure)

    def test_show_dimensions(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = plot_linkage_plotly(lk, loci=loci, show_dimensions=True)
        assert fig.layout.annotations

    def test_no_labels(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = plot_linkage_plotly(lk, loci=loci, show_labels=False)
        assert isinstance(fig, go.Figure)

    def test_no_loci_drawing(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = plot_linkage_plotly(lk, loci=loci, show_loci=False)
        assert isinstance(fig, go.Figure)

    def test_with_title(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = plot_linkage_plotly(lk, loci=loci, title="X", width=500, height=400)
        assert isinstance(fig, go.Figure)

    def test_empty_loci_raises(self):
        lk, _ = _fourbar()
        with pytest.raises(ValueError):
            plot_linkage_plotly(lk, loci=[])


class TestAnimateLinkagePlotly:
    def test_basic(self):
        lk, _ = _fourbar()
        fig = animate_linkage_plotly(lk)
        assert isinstance(fig, go.Figure)
        assert fig.frames

    def test_with_loci(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = animate_linkage_plotly(lk, loci=loci, title="Anim", frame_duration=10)
        assert len(fig.frames) == len(loci)

    def test_no_loci_paths(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = animate_linkage_plotly(lk, loci=loci, show_loci=False)
        assert isinstance(fig, go.Figure)

    def test_size(self):
        lk, _ = _fourbar()
        loci = _get_loci(lk)
        fig = animate_linkage_plotly(lk, loci=loci, width=400, height=300)
        assert fig.layout.width == 400

    def test_empty_loci_raises(self):
        lk, _ = _fourbar()
        with pytest.raises(ValueError):
            animate_linkage_plotly(lk, loci=[])


class TestPlotWithVelocity:
    def test_raises_no_omega(self):
        lk, _ = _fourbar()
        with pytest.raises(ValueError):
            plot_linkage_plotly_with_velocity(lk, frame_index=0)

    def test_bad_frame_index(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        wrapped = _Wrap(lk, crank)
        with pytest.raises(ValueError):
            plot_linkage_plotly_with_velocity(wrapped, frame_index=99999)

    def test_basic(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        wrapped = _Wrap(lk, crank)
        fig = plot_linkage_plotly_with_velocity(wrapped, frame_index=2)
        assert isinstance(fig, go.Figure)

    def test_no_velocity(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        wrapped = _Wrap(lk, crank)
        fig = plot_linkage_plotly_with_velocity(
            wrapped,
            frame_index=0,
            show_loci=False,
            show_velocity=False,
            velocity_scale=0.5,
            title="T",
        )
        assert isinstance(fig, go.Figure)

    def test_with_velocity_scale_and_color(self):
        lk, crank = _fourbar()
        lk.set_input_velocity(crank, omega=5.0)
        wrapped = _Wrap(lk, crank)
        fig = plot_linkage_plotly_with_velocity(
            wrapped,
            frame_index=1,
            velocity_scale=0.3,
            velocity_color="#00ff00",
            width=500,
            height=400,
        )
        assert isinstance(fig, go.Figure)


class TestInteractiveLinkagePlotly:
    def test_no_ipywidgets_raises(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ipywidgets":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        lk, _ = _fourbar()
        with pytest.raises(ImportError):
            interactive_linkage_plotly(lk)

    def test_basic(self):
        pytest.importorskip("ipywidgets")
        lk, _ = _fourbar()
        box = interactive_linkage_plotly(lk, iterations=5)
        assert box is not None

    def test_no_show_loci(self):
        pytest.importorskip("ipywidgets")
        lk, _ = _fourbar()
        box = interactive_linkage_plotly(
            lk, iterations=5, show_loci=False, title="T", width=500, height=400
        )
        assert box is not None
