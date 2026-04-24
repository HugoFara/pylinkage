from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.animated import (
    ANIMATIONS,
    plot_kinematic_linkage,
    show_linkage,
    swarm_tiled_repr,
    update_animated_plot,
)


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


def _get_loci(lk, n=10):
    return [tuple(frame) for frame in lk.step(iterations=n)]


class TestUpdateAnimatedPlot:
    def teardown_method(self):
        plt.close("all")

    def test_updates_lines(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        animation = plot_kinematic_linkage(lk, fig, ax, loci, frames=5)
        from pylinkage.visualizer.core import build_connections, get_components

        components = get_components(lk)
        connections = build_connections(lk, components)
        lines = []
        for _ in connections:
            (line,) = ax.plot([], [])
            lines.append(line)
        result = update_animated_plot(lk, 0, lines, loci)
        assert len(result) == len(lines)
        del animation


class TestPlotKinematicLinkage:
    def teardown_method(self):
        plt.close("all")

    def test_returns_animation(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        animation = plot_kinematic_linkage(lk, fig, ax, loci, frames=5)
        assert isinstance(animation, anim.FuncAnimation)

    def test_interval_override(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        animation = plot_kinematic_linkage(lk, fig, ax, loci, frames=3, interval=10)
        assert isinstance(animation, anim.FuncAnimation)


class TestShowLinkage:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        lk = _fourbar()
        animation = show_linkage(lk, points=10, duration=0.01, fps=10)
        assert isinstance(animation, anim.FuncAnimation)

    def test_with_loci(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        lk = _fourbar()
        loci = _get_loci(lk)
        animation = show_linkage(lk, loci=loci, duration=0.01, fps=10)
        assert isinstance(animation, anim.FuncAnimation)

    def test_with_title(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        lk = _fourbar()
        animation = show_linkage(lk, points=10, duration=0.01, fps=10, title="Custom Title")
        assert isinstance(animation, anim.FuncAnimation)

    def test_with_prev(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        lk = _fourbar()
        prev = [(0.0, 0.0), (3.0, 0.0), (1.0, 0.0), (1.5, 1.0)]
        animation = show_linkage(lk, prev=prev, points=10, duration=0.01, fps=10)
        assert isinstance(animation, anim.FuncAnimation)


class TestSwarmTiledRepr:
    def teardown_method(self):
        plt.close("all")

    def _build_swarm(self, lk, n_agents=4):
        constraints = lk.get_constraints()
        coords = [tuple(c.position) for c in lk.components]
        agents = []
        for i in range(n_agents):
            dims = [c * (1 + 0.01 * i) for c in constraints]
            agents.append((float(i), np.array(dims), coords))
        return (0, agents)

    def test_basic(self):
        lk = _fourbar()
        swarm = self._build_swarm(lk, n_agents=4)
        fig, axes = plt.subplots(2, 2)
        swarm_tiled_repr(lk, swarm, fig, axes, points=5, iteration_factor=1)

    def test_with_dimension_func(self):
        lk = _fourbar()
        swarm = self._build_swarm(lk, n_agents=4)
        fig, axes = plt.subplots(2, 2)
        swarm_tiled_repr(
            lk,
            swarm,
            fig,
            axes,
            dimension_func=lambda d: list(d),
            points=5,
        )

    def test_unbuildable_skipped(self):
        lk = _fourbar()
        constraints = lk.get_constraints()
        coords = [tuple(c.position) for c in lk.components]
        # Impossible constraints: tiny rocker
        bad_dims = [0.001 for _ in constraints]
        agents = [
            (1.0, np.array(constraints), coords),
            (0.5, np.array(bad_dims), coords),
        ]
        swarm = (0, agents)
        fig, axes = plt.subplots(1, 2)
        swarm_tiled_repr(lk, swarm, fig, axes, points=5)


def test_animations_list_exists():
    assert isinstance(ANIMATIONS, list)
