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
from pylinkage.visualizer.pso_plots import (
    animate_dashboard,
    animate_parallel_coordinates,
    dashboard_layout,
    normalize_data,
    parallel_coordinates_plot,
)


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


def _build_agents(lk, n=5, perturb=0.01):
    constraints = lk.get_constraints()
    coords = [tuple(c.position) for c in lk.components]
    agents = []
    rng = np.random.default_rng(42)
    for i in range(n):
        dims = np.array([c * (1 + perturb * rng.standard_normal()) for c in constraints])
        agents.append((float(i), dims, coords))
    return agents, constraints, coords


class TestNormalizeData:
    def test_auto_bounds(self):
        data = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        out = normalize_data(data)
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_explicit_bounds(self):
        data = np.array([[5.0, 5.0], [10.0, 10.0]])
        out = normalize_data(data, bounds=([0.0, 0.0], [10.0, 20.0]))
        assert out[0, 0] == 0.5
        assert out[1, 1] == 0.5

    def test_zero_range(self):
        data = np.array([[1.0, 1.0], [1.0, 1.0]])
        out = normalize_data(data)
        assert np.all(out == 0.0)


class TestParallelCoordinatesPlot:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=5)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        ax = parallel_coordinates_plot(swarm, dim_names)
        assert ax is not None

    def test_empty_agents(self):
        ax = parallel_coordinates_plot((0, []), ["a"])
        assert ax is not None

    def test_with_bounds(self):
        lk = _fourbar()
        agents, constraints, _ = _build_agents(lk, n=4)
        swarm = (1, agents)
        n_dims = len(constraints)
        dim_names = [f"d{i}" for i in range(n_dims)]
        low = [0.5 * c for c in constraints]
        high = [1.5 * c for c in constraints]
        ax = parallel_coordinates_plot(swarm, dim_names, bounds=(low, high))
        assert ax is not None

    def test_angle_dim_type(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (2, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        dim_types = ["length"] * (n_dims - 1) + ["angle"]
        ax = parallel_coordinates_plot(swarm, dim_names, dim_types=dim_types)
        assert ax is not None

    def test_uniform_scores(self):
        lk = _fourbar()
        agents_raw, _, _ = _build_agents(lk, n=3)
        agents = [(1.0, a[1], a[2]) for a in agents_raw]
        ax = parallel_coordinates_plot(
            (0, agents), [f"d{i}" for i in range(len(agents[0][1]))]
        )
        assert ax is not None

    def test_custom_axes(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        fig, ax = plt.subplots()
        _, cbar_ax = plt.subplots()
        out = parallel_coordinates_plot(
            swarm,
            [f"d{i}" for i in range(len(agents[0][1]))],
            ax=ax,
            cbar_ax=cbar_ax,
        )
        assert out is ax


class TestAnimateParallelCoordinates:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        lk = _fourbar()
        history = []
        for i in range(3):
            agents, _, _ = _build_agents(lk, n=4)
            history.append((i, agents))
        n_dims = len(history[0][1][0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        animation = animate_parallel_coordinates(history, dim_names, interval=50)
        assert isinstance(animation, anim.FuncAnimation)


class TestDashboardLayout:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        history = [2.0, 3.0, 4.0]
        fig = dashboard_layout(lk, swarm, history, dim_names)
        assert fig is not None

    def test_empty_agents(self):
        lk = _fourbar()
        fig = dashboard_layout(lk, (0, []), [], [])
        assert fig is not None

    def test_mixed_types_with_bounds(self):
        lk = _fourbar()
        agents, constraints, _ = _build_agents(lk, n=4)
        swarm = (1, agents)
        n_dims = len(constraints)
        dim_names = [f"d{i}" for i in range(n_dims)]
        dim_types = ["length"] * (n_dims - 1) + ["angle"]
        bounds = ([0.5 * c for c in constraints], [1.5 * c for c in constraints])
        fig = dashboard_layout(
            lk, swarm, [1.0, 2.0], dim_names, dim_types=dim_types, bounds=bounds
        )
        assert fig is not None

    def test_all_angle_types(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        dim_types = ["angle"] * n_dims
        fig = dashboard_layout(lk, swarm, [1.0], dim_names, dim_types=dim_types)
        assert fig is not None

    def test_all_length_types(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        dim_types = ["length"] * n_dims
        fig = dashboard_layout(lk, swarm, [1.0], dim_names, dim_types=dim_types)
        assert fig is not None

    def test_with_dimension_func(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        fig = dashboard_layout(
            lk, swarm, [1.0, 2.0], dim_names, dimension_func=lambda d: list(d)
        )
        assert fig is not None

    def test_unbuildable_best(self):
        lk = _fourbar()
        _, constraints, coords = _build_agents(lk, n=2)
        bad_dims = [0.0001 for _ in constraints]
        agents = [(100.0, np.array(bad_dims), coords)]
        swarm = (0, agents)
        n_dims = len(constraints)
        dim_names = [f"d{i}" for i in range(n_dims)]
        fig = dashboard_layout(lk, swarm, [100.0], dim_names)
        assert fig is not None

    def test_reuse_figure(self):
        lk = _fourbar()
        agents, _, _ = _build_agents(lk, n=4)
        swarm = (0, agents)
        n_dims = len(agents[0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        fig = plt.figure()
        result = dashboard_layout(lk, swarm, [1.0], dim_names, fig=fig)
        assert result is fig


class TestAnimateDashboard:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        lk = _fourbar()
        history = []
        for i in range(3):
            agents, _, _ = _build_agents(lk, n=3)
            history.append((i, agents))
        n_dims = len(history[0][1][0][1])
        dim_names = [f"d{i}" for i in range(n_dims)]
        animation = animate_dashboard(lk, history, dim_names, interval=100)
        assert isinstance(animation, anim.FuncAnimation)
