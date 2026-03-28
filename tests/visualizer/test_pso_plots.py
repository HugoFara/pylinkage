"""Tests for pylinkage.visualizer.pso_plots module."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

import pylinkage as pl
from pylinkage.joints import Crank, Revolute
from pylinkage.visualizer.pso_plots import (
    animate_dashboard,
    animate_parallel_coordinates,
    dashboard_layout,
    normalize_data,
    parallel_coordinates_plot,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Helpers to build fake PSO data structures
# --------------------------------------------------------------------------

def _make_agent(score, dims, n_joints=2):
    """Build an Agent tuple: (score, dimensions, initial_positions)."""
    positions = [(float(i), float(i + 1)) for i in range(n_joints)]
    return (score, list(dims), positions)


def _make_swarm(iteration, n_particles=10, n_dims=3, rng=None):
    """Build a Swarm tuple: (iteration, [Agent, ...])."""
    if rng is None:
        rng = np.random.default_rng(42)
    agents = []
    for _ in range(n_particles):
        dims = rng.uniform(0.5, 5.0, size=n_dims).tolist()
        score = float(rng.uniform(-1, 1))
        agents.append(_make_agent(score, dims))
    return (iteration, agents)


def _make_history(n_iters=3, n_particles=10, n_dims=3):
    """Build a History list of swarms."""
    rng = np.random.default_rng(123)
    return [_make_swarm(i, n_particles, n_dims, rng) for i in range(n_iters)]


def _make_four_bar():
    """Return a simple four-bar linkage and loci."""
    crank = Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="Crank")
    pin = Revolute(
        3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="Pin"
    )
    linkage = pl.Linkage(
        joints=[crank, pin], order=[crank, pin], name="TestFourBar"
    )
    linkage.rebuild()
    return linkage


# --------------------------------------------------------------------------
# normalize_data
# --------------------------------------------------------------------------

class TestNormalizeData:
    def test_basic_normalization(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = normalize_data(data)
        assert result.shape == data.shape
        np.testing.assert_allclose(result.min(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(result.max(axis=0), [1.0, 1.0])

    def test_with_bounds(self):
        data = np.array([[2.0, 3.0], [4.0, 5.0]])
        bounds = ([0.0, 0.0], [10.0, 10.0])
        result = normalize_data(data, bounds)
        np.testing.assert_allclose(result, [[0.2, 0.3], [0.4, 0.5]])

    def test_constant_column(self):
        """If a column has zero range, avoid division by zero."""
        data = np.array([[5.0, 1.0], [5.0, 2.0]])
        result = normalize_data(data)
        # Column 0 has range 0 -> treated as 1 -> (5-5)/1 = 0
        assert result[0, 0] == 0.0
        assert result[1, 0] == 0.0

    def test_single_row(self):
        data = np.array([[3.0, 7.0]])
        result = normalize_data(data)
        # Single row means min==max, ranges become 1
        np.testing.assert_allclose(result, [[0.0, 0.0]])


# --------------------------------------------------------------------------
# parallel_coordinates_plot
# --------------------------------------------------------------------------

class TestParallelCoordinatesPlot:
    def test_basic_call(self):
        swarm = _make_swarm(0, n_particles=15, n_dims=3)
        dim_names = ["d1", "d2", "d3"]
        ax = parallel_coordinates_plot(swarm, dim_names)
        assert ax is not None

    def test_with_existing_axes(self):
        swarm = _make_swarm(1, n_particles=10, n_dims=2)
        dim_names = ["a", "b"]
        fig, ax = plt.subplots()
        result_ax = parallel_coordinates_plot(swarm, dim_names, ax=ax)
        assert result_ax is ax

    def test_with_dim_types(self):
        swarm = _make_swarm(2, n_particles=10, n_dims=3)
        dim_names = ["len1", "ang1", "len2"]
        dim_types = ["length", "angle", "length"]
        ax = parallel_coordinates_plot(swarm, dim_names, dim_types=dim_types)
        assert ax is not None

    def test_with_bounds(self):
        swarm = _make_swarm(3, n_particles=8, n_dims=3)
        dim_names = ["d1", "d2", "d3"]
        bounds = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        ax = parallel_coordinates_plot(swarm, dim_names, bounds=bounds)
        assert ax is not None

    def test_custom_cmap_and_alpha(self):
        swarm = _make_swarm(4, n_particles=10, n_dims=2)
        dim_names = ["x", "y"]
        ax = parallel_coordinates_plot(
            swarm, dim_names, cmap="plasma", alpha=0.5, highlight_best=3
        )
        assert ax is not None

    def test_empty_swarm(self):
        swarm = (0, [])
        dim_names = ["d1", "d2"]
        ax = parallel_coordinates_plot(swarm, dim_names)
        assert ax is not None

    def test_single_particle(self):
        agent = _make_agent(0.5, [1.0, 2.0])
        swarm = (0, [agent])
        dim_names = ["d1", "d2"]
        ax = parallel_coordinates_plot(swarm, dim_names, highlight_best=1)
        assert ax is not None

    def test_with_colorbar_axes(self):
        """Use a dedicated colorbar axes (cbar_ax parameter)."""
        swarm = _make_swarm(0, n_particles=10, n_dims=2)
        dim_names = ["d1", "d2"]
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0])
        cbar_ax = fig.add_subplot(gs[1])
        result = parallel_coordinates_plot(
            swarm, dim_names, ax=ax, cbar_ax=cbar_ax
        )
        assert result is ax

    def test_all_same_scores(self):
        """All agents have the same score -> norm_scores should be 0.5."""
        agents = [_make_agent(1.0, [float(i), float(i + 1)]) for i in range(5)]
        swarm = (0, agents)
        dim_names = ["d1", "d2"]
        ax = parallel_coordinates_plot(swarm, dim_names)
        assert ax is not None


# --------------------------------------------------------------------------
# animate_parallel_coordinates
# --------------------------------------------------------------------------

class TestAnimateParallelCoordinates:
    def test_basic_animation(self):
        history = _make_history(n_iters=3, n_particles=5, n_dims=2)
        dim_names = ["d1", "d2"]
        animation = animate_parallel_coordinates(history, dim_names)
        assert animation is not None

    def test_with_dim_types_and_bounds(self):
        history = _make_history(n_iters=2, n_particles=5, n_dims=3)
        dim_names = ["len1", "ang1", "len2"]
        dim_types = ["length", "angle", "length"]
        bounds = ([0.0, 0.0, 0.0], [10.0, 6.28, 10.0])
        animation = animate_parallel_coordinates(
            history, dim_names, dim_types=dim_types, bounds=bounds,
            interval=100, cmap="coolwarm",
        )
        assert animation is not None


# --------------------------------------------------------------------------
# dashboard_layout
# --------------------------------------------------------------------------

class TestDashboardLayout:
    def test_basic_dashboard(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(0, n_particles=10, n_dims=3)
        score_history = [0.1, 0.3, 0.5]
        dim_names = ["d1", "d2", "d3"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
        )
        assert fig is not None

    def test_with_dim_types_length_only(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(1, n_particles=8, n_dims=3)
        score_history = [0.2, 0.4]
        dim_names = ["len1", "len2", "len3"]
        dim_types = ["length", "length", "length"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            dim_types=dim_types,
        )
        assert fig is not None

    def test_with_angle_types(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(2, n_particles=8, n_dims=3)
        score_history = [0.1, 0.2, 0.3]
        dim_names = ["len1", "ang1", "len2"]
        dim_types = ["length", "angle", "length"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            dim_types=dim_types,
        )
        assert fig is not None

    def test_with_bounds(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(3, n_particles=6, n_dims=3)
        score_history = [0.5]
        dim_names = ["d1", "d2", "d3"]
        bounds = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            bounds=bounds,
        )
        assert fig is not None

    def test_empty_swarm(self):
        linkage = _make_four_bar()
        swarm = (0, [])
        score_history = [0.0]
        dim_names = ["d1", "d2", "d3"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
        )
        assert fig is not None

    def test_with_existing_figure(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(0, n_particles=5, n_dims=3)
        score_history = [0.1, 0.2]
        dim_names = ["d1", "d2", "d3"]
        existing_fig = plt.figure(figsize=(10, 8))
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            fig=existing_fig,
        )
        assert fig is existing_fig

    def test_single_score_history(self):
        """Single-element score_history should not show improvement text."""
        linkage = _make_four_bar()
        swarm = _make_swarm(0, n_particles=5, n_dims=3)
        score_history = [0.5]
        dim_names = ["d1", "d2", "d3"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
        )
        assert fig is not None

    def test_with_dimension_func(self):
        linkage = _make_four_bar()
        swarm = _make_swarm(0, n_particles=5, n_dims=3)
        score_history = [0.1, 0.3]
        dim_names = ["d1", "d2", "d3"]

        def dim_func(dims):
            return list(dims)

        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            dimension_func=dim_func,
        )
        assert fig is not None

    def test_angle_only_dims(self):
        """All dimensions are angles — no length boxplot data."""
        linkage = _make_four_bar()
        swarm = _make_swarm(0, n_particles=5, n_dims=3)
        score_history = [0.1]
        dim_names = ["ang1", "ang2", "ang3"]
        dim_types = ["angle", "angle", "angle"]
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            dim_types=dim_types,
        )
        assert fig is not None

    def test_angles_with_bounds(self):
        """Angle dims with bounds — exercise the bounds normalization path."""
        linkage = _make_four_bar()
        rng = np.random.default_rng(77)
        agents = [
            _make_agent(float(rng.uniform(0, 1)), rng.uniform(0.5, 3.0, size=3).tolist())
            for _ in range(8)
        ]
        swarm = (0, agents)
        score_history = [0.1, 0.3]
        dim_names = ["ang1", "ang2", "ang3"]
        dim_types = ["angle", "angle", "angle"]
        bounds = ([0.0, 0.0, 0.0], [6.28, 6.28, 6.28])
        fig = dashboard_layout(
            linkage=linkage,
            swarm=swarm,
            score_history=score_history,
            dim_names=dim_names,
            dim_types=dim_types,
            bounds=bounds,
        )
        assert fig is not None


# --------------------------------------------------------------------------
# animate_dashboard
# --------------------------------------------------------------------------

class TestAnimateDashboard:
    def test_basic_animation(self):
        linkage = _make_four_bar()
        history = _make_history(n_iters=2, n_particles=5, n_dims=3)
        dim_names = ["d1", "d2", "d3"]
        animation = animate_dashboard(
            linkage=linkage,
            history=history,
            dim_names=dim_names,
            interval=100,
        )
        assert animation is not None

    def test_with_all_options(self):
        linkage = _make_four_bar()
        history = _make_history(n_iters=2, n_particles=5, n_dims=3)
        dim_names = ["len1", "ang1", "len2"]
        dim_types = ["length", "angle", "length"]
        bounds = ([0.0, 0.0, 0.0], [10.0, 6.28, 10.0])

        def dim_func(dims):
            return list(dims)

        animation = animate_dashboard(
            linkage=linkage,
            history=history,
            dim_names=dim_names,
            dim_types=dim_types,
            bounds=bounds,
            dimension_func=dim_func,
            interval=100,
        )
        assert animation is not None
