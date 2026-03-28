"""Tests for pylinkage.visualizer.kinematics module."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

import pylinkage as pl
from pylinkage.joints import Crank, Revolute, Static


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def four_bar():
    """Build a simple four-bar linkage using the legacy joints API."""
    crank = Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="Crank")
    pin = Revolute(
        3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1, name="Pin"
    )
    linkage = pl.Linkage(
        joints=[crank, pin], order=[crank, pin], name="TestFourBar"
    )
    linkage.rebuild()
    return linkage


@pytest.fixture()
def four_bar_positions(four_bar):
    """Positions array (n_joints, 2) for the four-bar's joints."""
    coords = four_bar.get_coords()
    return np.array([[c[0] or 0.0, c[1] or 0.0] for c in coords])


# --------------------------------------------------------------------------
# plot_velocity_vectors
# --------------------------------------------------------------------------

class TestPlotVelocityVectors:
    """Tests for plot_velocity_vectors."""

    def test_basic_call(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        velocities = np.random.default_rng(42).random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, four_bar_positions, velocities
        )
        assert quiver is not None

    def test_skip_static_true(self, four_bar, four_bar_positions):
        """When skip_static=True, static joints are excluded."""
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        velocities = np.ones_like(four_bar_positions)
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, four_bar_positions, velocities, skip_static=True
        )
        assert quiver is not None

    def test_skip_static_false(self, four_bar, four_bar_positions):
        """When skip_static=False, all joints are drawn."""
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        velocities = np.ones_like(four_bar_positions)
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, four_bar_positions, velocities, skip_static=False
        )
        assert quiver is not None

    def test_nan_velocities_filtered(self, four_bar, four_bar_positions):
        """NaN velocities should be filtered out without error."""
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        velocities = np.full_like(four_bar_positions, np.nan)
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, four_bar_positions, velocities, skip_static=False
        )
        assert quiver is not None

    def test_empty_after_filtering(self, four_bar):
        """If all positions are filtered, return empty quiver."""
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        # Only static joints with skip_static=True should leave nothing
        # Build a linkage with only one static joint
        s = Static(0, 0, name="only")
        linkage = pl.Linkage(joints=[s], order=[], name="OnlyStatic")
        positions = np.array([[0.0, 0.0]])
        velocities = np.array([[1.0, 1.0]])
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            linkage, ax, positions, velocities, skip_static=True
        )
        assert quiver is not None

    def test_custom_style_kwargs(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        velocities = np.ones_like(four_bar_positions) * 0.5
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, four_bar_positions, velocities,
            scale=0.5, color="green", width=0.01, label="Custom"
        )
        assert quiver is not None

    def test_list_inputs(self, four_bar):
        """Accepts plain lists (not just ndarrays)."""
        from pylinkage.visualizer.kinematics import plot_velocity_vectors

        positions = [(0.0, 1.0), (3.0, 2.0)]
        velocities = [(0.1, 0.2), (0.3, 0.4)]
        fig, ax = plt.subplots()
        quiver = plot_velocity_vectors(
            four_bar, ax, positions, velocities, skip_static=False
        )
        assert quiver is not None


# --------------------------------------------------------------------------
# plot_acceleration_vectors
# --------------------------------------------------------------------------

class TestPlotAccelerationVectors:
    """Tests for plot_acceleration_vectors."""

    def test_basic_call(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        accels = np.random.default_rng(99).random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            four_bar, ax, four_bar_positions, accels
        )
        assert quiver is not None

    def test_skip_static_true(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        accels = np.ones_like(four_bar_positions)
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            four_bar, ax, four_bar_positions, accels, skip_static=True
        )
        assert quiver is not None

    def test_skip_static_false(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        accels = np.ones_like(four_bar_positions)
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            four_bar, ax, four_bar_positions, accels, skip_static=False
        )
        assert quiver is not None

    def test_nan_accelerations_filtered(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        accels = np.full_like(four_bar_positions, np.nan)
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            four_bar, ax, four_bar_positions, accels, skip_static=False
        )
        assert quiver is not None

    def test_empty_after_filtering(self, four_bar):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        s = Static(0, 0, name="only")
        linkage = pl.Linkage(joints=[s], order=[], name="OnlyStatic")
        positions = np.array([[0.0, 0.0]])
        accels = np.array([[1.0, 1.0]])
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            linkage, ax, positions, accels, skip_static=True
        )
        assert quiver is not None

    def test_custom_style_kwargs(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_acceleration_vectors

        accels = np.ones_like(four_bar_positions) * 2.0
        fig, ax = plt.subplots()
        quiver = plot_acceleration_vectors(
            four_bar, ax, four_bar_positions, accels,
            scale=0.05, color="purple", width=0.008, label="Accel"
        )
        assert quiver is not None


# --------------------------------------------------------------------------
# plot_kinematics_frame
# --------------------------------------------------------------------------

class TestPlotKinematicsFrame:
    """Tests for plot_kinematics_frame."""

    def test_positions_only(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            show_velocity=False, show_acceleration=False,
        )
        # No error is success

    def test_with_velocity(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        vels = np.random.default_rng(7).random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            velocities=vels, show_velocity=True, show_acceleration=False,
        )

    def test_with_acceleration(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        accels = np.random.default_rng(8).random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            accelerations=accels, show_velocity=False, show_acceleration=True,
        )

    def test_with_both_vectors(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        rng = np.random.default_rng(9)
        vels = rng.random(four_bar_positions.shape)
        accels = rng.random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            velocities=vels, accelerations=accels,
            show_velocity=True, show_acceleration=True,
        )

    def test_none_velocities_not_drawn(self, four_bar, four_bar_positions):
        """If velocities is None but show_velocity is True, nothing crashes."""
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            velocities=None, show_velocity=True,
        )

    def test_none_accelerations_not_drawn(self, four_bar, four_bar_positions):
        """If accelerations is None but show_acceleration is True, nothing crashes."""
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            accelerations=None, show_acceleration=True,
        )

    def test_custom_scales(self, four_bar, four_bar_positions):
        from pylinkage.visualizer.kinematics import plot_kinematics_frame

        rng = np.random.default_rng(10)
        vels = rng.random(four_bar_positions.shape)
        accels = rng.random(four_bar_positions.shape)
        fig, ax = plt.subplots()
        plot_kinematics_frame(
            four_bar, ax, four_bar_positions,
            velocities=vels, accelerations=accels,
            show_velocity=True, show_acceleration=True,
            velocity_scale=0.5, acceleration_scale=0.05,
        )
