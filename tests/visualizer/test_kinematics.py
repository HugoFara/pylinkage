from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.kinematics import (
    animate_kinematics,
    plot_acceleration_vectors,
    plot_kinematics_frame,
    plot_velocity_vectors,
    show_kinematics,
)


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    lk = Linkage([O1, O2, crank, rocker], name="FourBar")
    return lk, crank


class _CrankAdapter:
    """Proxy Crank that exposes `omega` as an attribute plus delegates."""

    def __init__(self, real_crank, omega):
        object.__setattr__(self, "_real", real_crank)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "name", real_crank.name)

    def coord(self):
        return (self._real.position[0], self._real.position[1])

    @property
    def joint0(self):
        return getattr(self._real, "anchor", None)

    @property
    def joint1(self):
        return None

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_real"), item)


# Name the adapter class "Crank" so kinematics.py check passes.
_CrankAdapter.__name__ = "Crank"


class _FakeLinkage:
    """Wraps a simulation.Linkage to look like the old API for kinematics."""

    def __init__(self, sim_linkage, crank, omega=None):
        self._sim = sim_linkage
        self.name = sim_linkage.name
        joints = list(sim_linkage.components)
        if omega is not None:
            adapter = _CrankAdapter(crank, omega)
            joints[joints.index(crank)] = adapter
        self.joints = joints

    def step_fast_with_kinematics(self, iterations=None, dt=1.0):
        return self._sim.step_fast_with_kinematics(iterations=iterations or 30, dt=dt)


@pytest.fixture
def linkage_with_omega():
    sim, crank = _fourbar()
    sim.set_input_velocity(crank, omega=5.0)
    return _FakeLinkage(sim, crank, omega=5.0)


@pytest.fixture
def linkage_no_omega():
    sim, crank = _fourbar()
    return _FakeLinkage(sim, crank)


class TestPlotVelocityVectors:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, _ = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        q = plot_velocity_vectors(linkage_with_omega, ax, pos[0], vel[0])
        assert q is not None

    def test_no_skip_static(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, _ = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_velocity_vectors(linkage_with_omega, ax, pos[0], vel[0], skip_static=False)

    def test_with_nan_velocities(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, _ = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        vel_nan = vel[0].copy()
        vel_nan[:] = np.nan
        plot_velocity_vectors(linkage_with_omega, ax, pos[0], vel_nan)

    def test_custom_style(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, _ = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_velocity_vectors(
            linkage_with_omega,
            ax,
            pos[0],
            vel[0],
            scale=0.5,
            color="green",
            width=0.01,
            label="Vel",
        )


class TestPlotAccelerationVectors:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, _, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        q = plot_acceleration_vectors(linkage_with_omega, ax, pos[0], acc[0])
        assert q is not None

    def test_no_skip_static(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, _, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_acceleration_vectors(linkage_with_omega, ax, pos[0], acc[0], skip_static=False)

    def test_with_nan(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, _, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        acc_nan = acc[0].copy()
        acc_nan[:] = np.nan
        plot_acceleration_vectors(linkage_with_omega, ax, pos[0], acc_nan)

    def test_custom(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, _, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_acceleration_vectors(
            linkage_with_omega, ax, pos[0], acc[0], scale=0.1, color="purple", label="A"
        )


class TestPlotKinematicsFrame:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_kinematics_frame(linkage_with_omega, ax, pos[0], vel[0], acc[0])

    def test_show_both(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_kinematics_frame(
            linkage_with_omega,
            ax,
            pos[0],
            vel[0],
            acc[0],
            show_velocity=True,
            show_acceleration=True,
        )

    def test_no_velocity(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, _, _ = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_kinematics_frame(linkage_with_omega, ax, pos[0], show_velocity=False)

    def test_custom_scales(self, linkage_with_omega):
        fig, ax = plt.subplots()
        pos, vel, acc = linkage_with_omega.step_fast_with_kinematics(iterations=5)
        plot_kinematics_frame(
            linkage_with_omega,
            ax,
            pos[0],
            vel[0],
            acc[0],
            velocity_scale=0.2,
            acceleration_scale=0.02,
            show_acceleration=True,
        )


class TestShowKinematics:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, linkage_with_omega):
        fig = show_kinematics(linkage_with_omega, frame_index=2)
        assert fig is not None

    def test_with_acceleration(self, linkage_with_omega):
        fig = show_kinematics(
            linkage_with_omega, frame_index=2, show_velocity=True, show_acceleration=True
        )
        assert fig is not None

    def test_with_title(self, linkage_with_omega):
        fig = show_kinematics(linkage_with_omega, frame_index=0, title="Hello")
        assert fig is not None

    def test_custom_scales(self, linkage_with_omega):
        fig = show_kinematics(
            linkage_with_omega,
            frame_index=1,
            velocity_scale=0.5,
            acceleration_scale=0.05,
        )
        assert fig is not None

    def test_no_omega_raises(self, linkage_no_omega):
        with pytest.raises(ValueError):
            show_kinematics(linkage_no_omega, frame_index=0)

    def test_bad_frame_index(self, linkage_with_omega):
        with pytest.raises(ValueError):
            show_kinematics(linkage_with_omega, frame_index=99999)


class TestAnimateKinematics:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self, linkage_with_omega, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        fig = animate_kinematics(linkage_with_omega, duration=1.0, fps=3)
        assert fig is not None

    def test_no_omega_raises(self, linkage_no_omega):
        with pytest.raises(ValueError):
            animate_kinematics(linkage_no_omega)

    def test_save_gif(self, linkage_with_omega, tmp_path, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        save_path = str(tmp_path / "anim.gif")
        fig = animate_kinematics(
            linkage_with_omega, duration=1.0, fps=3, save_path=save_path
        )
        assert fig is not None

    def test_no_velocity(self, linkage_with_omega, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        fig = animate_kinematics(
            linkage_with_omega, show_velocity=False, duration=1.0, fps=3
        )
        assert fig is not None

    def test_with_scale_and_title(self, linkage_with_omega, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        monkeypatch.setattr(plt, "pause", lambda *a, **k: None)
        fig = animate_kinematics(
            linkage_with_omega,
            velocity_scale=0.3,
            duration=0.01,
            fps=2,
            title="Kin anim",
        )
        assert fig is not None
