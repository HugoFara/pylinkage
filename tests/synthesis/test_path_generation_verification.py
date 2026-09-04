"""The two verification back-ends must accept exactly the same solutions.

``_verify_coupler_path`` simulates each candidate, which is the dominant cost of
:func:`path_generation`. It runs through the numba solver where it can, and
through ``Linkage.step()`` otherwise. The two report an unassemblable mechanism
differently -- ``step()`` raises, the solver writes ``NaN`` -- so a naive switch
silently changes which candidates survive.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pylinkage.synthesis import path_generation

# ``pylinkage.synthesis.path_generation`` is the re-exported *function*, so the
# module has to come from the import system rather than attribute lookup.
pg_module = importlib.import_module("pylinkage.synthesis.path_generation")

# Point sets chosen to exercise different outcomes: plenty of solutions, few,
# and a three-point problem that finishes early.
POINT_SETS = {
    "readme": [(0, 0), (1, 1), (2, 1), (3, 0)],
    "documented": [(0.0, 1.0), (1.0, 2.0), (2.0, 1.5), (3.0, 0.0)],
    "three_points": [(0, 0), (1.5, 1.2), (3, 0)],
    "steep": [(0, 0), (0.5, 2.0), (1.0, 0.5), (2.0, 2.0)],
}


def solution_signature(result):
    """Link lengths of every raw solution, order-independent."""
    return sorted(
        (
            round(s.crank_length, 6),
            round(s.coupler_length, 6),
            round(s.rocker_length, 6),
            round(s.ground_length, 6),
        )
        for s in result.raw_solutions
    )


@pytest.mark.parametrize("points", POINT_SETS.values(), ids=list(POINT_SETS))
def test_numba_and_python_paths_agree(points, monkeypatch):
    """Same solutions whether or not the numba solver is used."""
    monkeypatch.setattr(pg_module, "HAS_NUMBA", False)
    without_numba = path_generation(points)

    monkeypatch.setattr(pg_module, "HAS_NUMBA", True)
    with_numba = path_generation(points)

    assert solution_signature(with_numba) == solution_signature(without_numba)


@pytest.mark.parametrize("points", POINT_SETS.values(), ids=list(POINT_SETS))
def test_solutions_pass_through_their_precision_points(points):
    """Whatever survives verification must actually trace the path."""
    result = path_generation(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diag = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
    tolerance = max(0.05 * diag, 0.01)

    for solution in result.raw_solutions:
        assert pg_module._verify_coupler_path(solution, points, tolerance=tolerance)


class TestCouplerTrajectory:
    """``_coupler_trajectory`` turns a simulation failure into None."""

    def test_returns_one_row_per_iteration(self):
        result = path_generation(POINT_SETS["readme"])
        linkage = result.solutions[0]
        trajectory = pg_module._coupler_trajectory(linkage, 50)
        assert trajectory is not None
        assert trajectory.shape == (50, 2)
        assert np.isfinite(trajectory).all()

    def test_nan_trajectory_is_rejected(self, monkeypatch):
        """A NaN is how the solver reports what step() reports by raising.

        Both mean the mechanism did not assemble at some crank angle, so both
        have to reject the candidate.
        """
        result = path_generation(POINT_SETS["readme"])
        linkage = result.solutions[0]

        def nan_run(*args, **kwargs):
            return np.full((10, len(linkage.components), 2), np.nan)

        monkeypatch.setattr(pg_module, "HAS_NUMBA", True)
        monkeypatch.setattr(type(linkage), "step_fast", nan_run)
        assert pg_module._coupler_trajectory(linkage, 10) is None

    def test_unsupported_component_falls_back_to_step(self, monkeypatch):
        """A mechanism the solver cannot represent still gets verified."""
        result = path_generation(POINT_SETS["readme"])
        linkage = result.solutions[0]

        def unsupported(*args, **kwargs):
            raise NotImplementedError("PPDyad")

        monkeypatch.setattr(pg_module, "HAS_NUMBA", True)
        monkeypatch.setattr(type(linkage), "step_fast", unsupported)
        trajectory = pg_module._coupler_trajectory(linkage, 20)
        assert trajectory is not None
        assert trajectory.shape == (20, 2)
