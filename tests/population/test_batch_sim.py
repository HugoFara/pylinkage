"""Tests for pylinkage.population._batch_sim."""

from __future__ import annotations

import numpy as np

from pylinkage.bridge.solver_conversion import linkage_to_solver_data
from pylinkage.population._batch_sim import (
    _build_user_to_solver_map,
    simulate_batch,
)


def _get_positions(linkage):
    return np.array(
        [
            [x if x is not None else 0.0, y if y is not None else 0.0]
            for (x, y) in linkage.get_coords()
        ],
        dtype=np.float64,
    )


def test_build_user_to_solver_map_skips_crank_angle_rate(fourbar):
    template = linkage_to_solver_data(fourbar)
    indices, base = _build_user_to_solver_map(template)
    # The crank joint exposes only the radius (one user constraint),
    # not the angle_rate. The RRR dyad exposes both distances.
    assert indices.dtype == np.intp
    assert len(indices) == 3
    # Template constraints must be copy (not shared)
    assert base is not template.constraints
    np.testing.assert_allclose(base, template.constraints)
    # Writing to the copy should not modify the template
    base[0] = 99.0
    assert template.constraints[0] != 99.0


def test_simulate_batch_need_expand_path(fourbar):
    template = linkage_to_solver_data(fourbar)
    positions = _get_positions(fourbar)

    # 3 user constraints per member
    all_dims = np.array(
        [[1.0, 2.5, 2.0], [0.9, 2.5, 2.0]],
        dtype=np.float64,
    )
    all_positions = np.stack([positions, positions])

    traj = simulate_batch(template, all_dims, all_positions, iterations=12, dt=1.0)
    assert traj.shape == (2, 12, template.n_joints, 2)
    assert not np.isnan(traj).any()

    # The initial positions array must not be mutated by the in-place solver
    np.testing.assert_allclose(all_positions[0], positions)


def test_simulate_batch_no_expand_path_when_matches_solver(fourbar):
    template = linkage_to_solver_data(fourbar)
    positions = _get_positions(fourbar)

    # Full solver constraints: [radius, angle_rate, d1, d2]
    solver_dims = template.constraints.copy()
    all_dims = np.stack([solver_dims, solver_dims])
    all_positions = np.stack([positions, positions])

    traj = simulate_batch(template, all_dims, all_positions, iterations=8, dt=1.0)
    assert traj.shape == (2, 8, template.n_joints, 2)
    assert not np.isnan(traj).any()


def test_simulate_batch_unbuildable_produces_nan(fourbar):
    template = linkage_to_solver_data(fourbar)
    positions = _get_positions(fourbar)

    # Make the second member's rocker too short to close the loop
    all_dims = np.array(
        [[1.0, 2.5, 2.0], [1.0, 0.01, 0.01]],
        dtype=np.float64,
    )
    all_positions = np.stack([positions, positions])

    traj = simulate_batch(template, all_dims, all_positions, iterations=6, dt=1.0)
    assert traj.shape == (2, 6, template.n_joints, 2)
    assert not np.isnan(traj[0]).any()
    # The unbuildable member should have produced at least some NaNs
    assert np.isnan(traj[1]).any()


def test_simulate_batch_single_member(fourbar):
    template = linkage_to_solver_data(fourbar)
    positions = _get_positions(fourbar)
    dims = np.array([[1.0, 2.5, 2.0]], dtype=np.float64)
    positions_batch = positions[None, ...]

    traj = simulate_batch(template, dims, positions_batch, iterations=4, dt=1.0)
    assert traj.shape == (1, 4, template.n_joints, 2)
    assert not np.isnan(traj).any()
