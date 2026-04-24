"""Tests for pylinkage.population._member."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.optimization.collections.agent import Agent
from pylinkage.optimization.collections.pareto import ParetoSolution
from pylinkage.population import Member


def test_member_default_score_nan_when_empty():
    m = Member(
        dimensions=np.array([1.0, 2.0]),
        initial_positions=np.zeros((2, 2)),
    )
    assert math.isnan(m.score)
    assert m.scores == {}
    assert m.trajectory is None
    assert m.metadata == {}


def test_member_score_primary_is_first_entry():
    m = Member(
        dimensions=np.array([1.0]),
        initial_positions=np.zeros((1, 2)),
        scores={"alpha": 5.0, "beta": 10.0},
    )
    assert m.score == 5.0


def test_member_metadata_accepts_arbitrary_data():
    meta = {"topology": "fourbar", "index": 3}
    m = Member(
        dimensions=np.array([1.0]),
        initial_positions=np.zeros((1, 2)),
        metadata=meta,
    )
    assert m.metadata["topology"] == "fourbar"
    assert m.metadata["index"] == 3


def test_from_agent_basic():
    agent = Agent(
        score=4.2,
        dimensions=[1.0, 2.0, 3.0],
        initial_positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)],
    )
    m = Member.from_agent(agent, n_joints=3)
    assert m.dimensions.dtype == np.float64
    np.testing.assert_allclose(m.dimensions, [1.0, 2.0, 3.0])
    assert m.initial_positions.shape == (3, 2)
    np.testing.assert_allclose(m.initial_positions[2], [2.0, 1.0])
    assert m.scores == {"score": 4.2}
    assert m.score == 4.2


def test_from_agent_handles_none_positions():
    agent = Agent(
        score=0.0,
        dimensions=[1.0],
        initial_positions=[(None, None), (1.0, None), (None, 2.0)],
    )
    m = Member.from_agent(agent, n_joints=3)
    np.testing.assert_allclose(m.initial_positions[0], [0.0, 0.0])
    np.testing.assert_allclose(m.initial_positions[1], [1.0, 0.0])
    np.testing.assert_allclose(m.initial_positions[2], [0.0, 2.0])


def test_from_agent_empty_positions_fills_with_zeros():
    agent = Agent(
        score=1.0,
        dimensions=[1.0, 2.0],
        initial_positions=[],
    )
    m = Member.from_agent(agent, n_joints=4)
    assert m.initial_positions.shape == (4, 2)
    np.testing.assert_allclose(m.initial_positions, np.zeros((4, 2)))


def test_from_pareto_solution_named_objectives():
    sol = ParetoSolution(
        scores=(1.5, 2.5),
        dimensions=np.array([0.5, 1.0]),
        initial_positions=[(0.0, 0.0), (1.0, 1.0)],
    )
    m = Member.from_pareto_solution(
        sol,
        n_joints=2,
        objective_names=("acc", "smooth"),
    )
    assert m.scores == {"acc": 1.5, "smooth": 2.5}
    np.testing.assert_allclose(m.dimensions, [0.5, 1.0])
    np.testing.assert_allclose(m.initial_positions, [[0.0, 0.0], [1.0, 1.0]])


def test_from_pareto_solution_autoname_when_mismatch():
    sol = ParetoSolution(
        scores=(1.0, 2.0, 3.0),
        dimensions=np.array([1.0]),
        initial_positions=[(0.0, 0.0)],
    )
    m = Member.from_pareto_solution(sol, n_joints=1, objective_names=("only_one",))
    assert m.scores == {"obj_0": 1.0, "obj_1": 2.0, "obj_2": 3.0}


def test_from_pareto_solution_handles_none_positions_and_empty():
    sol = ParetoSolution(
        scores=(0.0,),
        dimensions=np.array([1.0]),
        initial_positions=[(None, None), (2.0, None)],
    )
    m = Member.from_pareto_solution(sol, n_joints=2)
    np.testing.assert_allclose(m.initial_positions[0], [0.0, 0.0])
    np.testing.assert_allclose(m.initial_positions[1], [2.0, 0.0])

    sol_empty = ParetoSolution(
        scores=(1.0,),
        dimensions=np.array([1.0]),
        initial_positions=[],
    )
    m_empty = Member.from_pareto_solution(sol_empty, n_joints=3)
    assert m_empty.initial_positions.shape == (3, 2)
    np.testing.assert_allclose(m_empty.initial_positions, np.zeros((3, 2)))


def test_to_loci_raises_when_no_trajectory():
    m = Member(
        dimensions=np.array([1.0]),
        initial_positions=np.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="No trajectory"):
        m.to_loci()


def test_to_loci_returns_nested_tuples():
    traj = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
    m = Member(
        dimensions=np.array([1.0]),
        initial_positions=np.zeros((3, 2)),
        trajectory=traj,
    )
    loci = m.to_loci()
    assert isinstance(loci, tuple)
    assert len(loci) == 2
    assert len(loci[0]) == 3
    assert isinstance(loci[0][0], tuple)
    assert loci[0][0] == (0.0, 1.0)
    assert loci[1][2] == (10.0, 11.0)
    for frame in loci:
        for pt in frame:
            assert isinstance(pt[0], float)
            assert isinstance(pt[1], float)


def test_to_agent_roundtrip():
    agent = Agent(
        score=7.0,
        dimensions=np.array([1.0, 2.0]),
        initial_positions=[(0.5, 1.5), (2.0, 2.5)],
    )
    m = Member.from_agent(agent, n_joints=2)
    back = m.to_agent()
    assert back.score == 7.0
    np.testing.assert_allclose(back.dimensions, [1.0, 2.0])
    assert back.initial_positions == [(0.5, 1.5), (2.0, 2.5)]


def test_to_agent_with_empty_scores_gives_nan_score():
    m = Member(
        dimensions=np.array([1.0]),
        initial_positions=np.array([[0.0, 0.0]]),
    )
    agent = m.to_agent()
    assert math.isnan(agent.score)
    assert agent.initial_positions == [(0.0, 0.0)]
