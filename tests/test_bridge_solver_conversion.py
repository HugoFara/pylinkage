"""Tests for bridge/solver_conversion.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.bridge import (
    linkage_to_solver_data,
    solver_data_to_linkage,
    update_solver_constraints,
    update_solver_positions,
)
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.solver.types import (
    JOINT_CRANK,
    JOINT_REVOLUTE,
    JOINT_STATIC,
    SolverData,
)


def make_fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


def make_fourbar_with_fixed():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    fixed = FixedDyad(
        anchor1=crank.output,
        anchor2=rocker,
        distance=0.5,
        angle=math.pi / 4,
        name="fixed",
    )
    return Linkage([O1, O2, crank, rocker, fixed], name="FourBarFixed")


class TestLinkageToSolverData:
    def test_basic_structure(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        assert isinstance(data, SolverData)
        assert data.positions.shape == (4, 2)
        assert data.joint_types.shape == (4,)

    def test_joint_types(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # First two are ground (STATIC), third is crank, fourth is revolute
        assert data.joint_types[0] == JOINT_STATIC
        assert data.joint_types[1] == JOINT_STATIC
        assert data.joint_types[2] == JOINT_CRANK
        assert data.joint_types[3] == JOINT_REVOLUTE

    def test_constraints(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # Constraints are: crank [radius=1.0, omega=0.1], rocker [d1=2.5, d2=2.0]
        assert len(data.constraints) == 4
        assert data.constraints[0] == pytest.approx(1.0)
        assert data.constraints[1] == pytest.approx(0.1)
        assert data.constraints[2] == pytest.approx(2.5)
        assert data.constraints[3] == pytest.approx(2.0)

    def test_positions(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        assert data.positions[0, 0] == pytest.approx(0.0)
        assert data.positions[0, 1] == pytest.approx(0.0)
        assert data.positions[1, 0] == pytest.approx(3.0)

    def test_solve_order(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # Solve order: crank and rocker (indices 2 and 3)
        assert len(data.solve_order) >= 2
        assert 2 in data.solve_order
        assert 3 in data.solve_order

    def test_with_fixed_dyad(self):
        linkage = make_fourbar_with_fixed()
        data = linkage_to_solver_data(linkage)
        assert data.positions.shape == (5, 2)


class TestSolverDataToLinkage:
    def test_positions_updated(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # Set new positions in data
        data.positions[2, 0] = 7.0
        data.positions[2, 1] = 8.0
        solver_data_to_linkage(data, linkage)
        crank = linkage.components[2]
        assert crank.x == pytest.approx(7.0)
        assert crank.y == pytest.approx(8.0)

    def test_with_velocities(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # Set velocities on the data
        data.velocities = np.zeros((4, 2), dtype=np.float64)
        data.velocities[2, 0] = 1.5
        data.velocities[2, 1] = 2.5
        solver_data_to_linkage(data, linkage)
        crank = linkage.components[2]
        assert crank.velocity == pytest.approx((1.5, 2.5))

    def test_with_accelerations(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        data.accelerations = np.zeros((4, 2), dtype=np.float64)
        data.accelerations[2, 0] = 3.0
        data.accelerations[2, 1] = 4.0
        solver_data_to_linkage(data, linkage)
        crank = linkage.components[2]
        assert crank.acceleration == pytest.approx((3.0, 4.0))


class TestUpdateSolverConstraints:
    def test_update_crank(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        crank = linkage.components[2]
        crank.radius = 2.5
        crank.angular_velocity = 0.5
        update_solver_constraints(data, linkage)
        assert data.constraints[0] == pytest.approx(2.5)
        assert data.constraints[1] == pytest.approx(0.5)

    def test_update_rrr_dyad(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        rocker = linkage.components[3]
        rocker.distance1 = 3.0
        rocker.distance2 = 3.5
        update_solver_constraints(data, linkage)
        assert data.constraints[2] == pytest.approx(3.0)
        assert data.constraints[3] == pytest.approx(3.5)

    def test_update_fixed_dyad(self):
        linkage = make_fourbar_with_fixed()
        data = linkage_to_solver_data(linkage)
        fixed = linkage.components[4]
        fixed.distance = 0.75
        fixed.angle = 1.0
        update_solver_constraints(data, linkage)
        # Fixed dyad has its own constraint offset
        offset = data.constraint_offsets[4]
        assert data.constraints[offset] == pytest.approx(0.75)
        assert data.constraints[offset + 1] == pytest.approx(1.0)


class TestUpdateSolverPositions:
    def test_basic(self):
        linkage = make_fourbar()
        data = linkage_to_solver_data(linkage)
        # Modify linkage positions
        linkage.components[2].x = 10.0
        linkage.components[2].y = 20.0
        update_solver_positions(data, linkage)
        assert data.positions[2, 0] == pytest.approx(10.0)
        assert data.positions[2, 1] == pytest.approx(20.0)
