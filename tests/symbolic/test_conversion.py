"""Tests for conversion between symbolic and numeric linkages."""

import math

import numpy as np
import pytest

from pylinkage import Crank, Linkage, Revolute, Static
from pylinkage.symbolic import (
    SymbolicLinkage,
    SymCrank,
    SymRevolute,
    SymStatic,
    compute_trajectory_numeric,
    fourbar_symbolic,
    get_numeric_parameters,
    linkage_to_symbolic,
    symbolic_to_linkage,
)


class TestLinkageToSymbolic:
    """Tests for linkage_to_symbolic conversion."""

    def test_static_joint_conversion(self):
        """Test converting static joint."""
        A = Static(3, 4, name="A")
        linkage = Linkage(joints=[A], order=[])

        sym_linkage = linkage_to_symbolic(linkage)

        assert len(sym_linkage.joints) == 1
        joint = sym_linkage.joints[0]
        assert isinstance(joint, SymStatic)
        assert joint.name == "A"

        x, y = joint.position_expr()
        assert float(x) == 3.0
        assert float(y) == 4.0

    def test_crank_joint_conversion(self):
        """Test converting crank joint."""
        A = Static(0, 0, name="A")
        B = Crank(1, 0, joint0=A, distance=1, angle=0.1, name="B")
        linkage = Linkage(joints=[A, B], order=[B])

        sym_linkage = linkage_to_symbolic(linkage)

        assert len(sym_linkage.joints) == 2

        # Check crank
        sym_crank = sym_linkage.get_joint("B")
        assert isinstance(sym_crank, SymCrank)
        assert sym_crank._numeric_r == 1.0

    def test_revolute_joint_conversion(self):
        """Test converting revolute joint."""
        A = Static(0, 0, name="A")
        D = Static(4, 0, name="D")
        B = Crank(1, 0, joint0=A, distance=1, angle=0.1, name="B")
        C = Revolute(2.5, 2, joint0=B, joint1=D, distance0=3, distance1=3, name="C")
        linkage = Linkage(joints=[A, D, B, C], order=[B, C])

        sym_linkage = linkage_to_symbolic(linkage)

        assert len(sym_linkage.joints) == 4

        # Check revolute
        sym_rev = sym_linkage.get_joint("C")
        assert isinstance(sym_rev, SymRevolute)
        assert sym_rev._numeric_r0 == 3.0
        assert sym_rev._numeric_r1 == 3.0

    def test_parameter_prefix(self):
        """Test parameter prefix option."""
        A = Static(0, 0, name="A")
        B = Crank(1, 0, joint0=A, distance=1, angle=0.1, name="B")
        linkage = Linkage(joints=[A, B], order=[B])

        sym_linkage = linkage_to_symbolic(linkage, param_prefix="my_")

        params = sym_linkage.parameters
        # Should have parameter starting with "my_"
        assert any(name.startswith("my_") for name in params)


class TestSymbolicToLinkage:
    """Tests for symbolic_to_linkage conversion."""

    def test_static_joint_conversion(self):
        """Test converting symbolic static joint."""
        sym_A = SymStatic(3, 4, name="A")
        sym_linkage = SymbolicLinkage([sym_A])

        linkage = symbolic_to_linkage(sym_linkage, {})

        assert len(linkage.joints) == 1
        joint = linkage.joints[0]
        assert isinstance(joint, Static)
        assert joint.x == pytest.approx(3.0)
        assert joint.y == pytest.approx(4.0)

    def test_crank_joint_conversion(self):
        """Test converting symbolic crank joint."""
        sym_A = SymStatic(0, 0, name="A")
        sym_B = SymCrank(sym_A, radius="r", name="B")
        sym_linkage = SymbolicLinkage([sym_A, sym_B])

        linkage = symbolic_to_linkage(sym_linkage, {"r": 2.0}, initial_theta=0)

        # B should be at (2, 0) at theta=0
        B = None
        for j in linkage.joints:
            if j.name == "B":
                B = j
                break
        assert B is not None
        assert B.x == pytest.approx(2.0)
        assert B.y == pytest.approx(0.0)

    def test_fourbar_conversion(self):
        """Test converting four-bar linkage."""
        sym_A = SymStatic(0, 0, name="A")
        sym_D = SymStatic(4, 0, name="D")
        sym_B = SymCrank(sym_A, radius="r_AB", name="B")
        sym_C = SymRevolute(sym_B, sym_D, distance0="r_BC", distance1="r_CD", branch=1, name="C")
        sym_linkage = SymbolicLinkage([sym_A, sym_D, sym_B, sym_C])

        params = {"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0}
        linkage = symbolic_to_linkage(sym_linkage, params)

        assert len(linkage.joints) == 4

        # Check that solve order includes crank and revolute
        assert len(linkage._solve_order) == 2


class TestGetNumericParameters:
    """Tests for get_numeric_parameters function."""

    def test_extracts_crank_parameter(self):
        """Test extracting crank parameter."""
        A = Static(0, 0, name="A")
        B = Crank(1, 0, joint0=A, distance=2.5, angle=0.1, name="B")
        linkage = Linkage(joints=[A, B], order=[B])

        sym_linkage = linkage_to_symbolic(linkage)
        params = get_numeric_parameters(sym_linkage)

        # Should have the radius parameter
        assert len(params) == 1
        assert list(params.values())[0] == 2.5

    def test_extracts_revolute_parameters(self):
        """Test extracting revolute parameters."""
        A = Static(0, 0, name="A")
        D = Static(4, 0, name="D")
        B = Crank(1, 0, joint0=A, distance=1, angle=0.1, name="B")
        C = Revolute(2.5, 2, joint0=B, joint1=D, distance0=3, distance1=2, name="C")
        linkage = Linkage(joints=[A, D, B, C], order=[B, C])

        sym_linkage = linkage_to_symbolic(linkage)
        params = get_numeric_parameters(sym_linkage)

        # Should have crank radius + 2 revolute distances
        assert len(params) == 3
        assert 1.0 in params.values()  # Crank radius
        assert 3.0 in params.values()  # r0
        assert 2.0 in params.values()  # r1


class TestFourbarSymbolic:
    """Tests for fourbar_symbolic convenience function."""

    def test_creates_four_joints(self):
        """Test that four-bar has four joints."""
        linkage = fourbar_symbolic(4, 1, 3, 3)

        assert len(linkage.joints) == 4

    def test_joint_types(self):
        """Test that joints have correct types."""
        linkage = fourbar_symbolic(4, 1, 3, 3)

        joint_types = [type(j).__name__ for j in linkage.joints]
        assert joint_types.count("SymStatic") == 2
        assert joint_types.count("SymCrank") == 1
        assert joint_types.count("SymRevolute") == 1

    def test_symbolic_parameters(self):
        """Test four-bar with symbolic parameters."""
        linkage = fourbar_symbolic(
            ground_length=4,
            crank_length="L1",
            coupler_length="L2",
            rocker_length="L3",
        )

        params = linkage.parameters
        assert "L1" in params
        assert "L2" in params
        assert "L3" in params

    def test_ground_position(self):
        """Test ground position with offset."""
        linkage = fourbar_symbolic(4, 1, 3, 3, ground_x=10, ground_y=5)

        A = linkage.get_joint("A")
        D = linkage.get_joint("D")

        x_a, y_a = A.position_expr()
        x_d, y_d = D.position_expr()

        assert float(x_a) == 10.0
        assert float(y_a) == 5.0
        assert float(x_d) == 14.0
        assert float(y_d) == 5.0


class TestRoundTrip:
    """Tests for round-trip conversion numeric -> symbolic -> numeric."""

    def test_crank_roundtrip(self):
        """Test round-trip for simple crank."""
        # Create numeric linkage
        A = Static(0, 0, name="A")
        B = Crank(2, 0, joint0=A, distance=2, angle=0.1, name="B")
        original = Linkage(joints=[A, B], order=[B])

        # Convert to symbolic and back
        sym_linkage = linkage_to_symbolic(original)
        params = get_numeric_parameters(sym_linkage)
        restored = symbolic_to_linkage(sym_linkage, params, initial_theta=0)

        # Check positions match
        assert restored.joints[0].x == pytest.approx(0.0)
        assert restored.joints[0].y == pytest.approx(0.0)
        assert restored.joints[1].x == pytest.approx(2.0)
        assert restored.joints[1].y == pytest.approx(0.0)

    def test_fourbar_trajectory_match(self):
        """Test that symbolic trajectory is geometrically valid."""
        # Create symbolic four-bar linkage
        sym_A = SymStatic(0, 0, name="A")
        sym_D = SymStatic(4, 0, name="D")
        sym_B = SymCrank(sym_A, radius=1, name="B")
        sym_C = SymRevolute(sym_B, sym_D, distance0=3, distance1=3, branch=1, name="C")
        sym_linkage = SymbolicLinkage([sym_A, sym_D, sym_B, sym_C])

        # Compute symbolic trajectory
        theta_vals = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        sym_trajectories = compute_trajectory_numeric(sym_linkage, {}, theta_vals)

        # Verify constraint satisfaction: C should always be at distance 3 from B and D
        for i in range(len(theta_vals)):
            c_x, c_y = sym_trajectories["C"][i]
            b_x, b_y = sym_trajectories["B"][i]
            d_x, d_y = sym_trajectories["D"][i]

            dist_bc = math.sqrt((c_x - b_x) ** 2 + (c_y - b_y) ** 2)
            dist_cd = math.sqrt((c_x - d_x) ** 2 + (c_y - d_y) ** 2)

            assert dist_bc == pytest.approx(3.0, abs=1e-6)
            assert dist_cd == pytest.approx(3.0, abs=1e-6)

        # Verify the crank constraint: B should always be at distance 1 from A
        for i in range(len(theta_vals)):
            b_x, b_y = sym_trajectories["B"][i]
            a_x, a_y = sym_trajectories["A"][i]

            dist_ab = math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)
            assert dist_ab == pytest.approx(1.0, abs=1e-6)
