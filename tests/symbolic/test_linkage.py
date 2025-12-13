"""Tests for SymbolicLinkage class."""

import pytest
import sympy as sp

from pylinkage.symbolic import (
    SymbolicLinkage,
    SymCrank,
    SymRevolute,
    SymStatic,
    theta,
)


class TestSymbolicLinkage:
    """Tests for SymbolicLinkage class."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        return SymbolicLinkage([A, D, B, C], name="fourbar")

    def test_joints_tuple(self, fourbar_linkage):
        """Test that joints are stored as tuple."""
        assert isinstance(fourbar_linkage.joints, tuple)
        assert len(fourbar_linkage.joints) == 4

    def test_parameters_collection(self, fourbar_linkage):
        """Test that parameters are collected from joints."""
        params = fourbar_linkage.parameters

        # Should have 3 parameters: r_AB, r_BC, r_CD
        assert len(params) == 3
        assert "r_AB" in params
        assert "r_BC" in params
        assert "r_CD" in params

    def test_get_joint(self, fourbar_linkage):
        """Test getting joint by name."""
        joint = fourbar_linkage.get_joint("B")
        assert joint.name == "B"
        assert isinstance(joint, SymCrank)

    def test_get_joint_not_found(self, fourbar_linkage):
        """Test error when joint not found."""
        with pytest.raises(ValueError, match="not found"):
            fourbar_linkage.get_joint("nonexistent")

    def test_get_trajectory_expressions(self, fourbar_linkage):
        """Test getting trajectory expressions."""
        trajectories = fourbar_linkage.get_trajectory_expressions()

        assert "A" in trajectories
        assert "B" in trajectories
        assert "C" in trajectories
        assert "D" in trajectories

        # Each should be a tuple of (x_expr, y_expr)
        for name, (x, y) in trajectories.items():
            assert isinstance(x, sp.Basic)
            assert isinstance(y, sp.Basic)

    def test_coupler_curve(self, fourbar_linkage):
        """Test getting coupler curve for specific joint."""
        x, y = fourbar_linkage.coupler_curve("C")

        assert isinstance(x, sp.Basic)
        assert isinstance(y, sp.Basic)

        # Should contain theta
        assert theta in x.free_symbols or theta in y.free_symbols

    def test_get_constraint_equations(self, fourbar_linkage):
        """Test getting all constraint equations."""
        equations = fourbar_linkage.get_constraint_equations()

        # Crank has 1 constraint, Revolute has 2 constraints
        # Static joints have 0 constraints
        assert len(equations) == 3

    def test_jacobian(self, fourbar_linkage):
        """Test Jacobian computation."""
        jac = fourbar_linkage.jacobian()

        # Should be Matrix with shape (2*4, 3) = (8, 3)
        # 4 joints * 2 coords, 3 parameters
        assert jac.shape == (8, 3)

    def test_jacobian_specific_joints(self, fourbar_linkage):
        """Test Jacobian for specific joints."""
        jac = fourbar_linkage.jacobian(joint_names=["C"])

        # Only joint C: (2, 3)
        assert jac.shape == (2, 3)

    def test_jacobian_theta(self, fourbar_linkage):
        """Test Jacobian with respect to theta."""
        jac = fourbar_linkage.jacobian_theta()

        # Should be Matrix with shape (8, 1)
        assert jac.shape == (8, 1)

    def test_substitute(self, fourbar_linkage):
        """Test parameter substitution."""
        new_linkage = fourbar_linkage.substitute({
            "r_AB": 1.0,
            "r_BC": 3.0,
        })

        # New linkage should have fewer symbolic parameters
        assert "r_AB" not in new_linkage.parameters
        assert "r_BC" not in new_linkage.parameters
        assert "r_CD" in new_linkage.parameters

    def test_repr(self, fourbar_linkage):
        """Test string representation."""
        s = repr(fourbar_linkage)
        assert "SymbolicLinkage" in s
        assert "fourbar" in s


class TestFourbarNumeric:
    """Test four-bar linkage with numeric parameters."""

    def test_evaluate_trajectory(self):
        """Test evaluating trajectory at specific theta."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, branch=1, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        x, y = linkage.coupler_curve("C")

        # Evaluate at theta=0
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))

        # Verify C is at correct distance from B and D
        b_x, b_y = 1, 0  # B at theta=0
        d_x, d_y = 4, 0

        dist_b = ((x_val - b_x) ** 2 + (y_val - b_y) ** 2) ** 0.5
        dist_d = ((x_val - d_x) ** 2 + (y_val - d_y) ** 2) ** 0.5

        assert dist_b == pytest.approx(3.0, abs=1e-10)
        assert dist_d == pytest.approx(3.0, abs=1e-10)
