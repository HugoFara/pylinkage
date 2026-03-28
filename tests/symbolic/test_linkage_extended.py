"""Extended tests for SymbolicLinkage - targeting uncovered lines."""

from __future__ import annotations

import pytest
import sympy as sp

from pylinkage.symbolic import (
    SymbolicLinkage,
    SymCrank,
    SymRevolute,
    SymStatic,
    theta,
)


class TestSymbolicLinkageSimplify:
    """Tests for the simplify() method."""

    def test_simplify_static(self) -> None:
        """Test simplify with only static joints."""
        A = SymStatic(0, 0, name="A")
        B = SymStatic(4, 0, name="B")
        linkage = SymbolicLinkage([A, B], name="simple")
        simplified = linkage.simplify()
        assert len(simplified.joints) == 2
        assert simplified.name == "simple"

    def test_simplify_fourbar(self) -> None:
        """Test simplify on a full four-bar linkage."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C], name="fourbar")

        simplified = linkage.simplify()
        assert len(simplified.joints) == 4
        assert simplified.theta == linkage.theta

        # Should produce valid trajectory expressions
        trajectories = simplified.get_trajectory_expressions()
        assert "C" in trajectories

    def test_simplify_preserves_joint_names(self) -> None:
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        simplified = linkage.simplify()
        names = [j.name for j in simplified.joints]
        assert names == ["A", "D", "B", "C"]

    def test_simplify_with_numeric_params(self) -> None:
        """Test simplify when all parameters are numeric."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, branch=1, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        simplified = linkage.simplify()
        x, y = simplified.coupler_curve("C")
        # Evaluate at theta=0 to check it works
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))
        assert isinstance(x_val, float)
        assert isinstance(y_val, float)


class TestSymbolicLinkageSubstitute:
    """Tests for the substitute() method."""

    def test_substitute_all_params(self) -> None:
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C], name="fourbar")

        new_linkage = linkage.substitute({"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0})
        assert len(new_linkage.parameters) == 0

    def test_substitute_preserves_structure(self) -> None:
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        new_linkage = linkage.substitute({"r_AB": 1.0})
        assert len(new_linkage.joints) == 4
        joint_names = [j.name for j in new_linkage.joints]
        assert joint_names == ["A", "D", "B", "C"]

    def test_substitute_nonexistent_param(self) -> None:
        """Substituting a param that doesn't exist should be a no-op."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        new_linkage = linkage.substitute({"nonexistent": 42.0})
        assert "r" in new_linkage.parameters

    def test_substitute_with_symbolic_value(self) -> None:
        """Test substituting a symbol with another expression."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        k = sp.Symbol("k")
        new_linkage = linkage.substitute({"r_AB": 2 * k})
        # r_AB should be gone, replaced by expression with k
        assert "r_AB" not in new_linkage.parameters


class TestSymbolicLinkageCrankFromCoord:
    """Test SymCrank created from coordinate tuple (not SymJoint parent)."""

    def test_crank_from_coord(self) -> None:
        B = SymCrank((1, 2), radius=1, name="B")
        linkage = SymbolicLinkage([B])
        x, y = linkage.coupler_curve("B")
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))
        assert x_val == pytest.approx(2.0)
        assert y_val == pytest.approx(2.0)

    def test_simplify_crank_from_coord(self) -> None:
        """Test simplify when crank has coordinate parent."""
        B = SymCrank((1, 2), radius=1, name="B")
        linkage = SymbolicLinkage([B])
        simplified = linkage.simplify()
        x, y = simplified.coupler_curve("B")
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))
        assert x_val == pytest.approx(2.0)
        assert y_val == pytest.approx(2.0)

    def test_substitute_crank_from_coord(self) -> None:
        """Test substitute when crank has coordinate parent."""
        B = SymCrank((1, 2), radius="r", name="B")
        linkage = SymbolicLinkage([B])
        new_linkage = linkage.substitute({"r": 3.0})
        x, y = new_linkage.coupler_curve("B")
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))
        assert x_val == pytest.approx(4.0)
        assert y_val == pytest.approx(2.0)

    def test_crank_no_parent(self) -> None:
        """Test crank with no parent (defaults to origin)."""
        B = SymCrank((0, 0), radius=1, name="B")
        linkage = SymbolicLinkage([B])
        x, y = linkage.coupler_curve("B")
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))
        assert x_val == pytest.approx(1.0)
        assert y_val == pytest.approx(0.0)


class TestSymbolicLinkageJacobianExtended:
    """Extended Jacobian tests."""

    def test_jacobian_specific_multiple_joints(self) -> None:
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        jac = linkage.jacobian(joint_names=["B", "C"])
        # 2 joints * 2 coords = 4 rows, 3 params
        assert jac.shape == (4, 3)

    def test_jacobian_theta_specific_joints(self) -> None:
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        jac = linkage.jacobian_theta(joint_names=["C"])
        assert jac.shape == (2, 1)

    def test_jacobian_theta_all_joints(self) -> None:
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])

        jac = linkage.jacobian_theta()
        # 2 joints * 2 coords = 4 rows, 1 column (theta)
        assert jac.shape == (4, 1)


class TestSymbolicLinkageConstraints:
    """Tests for constraint equation extraction."""

    def test_constraints_static_only(self) -> None:
        """Static joints should have no constraints."""
        A = SymStatic(0, 0, name="A")
        B = SymStatic(4, 0, name="B")
        linkage = SymbolicLinkage([A, B])
        eqs = linkage.get_constraint_equations()
        assert len(eqs) == 0

    def test_constraints_crank(self) -> None:
        """Crank should have 1 distance constraint."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])
        eqs = linkage.get_constraint_equations()
        assert len(eqs) == 1

    def test_constraints_revolute(self) -> None:
        """Revolute should have 2 distance constraints."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, name="C")
        linkage = SymbolicLinkage([A, D, B, C])
        eqs = linkage.get_constraint_equations()
        # 1 from crank + 2 from revolute
        assert len(eqs) == 3


class TestSymbolicLinkageMisc:
    """Miscellaneous tests for full coverage."""

    def test_custom_theta(self) -> None:
        phi = sp.Symbol("phi")
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, theta=phi, name="B")
        linkage = SymbolicLinkage([A, B], theta=phi)
        assert linkage.theta == phi

    def test_default_name(self) -> None:
        A = SymStatic(0, 0, name="A")
        linkage = SymbolicLinkage([A])
        # Default name is str(id(self))
        assert linkage.name is not None
        assert len(linkage.name) > 0

    def test_repr(self) -> None:
        A = SymStatic(0, 0, name="A")
        linkage = SymbolicLinkage([A], name="test")
        r = repr(linkage)
        assert "SymbolicLinkage" in r
        assert "test" in r
        assert "A" in r

    def test_parameters_caching(self) -> None:
        """Test that parameters are cached (lazy init)."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        # First call initializes
        p1 = linkage.parameters
        # Second call returns cached
        p2 = linkage.parameters
        assert p1 is p2

    def test_get_joint_first(self) -> None:
        """Test getting the first joint."""
        A = SymStatic(0, 0, name="A")
        B = SymStatic(1, 0, name="B")
        linkage = SymbolicLinkage([A, B])
        assert linkage.get_joint("A").name == "A"

    def test_get_joint_last(self) -> None:
        """Test getting the last joint."""
        A = SymStatic(0, 0, name="A")
        B = SymStatic(1, 0, name="B")
        linkage = SymbolicLinkage([A, B])
        assert linkage.get_joint("B").name == "B"

    def test_coupler_curve_raises_for_missing(self) -> None:
        A = SymStatic(0, 0, name="A")
        linkage = SymbolicLinkage([A])
        with pytest.raises(ValueError, match="not found"):
            linkage.coupler_curve("Z")

    def test_simplify_unknown_joint_type(self) -> None:
        """Test that simplify raises TypeError for unknown joint types."""
        from pylinkage.symbolic.joints import SymJoint

        class CustomJoint(SymJoint):
            """A custom joint type not handled by simplify."""
            def position_expr(self):
                return (sp.Integer(0), sp.Integer(0))
            def constraint_equations(self):
                return []

        A = CustomJoint(x=0, y=0, name="A")
        linkage = SymbolicLinkage([A])
        with pytest.raises(TypeError, match="Unknown joint type"):
            linkage.simplify()

    def test_substitute_unknown_joint_type(self) -> None:
        """Test that substitute raises TypeError for unknown joint types."""
        from pylinkage.symbolic.joints import SymJoint

        class CustomJoint(SymJoint):
            """A custom joint type not handled by substitute."""
            def position_expr(self):
                return (sp.Integer(0), sp.Integer(0))
            def constraint_equations(self):
                return []

        A = CustomJoint(x=0, y=0, name="A")
        linkage = SymbolicLinkage([A])
        with pytest.raises(TypeError, match="Unknown joint type"):
            linkage.substitute({"x": 1.0})

    def test_trajectory_static(self) -> None:
        """Test trajectories for static joints are constant."""
        A = SymStatic(3, 7, name="A")
        linkage = SymbolicLinkage([A])
        traj = linkage.get_trajectory_expressions()
        x, y = traj["A"]
        assert x == sp.Integer(3)
        assert y == sp.Integer(7)

    def test_branch_selection(self) -> None:
        """Test that branch parameter affects revolute position."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")

        C1 = SymRevolute(B, D, distance0=3, distance1=3, branch=1, name="C1")
        C2 = SymRevolute(B, D, distance0=3, distance1=3, branch=-1, name="C2")

        linkage1 = SymbolicLinkage([A, D, B, C1])
        linkage2 = SymbolicLinkage([A, D, B, C2])

        x1, y1 = linkage1.coupler_curve("C1")
        x2, y2 = linkage2.coupler_curve("C2")

        # At theta=0, the two branches should give different y values
        y1_val = float(y1.subs(theta, 0))
        y2_val = float(y2.subs(theta, 0))
        assert y1_val != pytest.approx(y2_val)
        # One should be positive, other negative
        assert y1_val * y2_val < 0
