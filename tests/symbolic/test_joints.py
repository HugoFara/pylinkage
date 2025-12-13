"""Tests for symbolic joint classes."""

import math

import pytest
import sympy as sp

from pylinkage.symbolic import SymCrank, SymRevolute, SymStatic, theta


class TestSymStatic:
    """Tests for SymStatic joint."""

    def test_numeric_position(self):
        """Test static joint with numeric position."""
        joint = SymStatic(3, 4, name="A")
        x, y = joint.position_expr()

        assert float(x) == 3.0
        assert float(y) == 4.0

    def test_symbolic_position(self):
        """Test static joint with symbolic position."""
        a = sp.Symbol("a")
        joint = SymStatic(a, 0, name="A")
        x, y = joint.position_expr()

        assert x == a
        assert float(y) == 0.0

    def test_no_constraints(self):
        """Test that static joint has no constraints."""
        joint = SymStatic(0, 0)
        assert joint.constraint_equations() == []

    def test_no_parameters_for_numeric(self):
        """Test that numeric static has no parameters."""
        joint = SymStatic(0, 0)
        assert joint.parameters == set()

    def test_parameters_for_symbolic(self):
        """Test that symbolic static has parameters."""
        a = sp.Symbol("a")
        joint = SymStatic(a, 0)
        assert a in joint.parameters


class TestSymCrank:
    """Tests for SymCrank joint."""

    def test_position_at_zero(self):
        """Test crank position at theta=0."""
        parent = SymStatic(0, 0, name="A")
        crank = SymCrank(parent, radius=1, name="B")

        x, y = crank.position_expr()
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))

        assert x_val == pytest.approx(1.0)
        assert y_val == pytest.approx(0.0)

    def test_position_at_quarter_turn(self):
        """Test crank position at theta=pi/2."""
        parent = SymStatic(0, 0, name="A")
        crank = SymCrank(parent, radius=1, name="B")

        x, y = crank.position_expr()
        x_val = float(x.subs(theta, sp.pi / 2))
        y_val = float(y.subs(theta, sp.pi / 2))

        assert x_val == pytest.approx(0.0, abs=1e-10)
        assert y_val == pytest.approx(1.0)

    def test_with_offset_parent(self):
        """Test crank with non-zero parent position."""
        parent = SymStatic(2, 3, name="A")
        crank = SymCrank(parent, radius=1, name="B")

        x, y = crank.position_expr()
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))

        assert x_val == pytest.approx(3.0)
        assert y_val == pytest.approx(3.0)

    def test_symbolic_radius(self):
        """Test crank with symbolic radius."""
        parent = SymStatic(0, 0, name="A")
        crank = SymCrank(parent, radius="r", name="B")

        x, y = crank.position_expr()
        r = sp.Symbol("r", positive=True, real=True)

        # At theta=0, position should be (r, 0)
        x_expr = x.subs(theta, 0)
        y_expr = y.subs(theta, 0)

        # r should be in the expression
        assert crank.r in x.free_symbols

    def test_coordinate_parent(self):
        """Test crank with coordinate tuple as parent."""
        crank = SymCrank(parent=(2, 3), radius=1, name="B")

        x, y = crank.position_expr()
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))

        assert x_val == pytest.approx(3.0)
        assert y_val == pytest.approx(3.0)

    def test_constraint_equation(self):
        """Test that crank has distance constraint."""
        parent = SymStatic(0, 0, name="A")
        crank = SymCrank(parent, radius=1, name="B")

        constraints = crank.constraint_equations()
        assert len(constraints) == 1

        # Constraint should be x^2 + y^2 = 1 (distance from origin)
        eq = constraints[0]
        assert isinstance(eq, sp.Eq)


class TestSymRevolute:
    """Tests for SymRevolute joint."""

    def test_fourbar_coupler(self):
        """Test revolute in a four-bar configuration."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, branch=1, name="C")

        x, y = C.position_expr()

        # At theta=0, B is at (1, 0)
        # C should be at intersection of circles:
        # - Circle centered at B=(1,0) with radius 3
        # - Circle centered at D=(4,0) with radius 3
        x_val = float(x.subs(theta, 0))
        y_val = float(y.subs(theta, 0))

        # Check that C is at distance 3 from both B and D
        b_x, b_y = 1, 0
        d_x, d_y = 4, 0
        dist_to_b = math.sqrt((x_val - b_x) ** 2 + (y_val - b_y) ** 2)
        dist_to_d = math.sqrt((x_val - d_x) ** 2 + (y_val - d_y) ** 2)

        assert dist_to_b == pytest.approx(3.0, abs=1e-10)
        assert dist_to_d == pytest.approx(3.0, abs=1e-10)

    def test_branch_affects_position(self):
        """Test that branch selection changes position."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")

        C1 = SymRevolute(B, D, distance0=3, distance1=3, branch=1, name="C1")
        C2 = SymRevolute(B, D, distance0=3, distance1=3, branch=-1, name="C2")

        x1, y1 = C1.position_expr()
        x2, y2 = C2.position_expr()

        y1_val = float(y1.subs(theta, 0))
        y2_val = float(y2.subs(theta, 0))

        # Branches should give opposite y values (symmetric about x-axis)
        assert y1_val == pytest.approx(-y2_val, abs=1e-10)

    def test_symbolic_distances(self):
        """Test revolute with symbolic distances."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0="r0", distance1="r1", name="C")

        # Should have symbolic parameters
        params = C.parameters
        assert len(params) >= 2  # At least r0 and r1

    def test_constraint_equations(self):
        """Test that revolute has two distance constraints."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius=1, name="B")
        C = SymRevolute(B, D, distance0=3, distance1=3, name="C")

        constraints = C.constraint_equations()
        assert len(constraints) == 2

        # Both should be Eq objects
        for eq in constraints:
            assert isinstance(eq, sp.Eq)
