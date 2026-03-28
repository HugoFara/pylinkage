"""Extended tests for symbolic geometry -- targeting uncovered lines 172-211."""

import math

import pytest
import sympy as sp

from pylinkage.symbolic.geometry import symbolic_circle_line_intersect


class TestSymbolicCircleLineIntersect:
    """Tests for symbolic_circle_line_intersect (lines 172-211).

    Note: The algorithm uses a cross-product formulation that can be degenerate
    when the line is exactly horizontal (dy=0). We test with non-degenerate
    cases to exercise the code paths.
    """

    def test_vertical_line_through_center(self):
        """Circle at origin, vertical line x=0, should intersect at (0, +/-r)."""
        # Circle: center (0,0), radius 2
        # Line: from (0, -5) to (0, 5) -- vertical, dy != 0
        x1, y1 = symbolic_circle_line_intersect(0, 0, 2, 0, -5, 0, 5, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 2, 0, -5, 0, 5, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)
        # Both x should be 0
        assert x1_val == pytest.approx(0.0, abs=1e-10)
        assert x2_val == pytest.approx(0.0, abs=1e-10)
        # y values should be +/-2
        y_vals = sorted([y1_val, y2_val])
        assert y_vals[0] == pytest.approx(-2.0, abs=1e-10)
        assert y_vals[1] == pytest.approx(2.0, abs=1e-10)

    def test_vertical_line_offset(self):
        """Circle at origin, vertical line x=1, radius 2."""
        # Should intersect at (1, +/-sqrt(3))
        x1, y1 = symbolic_circle_line_intersect(0, 0, 2, 1, -5, 1, 5, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 2, 1, -5, 1, 5, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)
        assert x1_val == pytest.approx(1.0, abs=1e-10)
        assert x2_val == pytest.approx(1.0, abs=1e-10)
        y_vals = sorted([y1_val, y2_val])
        assert y_vals[0] == pytest.approx(-math.sqrt(3), abs=1e-10)
        assert y_vals[1] == pytest.approx(math.sqrt(3), abs=1e-10)

    def test_diagonal_line(self):
        """Circle at origin, diagonal line y=x, radius 1."""
        # Intersections at (1/sqrt(2), 1/sqrt(2)) and (-1/sqrt(2), -1/sqrt(2))
        x1, y1 = symbolic_circle_line_intersect(0, 0, 1, -1, -1, 1, 1, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 1, -1, -1, 1, 1, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)

        # Both points should be on unit circle
        assert x1_val**2 + y1_val**2 == pytest.approx(1.0, abs=1e-8)
        assert x2_val**2 + y2_val**2 == pytest.approx(1.0, abs=1e-8)
        # Both points should be on line y=x
        assert x1_val == pytest.approx(y1_val, abs=1e-8)
        assert x2_val == pytest.approx(y2_val, abs=1e-8)

    def test_branch_selection_gives_different_points(self):
        """Branch +1 and -1 should give different intersection points."""
        x1, y1 = symbolic_circle_line_intersect(0, 0, 2, 1, -5, 1, 5, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 2, 1, -5, 1, 5, branch=-1)
        _x1_val, y1_val = float(x1), float(y1)
        _x2_val, y2_val = float(x2), float(y2)
        # y values should differ
        assert y1_val != pytest.approx(y2_val, abs=1e-8)

    def test_offset_circle_diagonal_line(self):
        """Circle at (3, 4) with radius 5, line y=x."""
        # (x-3)^2 + (y-4)^2 = 25 and y=x
        # (x-3)^2 + (x-4)^2 = 25 => 2x^2 - 14x + 25 = 25 => x(2x-14) = 0
        # x = 0 or x = 7
        x1, y1 = symbolic_circle_line_intersect(3, 4, 5, -10, -10, 10, 10, branch=1)
        x2, y2 = symbolic_circle_line_intersect(3, 4, 5, -10, -10, 10, 10, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)
        # Points should be on line y=x
        assert x1_val == pytest.approx(y1_val, abs=1e-8)
        assert x2_val == pytest.approx(y2_val, abs=1e-8)
        # Points should be on circle
        assert (x1_val - 3)**2 + (y1_val - 4)**2 == pytest.approx(25.0, abs=1e-6)
        assert (x2_val - 3)**2 + (y2_val - 4)**2 == pytest.approx(25.0, abs=1e-6)
        # Should be at x=0 and x=7
        x_vals = sorted([x1_val, x2_val])
        assert x_vals[0] == pytest.approx(0.0, abs=1e-8)
        assert x_vals[1] == pytest.approx(7.0, abs=1e-8)

    def test_symbolic_parameters(self):
        """Test with symbolic radius parameter."""
        r = sp.Symbol("r", positive=True)
        # Use a non-degenerate line (vertical: x=0)
        x, y = symbolic_circle_line_intersect(0, 0, r, 0, -1, 0, 1, branch=1)
        # Should contain r in expression
        all_syms = x.free_symbols | y.free_symbols
        assert r in all_syms
        # Substituting r=3 should give numeric result on the circle
        x_val = float(x.subs(r, 3))
        y_val = float(y.subs(r, 3))
        assert x_val**2 + y_val**2 == pytest.approx(9.0, abs=1e-8)

    def test_tangent_vertical_line(self):
        """Vertical tangent line x=1 on unit circle."""
        # Circle at origin, radius 1, vertical tangent at x=1
        x1, y1 = symbolic_circle_line_intersect(0, 0, 1, 1, -5, 1, 5, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 1, 1, -5, 1, 5, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)
        # Both should be at (1, 0)
        assert x1_val == pytest.approx(x2_val, abs=1e-10)
        assert y1_val == pytest.approx(y2_val, abs=1e-10)
        assert x1_val == pytest.approx(1.0, abs=1e-10)
        assert y1_val == pytest.approx(0.0, abs=1e-10)

    def test_negative_dy_line(self):
        """Test line with negative dy direction (sign_dy branch)."""
        # Line from (0, 5) to (0, -5) -- vertical with negative dy
        x1, y1 = symbolic_circle_line_intersect(0, 0, 2, 0, 5, 0, -5, branch=1)
        x1_val, y1_val = float(x1), float(y1)
        # Point should be on circle
        assert x1_val**2 + y1_val**2 == pytest.approx(4.0, abs=1e-8)
        # Point should be on line x=0
        assert x1_val == pytest.approx(0.0, abs=1e-10)

    def test_angled_line_not_through_origin(self):
        """Test an angled line not passing through the circle center."""
        # Circle at (0,0) radius 3, line from (0, 2) to (4, 4)
        # dy = 2, non-zero, exercises sign_dy and Abs(dy) paths
        x1, y1 = symbolic_circle_line_intersect(0, 0, 3, 0, 2, 4, 4, branch=1)
        x2, y2 = symbolic_circle_line_intersect(0, 0, 3, 0, 2, 4, 4, branch=-1)
        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)
        # Both points should be on circle
        assert x1_val**2 + y1_val**2 == pytest.approx(9.0, abs=1e-6)
        assert x2_val**2 + y2_val**2 == pytest.approx(9.0, abs=1e-6)
        # Both points should be on line: y = 0.5*x + 2
        assert y1_val == pytest.approx(0.5 * x1_val + 2, abs=1e-6)
        assert y2_val == pytest.approx(0.5 * x2_val + 2, abs=1e-6)
