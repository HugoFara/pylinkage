"""Tests for symbolic geometry functions."""

import math

import numpy as np
import pytest
import sympy as sp

from pylinkage.geometry import circle_intersect
from pylinkage.symbolic import (
    symbolic_circle_intersect,
    symbolic_cyl_to_cart,
    symbolic_dist,
    symbolic_sqr_dist,
)


class TestSymbolicDist:
    """Tests for symbolic distance functions."""

    def test_numeric_values(self):
        """Test with numeric values."""
        result = symbolic_dist(0, 0, 3, 4)
        assert float(result) == pytest.approx(5.0)

    def test_symbolic_values(self):
        """Test with symbolic values."""
        x = sp.Symbol("x")
        result = symbolic_dist(0, 0, x, 0)
        # Should be |x| = sqrt(x^2)
        assert result == sp.sqrt(x**2)

    def test_sqr_dist(self):
        """Test squared distance."""
        result = symbolic_sqr_dist(0, 0, 3, 4)
        assert float(result) == 25.0


class TestSymbolicCylToCart:
    """Tests for polar to cartesian conversion."""

    def test_zero_angle(self):
        """Test at theta=0."""
        x, y = symbolic_cyl_to_cart(1, 0)
        assert float(x) == pytest.approx(1.0)
        assert float(y) == pytest.approx(0.0)

    def test_quarter_turn(self):
        """Test at theta=pi/2."""
        x, y = symbolic_cyl_to_cart(1, sp.pi / 2)
        assert float(x) == pytest.approx(0.0, abs=1e-10)
        assert float(y) == pytest.approx(1.0)

    def test_with_origin(self):
        """Test with non-zero origin."""
        x, y = symbolic_cyl_to_cart(1, 0, 2, 3)
        assert float(x) == pytest.approx(3.0)
        assert float(y) == pytest.approx(3.0)

    def test_symbolic_angle(self):
        """Test with symbolic angle."""
        theta = sp.Symbol("theta")
        x, y = symbolic_cyl_to_cart(1, theta)
        assert x == sp.cos(theta)
        assert y == sp.sin(theta)


class TestSymbolicCircleIntersect:
    """Tests for symbolic circle-circle intersection."""

    def test_unit_circles_at_origin_and_one(self):
        """Test two unit circles at (0,0) and (1,0)."""
        # Two unit circles with centers distance 1 apart
        # Should intersect at (0.5, +/-sqrt(3)/2)
        x, y = symbolic_circle_intersect(0, 0, 1, 1, 0, 1, branch=1)
        x_val = float(x)
        y_val = float(y)

        assert x_val == pytest.approx(0.5, abs=1e-10)
        assert abs(y_val) == pytest.approx(math.sqrt(3) / 2, abs=1e-10)

    def test_branch_selection(self):
        """Test that branch +1 and -1 give different points."""
        x1, y1 = symbolic_circle_intersect(0, 0, 1, 1, 0, 1, branch=1)
        x2, y2 = symbolic_circle_intersect(0, 0, 1, 1, 0, 1, branch=-1)

        x1_val, y1_val = float(x1), float(y1)
        x2_val, y2_val = float(x2), float(y2)

        # Both should have same x but opposite y
        assert x1_val == pytest.approx(x2_val, abs=1e-10)
        assert y1_val == pytest.approx(-y2_val, abs=1e-10)

    def test_matches_numeric_implementation(self):
        """Test that symbolic results match numeric implementation."""
        # Test case: circles at (0,0) r=2 and (3,0) r=2
        n_result = circle_intersect(0, 0, 2, 3, 0, 2)
        # n_result is (n_intersections, x1, y1, x2, y2)

        assert n_result[0] == 2  # Two intersections

        # Get symbolic result
        sx1, sy1 = symbolic_circle_intersect(0, 0, 2, 3, 0, 2, branch=1)
        sx2, sy2 = symbolic_circle_intersect(0, 0, 2, 3, 0, 2, branch=-1)

        sx1_val, sy1_val = float(sx1), float(sy1)
        sx2_val, sy2_val = float(sx2), float(sy2)

        # One of the symbolic results should match each numeric result
        # (order might be different based on branch)
        points_sym = {(round(sx1_val, 10), round(sy1_val, 10)),
                      (round(sx2_val, 10), round(sy2_val, 10))}
        points_num = {(round(n_result[1], 10), round(n_result[2], 10)),
                      (round(n_result[3], 10), round(n_result[4], 10))}

        assert points_sym == points_num

    def test_symbolic_parameters(self):
        """Test with symbolic parameters."""
        r1 = sp.Symbol("r1", positive=True)
        r2 = sp.Symbol("r2", positive=True)

        x, y = symbolic_circle_intersect(0, 0, r1, 1, 0, r2, branch=1)

        # Should contain r1 and r2 in the expression
        assert r1 in x.free_symbols or r1 in y.free_symbols
        assert r2 in x.free_symbols or r2 in y.free_symbols

        # Substituting values should give numeric result
        x_val = float(x.subs([(r1, 1), (r2, 1)]))
        y_val = float(y.subs([(r1, 1), (r2, 1)]))

        assert x_val == pytest.approx(0.5, abs=1e-10)
        assert abs(y_val) == pytest.approx(math.sqrt(3) / 2, abs=1e-10)
