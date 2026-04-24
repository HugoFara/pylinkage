"""Extended tests for geometry.core covering all functions."""

from __future__ import annotations

import math

import pytest

from pylinkage.geometry.core import (
    cyl_to_cart,
    dist,
    get_nearest_point,
    line_from_points,
    norm,
    sqr_dist,
)


class TestDist:
    def test_horizontal(self):
        assert dist(0.0, 0.0, 5.0, 0.0) == 5.0

    def test_vertical(self):
        assert dist(0.0, 0.0, 0.0, 4.0) == 4.0

    def test_pythag(self):
        assert dist(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_same_point(self):
        assert dist(2.5, 2.5, 2.5, 2.5) == 0.0

    def test_negative(self):
        assert dist(-3.0, -4.0, 0.0, 0.0) == pytest.approx(5.0)


class TestSqrDist:
    def test_pythag(self):
        assert sqr_dist(0.0, 0.0, 3.0, 4.0) == 25.0

    def test_same(self):
        assert sqr_dist(1.0, 1.0, 1.0, 1.0) == 0.0

    def test_axis_aligned(self):
        assert sqr_dist(0.0, 0.0, 7.0, 0.0) == 49.0


class TestGetNearestPoint:
    def test_first_closer(self):
        x, y = get_nearest_point(0.0, 0.0, 1.0, 0.0, 10.0, 0.0)
        assert (x, y) == (1.0, 0.0)

    def test_second_closer(self):
        x, y = get_nearest_point(0.0, 0.0, 10.0, 0.0, 1.0, 0.0)
        assert (x, y) == (1.0, 0.0)

    def test_equidistant(self):
        # When equidistant, second point wins based on the if d1 < d2 check
        x, y = get_nearest_point(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        assert (x, y) == (0.0, 1.0)


class TestNorm:
    def test_zero(self):
        assert norm(0.0, 0.0) == 0.0

    def test_unit_x(self):
        assert norm(1.0, 0.0) == 1.0

    def test_unit_y(self):
        assert norm(0.0, 1.0) == 1.0

    def test_three_four_five(self):
        assert norm(3.0, 4.0) == pytest.approx(5.0)

    def test_negative(self):
        assert norm(-3.0, -4.0) == pytest.approx(5.0)


class TestCylToCart:
    def test_zero_angle(self):
        x, y = cyl_to_cart(1.0, 0.0)
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_quarter_turn(self):
        x, y = cyl_to_cart(2.0, math.pi / 2)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(2.0)

    def test_half_turn(self):
        x, y = cyl_to_cart(3.0, math.pi)
        assert x == pytest.approx(-3.0)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_with_offset_origin(self):
        x, y = cyl_to_cart(1.0, 0.0, ori_x=10.0, ori_y=20.0)
        assert x == pytest.approx(11.0)
        assert y == pytest.approx(20.0, abs=1e-12)

    def test_zero_radius(self):
        x, y = cyl_to_cart(0.0, 1.2345, ori_x=3.0, ori_y=4.0)
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(4.0)


class TestLineFromPoints:
    def test_same_point(self):
        assert line_from_points(1.0, 2.0, 1.0, 2.0) == (0.0, 0.0, 0.0)

    def test_horizontal_line(self):
        # Two points on the line y = 3
        a, b, c = line_from_points(0.0, 3.0, 5.0, 3.0)
        # Both points should satisfy a*x + b*y + c = 0
        assert a * 0.0 + b * 3.0 + c == pytest.approx(0.0)
        assert a * 5.0 + b * 3.0 + c == pytest.approx(0.0)

    def test_vertical_line(self):
        a, b, c = line_from_points(4.0, 0.0, 4.0, 5.0)
        assert a * 4.0 + b * 0.0 + c == pytest.approx(0.0)
        assert a * 4.0 + b * 5.0 + c == pytest.approx(0.0)

    def test_diagonal_line(self):
        # y = x
        a, b, c = line_from_points(0.0, 0.0, 1.0, 1.0)
        assert a * 2.0 + b * 2.0 + c == pytest.approx(0.0)
        assert a * -1.0 + b * -1.0 + c == pytest.approx(0.0)
