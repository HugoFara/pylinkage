"""Extended tests for geometry.secants covering uncovered branches."""

from __future__ import annotations

import math

import pytest

from pylinkage.geometry.core import dist
from pylinkage.geometry.secants import (
    INTERSECTION_NONE,
    INTERSECTION_ONE,
    INTERSECTION_SAME,
    INTERSECTION_TWO,
    bounding_box,
    circle_intersect,
    circle_line_from_points_intersection,
    circle_line_intersection,
    intersection,
    line_line_intersection,
)


class TestCircleIntersectBranches:
    def test_internal_tangent(self):
        # One circle internally tangent: |distance - |r2 - r1|| == eps
        # d = 2, r1=3, r2=1 -> internally tangent at (3, 0) + ?
        result = circle_intersect(0.0, 0.0, 3.0, 2.0, 0.0, 1.0, tol=0.01)
        assert result[0] == INTERSECTION_ONE

    def test_external_tangent_distance_equal_sum(self):
        result = circle_intersect(0.0, 0.0, 1.0, 2.0, 0.0, 1.0, tol=0.01)
        assert result[0] == INTERSECTION_ONE

    def test_same_circle_zero_distance(self):
        result = circle_intersect(1.0, 1.0, 2.0, 1.0, 1.0, 2.0, tol=0.0)
        assert result[0] == INTERSECTION_SAME
        assert result[1] == 1.0
        assert result[2] == 1.0
        assert result[3] == 2.0

    def test_two_intersections_vertical(self):
        result = circle_intersect(0.0, 0.0, 2.0, 0.0, 3.0, 2.0)
        assert result[0] == INTERSECTION_TWO
        # Both points should be at distance 2 from both centers
        assert dist(0, 0, result[1], result[2]) == pytest.approx(2.0, abs=1e-8)
        assert dist(0, 3, result[1], result[2]) == pytest.approx(2.0, abs=1e-8)
        assert dist(0, 0, result[3], result[4]) == pytest.approx(2.0, abs=1e-8)
        assert dist(0, 3, result[3], result[4]) == pytest.approx(2.0, abs=1e-8)

    def test_far_apart_with_tolerance(self):
        result = circle_intersect(0.0, 0.0, 1.0, 100.0, 0.0, 1.0, tol=0.01)
        assert result[0] == INTERSECTION_NONE

    def test_inside_with_tolerance(self):
        result = circle_intersect(0.0, 0.0, 10.0, 0.5, 0.0, 1.0, tol=0.01)
        assert result[0] == INTERSECTION_NONE


class TestCircleLineFromPointsIntersectionBranches:
    def test_tangent_exact(self):
        # A line tangent to a circle yields discriminant == 0 (or very close)
        # Tangent line y = 1 to unit circle
        result = circle_line_from_points_intersection(
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0
        )
        # Could be one tangent or two (very close) — accept either
        assert result[0] in (INTERSECTION_ONE, INTERSECTION_TWO)

    def test_no_intersection(self):
        result = circle_line_from_points_intersection(
            0.0, 0.0, 1.0, -5.0, 5.0, 5.0, 5.0
        )
        assert result[0] == INTERSECTION_NONE

    def test_two_intersections_positive_dy(self):
        # Vertical line crossing circle from below to above
        result = circle_line_from_points_intersection(
            0.0, 0.0, 2.0, 1.0, -5.0, 1.0, 5.0
        )
        assert result[0] == INTERSECTION_TWO

    def test_two_intersections_negative_dy(self):
        # Vertical line with dy < 0
        result = circle_line_from_points_intersection(
            0.0, 0.0, 2.0, 1.0, 5.0, 1.0, -5.0
        )
        assert result[0] == INTERSECTION_TWO

    def test_horizontal_line_through_center(self):
        result = circle_line_from_points_intersection(
            0.0, 0.0, 2.0, -5.0, 0.0, 5.0, 0.0
        )
        assert result[0] == INTERSECTION_TWO


class TestCircleLineIntersection:
    def test_with_b_zero_branch(self):
        # Line a*x + c = 0 (vertical line x = -c/a): b = 0
        # Use the vertical line x = 1 -> a=1, b=0, c=-1
        result = circle_line_intersection(0.0, 0.0, 2.0, 1.0, 0.0, -1.0)
        assert result[0] == INTERSECTION_TWO

    def test_with_b_nonzero(self):
        # Line y = 0 -> a=0, b=1, c=0
        result = circle_line_intersection(0.0, 0.0, 2.0, 0.0, 1.0, 0.0)
        assert result[0] == INTERSECTION_TWO

    def test_line_too_far(self):
        result = circle_line_intersection(0.0, 0.0, 1.0, 0.0, 1.0, -10.0)
        assert result[0] == INTERSECTION_NONE


class TestIntersectionMisc:
    def test_one_intersection_circles(self):
        # External tangent
        result = intersection((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), tol=0.01)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_same_circle_tolerance(self):
        result = intersection((0.0, 0.0, 2.0), (0.0, 0.0, 2.0), tol=0.01)
        assert result == (0.0, 0.0, 2.0)

    def test_invalid_length_returns_none(self):
        # Length 4 inputs — no branch matches
        result = intersection((0.0, 0.0, 1.0, 2.0), (0.0, 0.0, 1.0, 2.0))  # type: ignore[arg-type]
        assert result is None


class TestLineLineIntersection:
    def test_intersecting_lines(self):
        # Line 1: y = x, Line 2: y = -x + 2
        # Intersection at (1, 1)
        result = line_line_intersection(0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0)
        assert result[0] == INTERSECTION_ONE
        assert result[1] == pytest.approx(1.0, abs=1e-8)
        assert result[2] == pytest.approx(1.0, abs=1e-8)

    def test_parallel_lines_no_intersection(self):
        # Two parallel horizontal lines
        result = line_line_intersection(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert result[0] == INTERSECTION_NONE

    def test_coincident_lines(self):
        # Same line
        result = line_line_intersection(0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0)
        assert result[0] == INTERSECTION_SAME

    def test_perpendicular_lines_at_origin(self):
        result = line_line_intersection(-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0)
        assert result[0] == INTERSECTION_ONE
        assert result[1] == pytest.approx(0.0, abs=1e-10)
        assert result[2] == pytest.approx(0.0, abs=1e-10)


class TestBoundingBoxExtended:
    def test_empty_locus(self):
        result = bounding_box([])
        assert result[0] == math.inf
        assert result[1] == -math.inf
        assert result[2] == -math.inf
        assert result[3] == math.inf

    def test_many_points(self):
        locus = [(i, i * 2) for i in range(10)]
        y_min, x_max, y_max, x_min = bounding_box(locus)
        assert y_min == 0
        assert y_max == 18
        assert x_min == 0
        assert x_max == 9
