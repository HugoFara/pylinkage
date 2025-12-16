"""Tests for line-line intersection geometry."""

import math

import pytest

from pylinkage.geometry.secants import (
    INTERSECTION_NONE,
    INTERSECTION_ONE,
    INTERSECTION_SAME,
    line_line_intersection,
)


class TestLineLineIntersection:
    """Test line_line_intersection function."""

    def test_simple_intersection(self):
        """Two perpendicular lines through origin."""
        # Line 1: horizontal through origin (y=0)
        # Line 2: vertical through origin (x=0)
        result = line_line_intersection(
            -1.0, 0.0, 1.0, 0.0,  # Line 1
            0.0, -1.0, 0.0, 1.0,  # Line 2
        )
        assert result[0] == INTERSECTION_ONE
        assert abs(result[1]) < 1e-10  # x = 0
        assert abs(result[2]) < 1e-10  # y = 0

    def test_intersection_at_non_origin(self):
        """Lines intersecting at (2, 3)."""
        # Line 1: y = 3 (horizontal)
        # Line 2: x = 2 (vertical)
        result = line_line_intersection(
            0.0, 3.0, 4.0, 3.0,  # Line 1: y = 3
            2.0, 0.0, 2.0, 5.0,  # Line 2: x = 2
        )
        assert result[0] == INTERSECTION_ONE
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[2] - 3.0) < 1e-10

    def test_diagonal_lines_intersection(self):
        """Two diagonal lines intersecting."""
        # Line 1: y = x (through origin at 45 degrees)
        # Line 2: y = -x + 2 (negative slope through (2, 0))
        # Intersection at (1, 1)
        result = line_line_intersection(
            0.0, 0.0, 2.0, 2.0,  # Line 1: y = x
            0.0, 2.0, 2.0, 0.0,  # Line 2: y = -x + 2
        )
        assert result[0] == INTERSECTION_ONE
        assert abs(result[1] - 1.0) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10

    def test_parallel_lines_no_intersection(self):
        """Parallel lines should return no intersection."""
        # Two horizontal lines
        result = line_line_intersection(
            0.0, 0.0, 2.0, 0.0,  # Line 1: y = 0
            0.0, 1.0, 2.0, 1.0,  # Line 2: y = 1
        )
        assert result[0] == INTERSECTION_NONE

    def test_parallel_diagonal_lines(self):
        """Parallel diagonal lines."""
        # Two parallel lines with slope 1
        result = line_line_intersection(
            0.0, 0.0, 1.0, 1.0,  # Line 1: y = x
            0.0, 1.0, 1.0, 2.0,  # Line 2: y = x + 1
        )
        assert result[0] == INTERSECTION_NONE

    def test_coincident_lines(self):
        """Same line should return INTERSECTION_SAME."""
        result = line_line_intersection(
            0.0, 0.0, 2.0, 2.0,  # Line 1
            1.0, 1.0, 3.0, 3.0,  # Line 2 (same line, different points)
        )
        assert result[0] == INTERSECTION_SAME

    def test_coincident_horizontal_lines(self):
        """Coincident horizontal lines."""
        result = line_line_intersection(
            0.0, 5.0, 3.0, 5.0,  # Line 1
            1.0, 5.0, 4.0, 5.0,  # Line 2 (same line)
        )
        assert result[0] == INTERSECTION_SAME

    def test_nearly_parallel_lines(self):
        """Lines that are almost but not quite parallel."""
        # Two lines with very similar slopes
        result = line_line_intersection(
            0.0, 0.0, 1000.0, 1000.0,  # Line 1: y = x
            0.0, 1.0, 1000.0, 1001.001,  # Line 2: y = 1.000001x + 1
        )
        # Should find an intersection somewhere far away
        assert result[0] == INTERSECTION_ONE

    def test_intersection_outside_segment(self):
        """Lines as infinite - intersection can be outside defined segments."""
        # Define short segments but intersection is valid for lines
        result = line_line_intersection(
            0.0, 0.0, 0.1, 0.1,  # Line 1: short segment near origin
            10.0, 0.0, 10.1, 0.1,  # Line 2: short segment far from origin
        )
        # Lines are parallel (same slope)
        assert result[0] == INTERSECTION_NONE

    def test_obtuse_angle_intersection(self):
        """Lines meeting at an obtuse angle."""
        result = line_line_intersection(
            0.0, 0.0, 10.0, 1.0,  # Line 1: shallow slope
            10.0, 0.0, 0.0, 1.0,  # Line 2: negative shallow slope
        )
        assert result[0] == INTERSECTION_ONE
        # Intersection should be around x=5, y=0.5
        assert 4.9 < result[1] < 5.1
        assert 0.4 < result[2] < 0.6


class TestLineLineIntersectionNumerical:
    """Numerical stability tests."""

    def test_large_coordinates(self):
        """Lines with large coordinate values."""
        result = line_line_intersection(
            1000.0, 1000.0, 1001.0, 1000.0,  # Horizontal at y=1000
            1000.5, 999.0, 1000.5, 1001.0,  # Vertical at x=1000.5
        )
        assert result[0] == INTERSECTION_ONE
        assert abs(result[1] - 1000.5) < 1e-9
        assert abs(result[2] - 1000.0) < 1e-9

    def test_small_coordinates(self):
        """Lines with small coordinate values."""
        result = line_line_intersection(
            0.0001, 0.0001, 0.0002, 0.0001,  # Horizontal
            0.00015, 0.0, 0.00015, 0.001,  # Vertical
        )
        assert result[0] == INTERSECTION_ONE
        assert abs(result[1] - 0.00015) < 1e-12
        assert abs(result[2] - 0.0001) < 1e-12
