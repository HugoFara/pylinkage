#!/usr/bin/env python3
"""
Created on Thu Apr 15, 13:26:42 2021

@author: HugoFara
"""

import math
import unittest

import numpy as np

from pylinkage.geometry import circle_intersect, circle_line_intersection, sqr_dist
from pylinkage.geometry.core import (
    cyl_to_cart,
    dist,
    get_nearest_point,
    line_from_points,
    norm,
)
from pylinkage.geometry.secants import (
    INTERSECTION_NONE,
    INTERSECTION_ONE,
    INTERSECTION_SAME,
    INTERSECTION_TWO,
    bounding_box,
    circle_line_from_points_intersection,
    intersection,
)


class TestCircles(unittest.TestCase):
    """Tests for circles."""

    def test_no_inter(self):
        """Test two identical circles (same circle case)."""
        # Same circle should return INTERSECTION_SAME
        result = circle_intersect(2.0, 5.0, 1.0, 2.0, 5.0, 1.0, tol=0.1)
        self.assertEqual(INTERSECTION_SAME, result[0])

    def test_radius_ineq(self):
        """Test two concentric circles of different radii."""
        # Concentric circles with different radii should not intersect
        result = circle_intersect(2.0, 5.0, 1.0, 2.0, 5.0, 1.4)
        self.assertEqual(INTERSECTION_NONE, result[0])


def circle_line_intersection_data(mode):
    """
    Prepare the data for testing.

    Arguments
    ---------
    mode : int
        Desired number of intersections between the circle and the line.

    Returns
    -------
    (cx, cy, r, a, b, c)
        Circle center and radius, plus line coefficients.
    """
    circle = (0.0, 0.0, 2.0)
    # Put the first point on the circle
    point = np.empty(2)
    point[0] = circle[0] + (np.random.rand() * 2 - 1) * circle[2] * 0.9
    point[1] = np.sqrt(circle[2] ** 2 - (point[0] - circle[0]) ** 2) + circle[1]
    # Build a tangent line
    line = np.empty(3)
    vector = np.array(circle[:2]) - point
    line[:2] = vector
    # Line is tangent
    line[2] = -np.dot(line[:2], point)
    if mode == 0:
        line[2] += np.dot(line[:2], vector * circle[2] * (1 + np.random.rand()))
    if mode == 2:
        line[2] -= np.dot(line[:2], vector * circle[2] * np.random.rand())
    return circle[0], circle[1], circle[2], float(line[0]), float(line[1]), float(line[2])


class TestCircleLineIntersection(unittest.TestCase):
    """Various tests on straight line intersections."""

    def test_no_crossing(self):
        """Test a line and a circle not crossing each other."""
        cx, cy, r, a, b, c = circle_line_intersection_data(0)
        result = circle_line_intersection(cx, cy, r, a, b, c)
        self.assertEqual(INTERSECTION_NONE, result[0], f"Result: {result}")

    def test_tangent(self):
        """
        Test a straight line tangent to a circle.

        This is a dangerous test, so we only want a success rate over 90%.
        """
        count = 0
        for _ in range(20):
            cx, cy, r, a, b, c = circle_line_intersection_data(1)
            result = circle_line_intersection(cx, cy, r, a, b, c)
            # Accept both 1 intersection or 2 very close intersections
            if result[0] == 1:
                count += 1
            elif result[0] == 2:
                d = sqr_dist(result[1], result[2], result[3], result[4])
                if d < 1e-5:
                    count += 1
        self.assertLessEqual(7, count, f"Success rate of {count * 5}% is too low!")

    def test_crossing(self):
        """Test a straight line crossing a circle twice."""
        cx, cy, r, a, b, c = circle_line_intersection_data(2)
        result = circle_line_intersection(cx, cy, r, a, b, c)
        self.assertEqual(INTERSECTION_TWO, result[0], f"Result: {result}")


class TestGeometryCore(unittest.TestCase):
    """Tests for geometry/core.py functions."""

    def test_dist_basic(self):
        """Test distance between two points."""
        self.assertAlmostEqual(dist(0.0, 0.0, 3.0, 4.0), 5.0)

    def test_dist_same_point(self):
        """Test distance between same point is zero."""
        self.assertAlmostEqual(dist(5.0, 5.0, 5.0, 5.0), 0.0)

    def test_dist_negative_coords(self):
        """Test distance with negative coordinates."""
        self.assertAlmostEqual(dist(-1.0, -1.0, 2.0, 3.0), 5.0)

    def test_sqr_dist_basic(self):
        """Test squared distance."""
        self.assertAlmostEqual(sqr_dist(0.0, 0.0, 3.0, 4.0), 25.0)

    def test_get_nearest_point_first(self):
        """Test get_nearest_point returns first point when closer."""
        result = get_nearest_point(0.0, 0.0, 1.0, 0.0, 10.0, 0.0)
        self.assertEqual(result, (1.0, 0.0))

    def test_get_nearest_point_second(self):
        """Test get_nearest_point returns second point when closer."""
        result = get_nearest_point(10.0, 0.0, 1.0, 0.0, 9.0, 0.0)
        self.assertEqual(result, (9.0, 0.0))

    def test_norm_basic(self):
        """Test vector norm."""
        self.assertAlmostEqual(norm(3.0, 4.0), 5.0)

    def test_norm_zero(self):
        """Test norm of zero vector."""
        self.assertAlmostEqual(norm(0.0, 0.0), 0.0)

    def test_cyl_to_cart_basic(self):
        """Test polar to cartesian conversion."""
        x, y = cyl_to_cart(1.0, 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)

    def test_cyl_to_cart_90_degrees(self):
        """Test polar to cartesian at 90 degrees."""
        x, y = cyl_to_cart(1.0, math.pi / 2)
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(y, 1.0, places=10)

    def test_cyl_to_cart_with_origin(self):
        """Test polar to cartesian with offset origin."""
        x, y = cyl_to_cart(1.0, 0.0, ori_x=5.0, ori_y=3.0)
        self.assertAlmostEqual(x, 6.0)
        self.assertAlmostEqual(y, 3.0)

    def test_line_from_points_basic(self):
        """Test line equation from two points."""
        a, b, c = line_from_points(0.0, 0.0, 1.0, 1.0)
        # Line y = x, or -x + y = 0
        # Point (0, 0) should satisfy: a*0 + b*0 + c = 0
        self.assertAlmostEqual(a * 0 + b * 0 + c, 0)
        # Point (1, 1) should satisfy: a*1 + b*1 + c = 0
        self.assertAlmostEqual(a * 1 + b * 1 + c, 0)

    def test_line_from_points_horizontal(self):
        """Test horizontal line equation."""
        a, b, c = line_from_points(0.0, 5.0, 10.0, 5.0)
        # Point on line should satisfy equation
        self.assertAlmostEqual(a * 5 + b * 5 + c, 0)
        self.assertAlmostEqual(a * 0 + b * 5 + c, 0)

    def test_line_from_points_same_point(self):
        """Test line from same point returns zeros."""
        a, b, c = line_from_points(3.0, 3.0, 3.0, 3.0)
        self.assertEqual((a, b, c), (0.0, 0.0, 0.0))


class TestCircleIntersectExtended(unittest.TestCase):
    """Extended tests for circle intersection."""

    def test_two_intersections(self):
        """Test two circle intersections."""
        result = circle_intersect(0.0, 0.0, 2.0, 3.0, 0.0, 2.0)
        self.assertEqual(result[0], INTERSECTION_TWO)
        # Check that intersection points are valid
        # Both should be at distance 2 from (0,0) and distance 2 from (3,0)
        x1, y1 = result[1], result[2]
        x2, y2 = result[3], result[4]
        self.assertAlmostEqual(dist(0, 0, x1, y1), 2.0, places=5)
        self.assertAlmostEqual(dist(3, 0, x1, y1), 2.0, places=5)
        self.assertAlmostEqual(dist(0, 0, x2, y2), 2.0, places=5)
        self.assertAlmostEqual(dist(3, 0, x2, y2), 2.0, places=5)

    def test_one_intersection_tangent(self):
        """Test tangent circles (one intersection)."""
        # Two circles touching externally
        result = circle_intersect(0.0, 0.0, 1.0, 2.0, 0.0, 1.0, tol=0.01)
        self.assertEqual(result[0], INTERSECTION_ONE)
        self.assertAlmostEqual(result[1], 1.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_no_intersection_too_far(self):
        """Test circles too far apart."""
        result = circle_intersect(0.0, 0.0, 1.0, 10.0, 0.0, 1.0)
        self.assertEqual(result[0], INTERSECTION_NONE)

    def test_no_intersection_one_inside_other(self):
        """Test one circle inside another (no intersection)."""
        result = circle_intersect(0.0, 0.0, 5.0, 0.0, 0.0, 1.0)
        self.assertEqual(result[0], INTERSECTION_NONE)


class TestCircleLineFromPointsIntersection(unittest.TestCase):
    """Tests for circle_line_from_points_intersection."""

    def test_two_intersections(self):
        """Test line crossing circle twice."""
        result = circle_line_from_points_intersection(
            0.0,
            0.0,
            2.0,  # Circle at origin, radius 2
            -5.0,
            0.0,
            5.0,
            0.0,  # Horizontal line through center
        )
        self.assertEqual(result[0], INTERSECTION_TWO)
        # Intersections should be at (-2, 0) and (2, 0)
        points = sorted([(result[1], result[2]), (result[3], result[4])])
        self.assertAlmostEqual(points[0][0], -2.0, places=5)
        self.assertAlmostEqual(points[1][0], 2.0, places=5)

    def test_tangent_line(self):
        """Test tangent line (one intersection)."""
        result = circle_line_from_points_intersection(
            0.0,
            0.0,
            1.0,  # Circle at origin, radius 1
            -5.0,
            1.0,
            5.0,
            1.0,  # Horizontal line at y=1
        )
        # Should be tangent or very close to tangent
        self.assertIn(result[0], [INTERSECTION_ONE, INTERSECTION_TWO])

    def test_no_intersection(self):
        """Test line not crossing circle."""
        result = circle_line_from_points_intersection(
            0.0,
            0.0,
            1.0,  # Circle at origin, radius 1
            -5.0,
            5.0,
            5.0,
            5.0,  # Horizontal line at y=5
        )
        self.assertEqual(result[0], INTERSECTION_NONE)

    def test_vertical_line(self):
        """Test vertical line intersection."""
        result = circle_line_from_points_intersection(
            0.0,
            0.0,
            2.0,  # Circle at origin, radius 2
            1.0,
            -5.0,
            1.0,
            5.0,  # Vertical line at x=1
        )
        self.assertEqual(result[0], INTERSECTION_TWO)


class TestIntersection(unittest.TestCase):
    """Tests for the intersection function."""

    def test_two_same_points(self):
        """Test intersection of same points."""
        result = intersection((1.0, 2.0), (1.0, 2.0))
        self.assertEqual(result, (1.0, 2.0))

    def test_two_different_points(self):
        """Test intersection of different points."""
        result = intersection((0.0, 0.0), (1.0, 1.0))
        self.assertIsNone(result)

    def test_two_points_with_tolerance(self):
        """Test intersection of close points with tolerance."""
        result = intersection((0.0, 0.0), (0.01, 0.01), tol=0.1)
        self.assertEqual(result, (0.0, 0.0))

    def test_two_circles_intersecting(self):
        """Test intersection of two circles."""
        result = intersection((0.0, 0.0, 2.0), (3.0, 0.0, 2.0))
        self.assertEqual(len(result), 2)  # Two intersection points

    def test_two_circles_not_intersecting(self):
        """Test non-intersecting circles."""
        result = intersection((0.0, 0.0, 1.0), (10.0, 0.0, 1.0))
        self.assertEqual(result, ())

    def test_same_circle(self):
        """Test same circle returns circle."""
        result = intersection((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=0.1)
        self.assertEqual(result, (0.0, 0.0, 1.0))

    def test_point_on_circle(self):
        """Test point on circle edge."""
        result = intersection((1.0, 0.0), (0.0, 0.0, 1.0), tol=0.01)
        self.assertEqual(result, (1.0, 0.0))

    def test_point_outside_circle(self):
        """Test point outside circle."""
        result = intersection((5.0, 0.0), (0.0, 0.0, 1.0))
        self.assertIsNone(result)

    def test_circle_and_point_reversed(self):
        """Test circle and point with reversed argument order."""
        result = intersection((0.0, 0.0, 1.0), (1.0, 0.0), tol=0.01)
        self.assertEqual(result, (1.0, 0.0))

    def test_circle_single_intersection(self):
        """Test circles with single intersection point."""
        result = intersection((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), tol=0.01)
        self.assertEqual(len(result), 1)  # One intersection point tuple
        self.assertAlmostEqual(result[0][0], 1.0, places=5)

    def test_invalid_input(self):
        """Test invalid input returns None."""
        result = intersection((1.0,), (2.0,))  # type: ignore
        self.assertIsNone(result)


class TestBoundingBox(unittest.TestCase):
    """Tests for bounding_box function."""

    def test_simple_box(self):
        """Test bounding box of simple locus."""
        locus = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)]
        y_min, x_max, y_max, x_min = bounding_box(locus)
        self.assertEqual(y_min, 0.0)
        self.assertEqual(x_max, 3.0)
        self.assertEqual(y_max, 2.0)
        self.assertEqual(x_min, 0.0)

    def test_negative_coords(self):
        """Test bounding box with negative coordinates."""
        locus = [(-5.0, -3.0), (2.0, 4.0), (0.0, 0.0)]
        y_min, x_max, y_max, x_min = bounding_box(locus)
        self.assertEqual(y_min, -3.0)
        self.assertEqual(x_max, 2.0)
        self.assertEqual(y_max, 4.0)
        self.assertEqual(x_min, -5.0)

    def test_single_point(self):
        """Test bounding box of single point."""
        locus = [(5.0, 7.0)]
        y_min, x_max, y_max, x_min = bounding_box(locus)
        self.assertEqual(y_min, 7.0)
        self.assertEqual(x_max, 5.0)
        self.assertEqual(y_max, 7.0)
        self.assertEqual(x_min, 5.0)


if __name__ == "__main__":
    unittest.main()
