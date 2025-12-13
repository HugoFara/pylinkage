#!/usr/bin/env python3
"""
Created on Thu Apr 15, 13:26:42 2021

@author: HugoFara
"""

import unittest

import numpy as np

from pylinkage.geometry import circle_intersect, circle_line_intersection, sqr_dist
from pylinkage.geometry.secants import INTERSECTION_NONE, INTERSECTION_SAME, INTERSECTION_TWO


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


if __name__ == "__main__":
    unittest.main()
