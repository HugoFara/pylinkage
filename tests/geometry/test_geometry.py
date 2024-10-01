#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15, 13:26:42 2021

@author: HugoFara
"""

import unittest
import numpy as np

from pylinkage.geometry import circle_intersect, circle_line_intersection, sqr_dist


class TestCircles(unittest.TestCase):
    """Tests for circles."""

    def test_no_inter(self):
        """Test two circles not intersecting."""
        c1 = 2., 5, 1
        c2 = 2, 5., 1.0
        inter = circle_intersect(c1, c2)
        self.assertEqual(2, len(inter))
        self.assertEqual(3, inter[0])

    def test_radius_ineq(self):
        """Test two concentric circles of different radi."""
        c1 = 2., 5, 1
        c2 = 2, 5., 1.4
        inter = circle_intersect(c1, c2)
        self.assertEqual(0, inter[0])


def circle_line_intersection_data(mode):
    """
    Prepare the data for testing.

    Arguments
    ---------
    mode : int
        Desired number of intersections between the circle and the line.

    Returns
    -------
    ((float, float, float), (float, float, float))
        A circle and a line crossing {mode} times
    """
    circle = np.random.rand(3) * 10
    circle[2] = abs(circle[2])
    circle = (0, 0, 2)
    # Put the first point on the circle
    point = np.empty(2)
    point[0] = circle[0] + (np.random.rand() * 2 - 1) * circle[2] * .9
    point[1] = np.sqrt(circle[2] ** 2 - (point[0] - circle[0]) ** 2) + circle[1]
    # Build a tangent line
    line = np.empty(3)
    vector = circle[:2] - point
    line[:2] = vector
    # Line is tangent
    line[2] = -np.dot(line[:2], point)
    if mode == 0:
        line[2] += np.dot(
            line[:2],
            vector * circle[2] * (1 + np.random.rand())
        )
    if mode == 2:
        line[2] -= np.dot(
            line[:2],
            vector * circle[2] * np.random.rand()
        )
    return circle, tuple(map(float, line))


class TestCircleLineIntersection(unittest.TestCase):
    """Various tests on straight line intersections."""

    def test_no_crossing(self):
        """Test a line and a circle not crossing each other."""
        circle, line = circle_line_intersection_data(0)
        intersection = circle_line_intersection(circle, tuple(line))
        self.assertEqual(
            0, len(intersection), f'Intersections: {intersection}:'
        )

    def test_tangent(self):
        """
        Test a straight line tangent to a circle.

        This is a dangerous test, so we only want a success rate over 90%.
        """
        count = 0
        for _ in range(20):
            circle, line = circle_line_intersection_data(1)
            intersection = circle_line_intersection(circle, line)
            if len(intersection) == 1:
                count += 1
            elif len(intersection) == 2 and 1e-5 > sqr_dist(intersection[0], intersection[1]):
                count += 1
        self.assertLessEqual(
            7, count, f"Success rate of {count * 5}% is too low!"
        )

    def test_crossing(self):
        """Test a straight line crossing a circle twice."""
        circle, line = circle_line_intersection_data(2)
        intersection = circle_line_intersection(circle, tuple(line))
        self.assertEqual(
            2, len(intersection), f'Intersection: {intersection}:'
        )


if __name__ == '__main__':
    unittest.main()
