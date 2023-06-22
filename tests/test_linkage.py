#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:19:47 2021

@author: HugoFara
"""

import unittest
from math import sqrt
from pylinkage.exceptions import UnbuildableError
from pylinkage.linkage import Pivot, Fixed


class TestPivot(unittest.TestCase):
    """Test Pivot Joint."""

    pivot1 = Pivot(0, 0)

    def test_buildable(self):
        """Upper intersect test."""
        pivot2 = Pivot(1, 0)
        pivot3 = Pivot(y=1, joint0=self.pivot1, joint1=pivot2,
                       distance0=1, distance1=1)
        pivot3.reload()
        self.assertEqual(pivot3.coord(), (.5, sqrt(.75)))

    def test_under_intersect(self):
        """Under intersect test."""
        pivot2 = Pivot(1, 0)
        pivot3 = Pivot(y=-1, joint0=self.pivot1, joint1=pivot2,
                       distance0=1, distance1=1)
        pivot3.reload()
        self.assertEqual(pivot3.coord(), (.5, -sqrt(.75)))

    def test_limit_intersect(self):
        """Test system almost breaking."""
        pivot2 = Pivot(2, 0)
        pivot3 = Pivot(y=1, joint0=self.pivot1, joint1=pivot2,
                       distance0=1, distance1=1)
        pivot3.reload()
        self.assertEqual(pivot3.coord(), (1, 0))

    def test_no_intersect(self):
        """Test system almost breaking."""
        pivot2 = Pivot(0, 3)
        pivot3 = Pivot(y=1, joint0=self.pivot1, joint1=pivot2,
                       distance0=1, distance1=1)
        with self.assertRaises(UnbuildableError):
            pivot3.reload()


class TestFixed(unittest.TestCase):
    """Test Fixed_Joint."""

    pivot1 = Pivot(0, 0)

    def test_pos(self):
        """Test Fixed_Joint positioning."""
        pivot2 = Pivot(1, 0)
        fixed = Fixed(joint0=self.pivot1, joint1=pivot2, angle=0, distance=1)
        fixed.reload()
        self.assertEqual(fixed.coord(), (1, 0))


if __name__ == '__main__':
    unittest.main()
