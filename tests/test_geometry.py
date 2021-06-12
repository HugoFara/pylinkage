#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:26:42 2021

@author: HugoFara
"""

import unittest
import lib.geometry as geo


class TestCircles(unittest.TestCase):

    def test_no_inter(self):
        c1 = (2., 5, 1)
        c2 = (2, 5., 1.0)
        inter = geo.circle_intersect(c1, c2)
        self.assertEqual(len(inter), 2)
        self.assertEqual(inter[0], 3)

    def test_radius_ineq(self):
        c1 = (2., 5, 1)
        c2 = (2, 5., 1.4)
        inter = geo.circle_intersect(c1, c2)
        self.assertEqual(len(inter), 1)
        self.assertEqual(inter[0], 0)

if __name__ == '__main__':
    unittest.main()