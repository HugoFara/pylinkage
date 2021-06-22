#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:32:37 2021.

@author: HugoFara
"""

import pylinkage.linkage as pl
import pylinkage.visualizer as visu
import pylinkage.optimizer as opti

# Static points in space, belonging to the frame
frame_first = pl.Static(0, 0, name="A")
frame_second = pl.Static(3, 0, name="D")
# Main motor
crank = pl.Crank(0, 1, joint0=frame_first, angle=0.31, distance=1, name="B")
# Close the loop
pin = pl.Pivot(3, 2, joint0=crank, joint1=frame_second,
               distance0=3, distance1=1, name="C")

# Linkage definition
my_linkage = pl.Linkage(
    joints=(frame_first, frame_second, crank, pin),
    order=(frame_first, frame_second, crank, pin),
    name="My four-bar linkage")

# Visualization
visu.show_linkage(my_linkage)


def fitness_func(linkage, params):
    """Return some stuff."""
    linkage.set_constraints(*params)
    return 0


# Exhaustive optimization
opti.exhaustive_optimization(eval_func=fitness_func, linkage=my_linkage)
