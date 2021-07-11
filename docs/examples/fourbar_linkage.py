#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:32:37 2021.

@author: HugoFara
"""

import numpy as np

import pylinkage as pl

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
    joints=(crank, pin),
    order=(crank, pin),
    name="My four-bar linkage"
)

# Visualization
pl.show_linkage(my_linkage)

# Optimization part

# We save initial position because we don't want a completely different movement
init_pos = my_linkage.get_coords()

def fitness_func(linkage, params, *args):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minisation problem and the theorical best score is 0.
    """
    linkage.set_coords(init_pos)
    linkage.set_num_constraints(params)
    try:
        points = 12
        n = linkage.get_rotation_period()
        # Complete revolution with 12 points
        tuple(
            tuple(i) for i in linkage.step(
                iterations=points + 1, dt=n / points
            )
        )
        # Again with n points, and at least 12 iterations
        n = 96
        factor = int(points / n) + 1
        loci = tuple(
            tuple(i) for i in linkage.step(
                iterations=n * factor, dt=1 / factor
            )
        )
    except pl.UnbuildableError:
        return float('inf')
    else:
        # Locus of the Joint 'pin", mast in linkage order
        tip_locus = tuple(x[-1] for x in loci)
        # We get the bounding box
        curr_bb = pl.bounding_box(tip_locus)
        # We set the reference bounding box with frame_second as down-left
        # corner and size 2
        ref_bb = (frame_second.y, frame_second.x + 2,
                  frame_second.y + 2, frame_second.x)
        # Our score is the square sum of the edges distances
        return sum(
            (pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb)
        )

constraints = tuple(my_linkage.get_num_constraints())

print(
    "Score before optimization: {}".format(
        fitness_func(my_linkage, constraints)
    )
)

# Trials and errors optimization as an example ONLY
score, position, coord = pl.trials_and_errors_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    divisions=25,
    order_relation=min,
)[0]

print("Score after trials and errors optimization: {}".format(score))

# We reinitialize the linkage (an optimal linkage is not interesting)
my_linkage.set_num_constraints(constraints)
# As we do for initial positions
my_linkage.set_coords(init_pos)

# Particle Swarm Optimization


# Optimization is more efficient with a start space
bounds = pl.generate_bounds(my_linkage.get_num_constraints())

score, position, coord = pl.particle_swarm_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    bounds=bounds,
    order_relation=min,
)[0]

print("Score after particle swarm optimization: {}".format(score))

# Visualization of the optimized linkage
my_linkage.set_num_constraints(position)
pl.show_linkage(my_linkage)
