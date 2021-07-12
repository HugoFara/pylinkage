#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The fourbar_linkage module demonstrats the functionnalities of pylinkage.

It is not intended to be imported in another project, but fell welcome to
copy-paste chunks of code.

Created on Sat Jun 19 12:32:37 2021.

@author: HugoFara
"""

import numpy as np

import pylinkage as pl

# Main motor
crank = pl.Crank(
    0, 1,
    joint0=(0, 0), # Fixed to a single point in space
    angle=0.31, distance=1,
    name="B"
)
# Close the loop
pin = pl.Pivot(
    3, 2,
    joint0=crank, joint1=(3, 0),
    distance0=3, distance1=1, name="C"
)

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


@pl.kinematic_minimization
def fitness_func(loci, **kwargs):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minisation problem and the theorical best score is 0.

    Returns
    -------
    float
        Sum of square distances between tip locus bounding box and a defined
        square.
    """
    # Locus of the Joint 'pin", mast in linkage order
    tip_locus = tuple(x[-1] for x in loci)
    # We get the bounding box
    curr_bb = pl.bounding_box(tip_locus)
    # Reference bounding box in order (min_y, max_x, max_y, min_x)
    ref_bb = (0, 5, 2, 3)
    # Our score is the square sum of the edges distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))


constraints = tuple(my_linkage.get_num_constraints())

print(
    "Score before optimization: {}".format(
        fitness_func(my_linkage, constraints, init_pos)
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
