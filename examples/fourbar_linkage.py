#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The fourbar_linkage module demonstrates the features of pylinkage.

It is not intended to be imported in another project,
but feel welcome to copy-paste chunks of code.

Created on Sat Jun 19, 12:32:37 2021.

@author: HugoFara
"""
import pylinkage as pl


def define_linkage():
    """Define a simple four-bar linkage.

    :return: A demo four-bar linkage.
    :rtype: pylinkage.Linkage
    """
    # Main motor
    crank = pl.Crank(
        0, 1,
        joint0=(0, 0),  # Fixed to a single point in space
        angle=0.31, distance=1,
        name="B"
    )
    # Close the loop
    pin = pl.Revolute(
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
    return my_linkage


@pl.kinematic_minimization
def quadrant_fitness(loci, **kwargs):
    """Return how fit the locus is to describe a quarter of circle.
    
    It is a minimization problem and the theoretical best score is 0.

    :param loci: Successive positions of joints
    :type loci: tuple[tuple[tuple[float, float]]]
    :param **kwargs:
    :return: Sum of square distances between tip locus bounding box and a defined
        square.
    :rtype: float

    
    """
    # Locus of the Joint "pin" must in linkage order
    tip_locus = list(x[-1] for x in loci)
    # We get the bounding box
    curr_bb = pl.bounding_box(tip_locus)
    # Reference bounding box in order (min_y, max_x, max_y, min_x)
    ref_bb = (0, 5, 2, 3)
    # Our score is the square sum of the edge distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))


def main():
    """Define and optimize a demo linkage."""
    my_linkage = define_linkage()
    # Visualization
    pl.show_linkage(my_linkage)

    # Optimization part

    # We save the initial position because we don't want a completely different movement
    init_pos = my_linkage.get_coords()

    constraints = tuple(my_linkage.get_num_constraints())

    print(
        "Score before optimization:",
        quadrant_fitness(my_linkage, constraints, init_pos)
    )

    # Trials and errors optimization as an example ONLY

    score, position, coord = pl.trials_and_errors_optimization(
        eval_func=quadrant_fitness,
        linkage=my_linkage,
        divisions=25,
        order_relation=min,
    )[0]

    print("Score after trials and errors optimization", score)
    # We reinitialize the linkage (an optimal linkage is not interesting)
    my_linkage.set_num_constraints(constraints)
    # As we do for initial positions
    my_linkage.set_coords(init_pos)

    # Particle Swarm Optimization

    # Optimization is more efficient with a start space
    bounds = pl.generate_bounds(my_linkage.get_num_constraints())

    score, position, _coord = pl.particle_swarm_optimization(
        eval_func=quadrant_fitness,
        linkage=my_linkage,
        bounds=bounds,
        order_relation=min,
    )[0]

    print("Score after particle swarm optimization:", score)

    # Visualization of the optimized linkage
    my_linkage.set_num_constraints(position)
    pl.show_linkage(my_linkage)


if __name__ == "__main__":
    main()
