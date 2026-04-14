#!/usr/bin/env python3
"""
The fourbar_linkage module demonstrates the features of pylinkage.

It is not intended to be imported in another project, but feel
welcome to copy-paste chunks of code.

Created on Sat Jun 19, 12:32:37 2021.

@author: HugoFara
"""

import pylinkage as pl
from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage


def define_linkage() -> Linkage:
    """Define a simple four-bar linkage using the component API."""
    # Ground anchors
    A = Ground(0.0, 0.0, name="A")
    D = Ground(3.0, 0.0, name="D")

    # Driver crank
    crank = Crank(
        anchor=A,
        radius=1.0,
        angular_velocity=0.31,
        name="B",
    )

    # Close the loop with an RRR dyad (coupler + rocker)
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=D,
        distance1=3.0,
        distance2=1.0,
        name="C",
    )

    return Linkage([A, D, crank, pin], name="My four-bar linkage")


@pl.kinematic_minimization
def quadrant_fitness(loci, **kwargs):
    """Return how fit the locus is to describe a quarter of circle.

    It is a minimization problem and the theoretical best score is 0.

    :param loci: Successive positions of joints.
    :param kwargs: Extra kwargs (unused).
    :return: Sum of square distances between tip locus bounding box
        and a defined square.
    """
    # Locus of the coupler point (last component in the linkage)
    tip_locus = [x[-1] for x in loci]
    # Bounding box
    curr_bb = pl.bounding_box(tip_locus)
    # Reference bounding box in order (min_y, max_x, max_y, min_x)
    ref_bb = (0, 5, 2, 3)
    # Score is the square sum of the edge distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb, strict=False))


def main() -> None:
    """Define and optimize a demo linkage."""
    my_linkage = define_linkage()
    pl.show_linkage(my_linkage)

    # Save the initial position; we don't want a radically different motion.
    init_pos = my_linkage.get_coords()
    constraints = tuple(my_linkage.get_constraints())

    print("Score before optimization:", quadrant_fitness(my_linkage, constraints, init_pos))

    # Trials and errors optimization as an example ONLY.
    result = pl.trials_and_errors_optimization(
        eval_func=quadrant_fitness,
        linkage=my_linkage,
        divisions=25,
        order_relation=min,
    )[0]
    print("Score after trials and errors optimization", result.score)

    # Reset to the original mechanism.
    my_linkage.set_constraints(list(constraints))
    my_linkage.set_coords(init_pos)

    # Particle Swarm Optimization.
    bounds = pl.generate_bounds(my_linkage.get_constraints())
    result = pl.particle_swarm_optimization(
        eval_func=quadrant_fitness,
        linkage=my_linkage,
        bounds=bounds,
        order_relation=min,
    )[0]
    print("Score after particle swarm optimization:", result.score)

    # Visualize the optimized linkage.
    my_linkage.set_constraints(list(result.dimensions))
    pl.show_linkage(my_linkage)


if __name__ == "__main__":
    main()
