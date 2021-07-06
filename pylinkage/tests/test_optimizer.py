import unittest
import numpy as np

from ..geometry import bounding_box
from ..exceptions import UnbuildableError
from .. import linkage as pl
from .. import optimizer as opti

def prepare_linkage():
    """Return a simple four-bar linkage."""
    # Static points in space, belonging to the frame
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    # Main motor
    crank = pl.Crank(0, 1, joint0=frame_first, angle=0.31, distance=1)
    # Close the loop
    pin = pl.Pivot(3, 2, joint0=crank, joint1=frame_second,
                   distance0=3, distance1=1)

    # Linkage definition
    linkage =  pl.Linkage(
        joints=(crank, pin),
        order=(crank, pin),
    )

    global init_pos
    init_pos = tuple(linkage.get_coords())
    return linkage

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
        tuple(tuple(i) for i in linkage.step(iterations=points + 1,
                                             dt=n/points))
        # Again with n points, and at least 12 iterations
        n = 96
        factor = int(points / n) + 1
        L = tuple(tuple(i) for i in linkage.step(
            iterations=n * factor, dt=1 / factor))
    except UnbuildableError:
        return -float('inf')
    else:
        # Locus of the Joint 'pin", mast in linkage order
        tip_locus = tuple(x[-1] for x in L)
        # We get the bounding box
        curr_bb = bounding_box(tip_locus)
        # We set the reference bounding box with frame_second as down-left
        # corner and size 2
        parent = linkage.joints[1].joint1
        ref_bb = (parent.y, parent.x + 2, parent.y + 2, parent.x)
        # Our score is the square sum of the edges distances
        return -sum((pos - ref_pos) ** 2
                    for pos, ref_pos in zip(curr_bb, ref_bb))


class TestEvaluation(unittest.TestCase):
    """Test if a linkage can properly be evaluated."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_score(self):
        """Test if score is well returned."""
        score = fitness_func(self.linkage, self.constraints)
        self.assertAlmostEqual(score, -3, delta=.1)


class TestTrialsAndErros(unittest.TestCase):
    linkage = prepare_linkage()

    def test_convergence(self):
        """Test if the output after some iterations is improved."""
        score, position, coord = opti.trials_and_errors_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            delta_dim=.1,
            n_results=1,
        )[0]
        self.assertAlmostEqual(score, 0.0, delta=0.3)


class TestPSO(unittest.TestCase):
    """Test the particle swarm optimization."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def __PSO_fitness_wrapper__(self, constraints, *args):
        """A simple wrapper to make the fitness function compatible."""
        return fitness_func(self.linkage, constraints, *args)

    def test_convergence(self):
        """Test if the result is not to far from 0.0."""
        dim = len(self.constraints)
        bounds = (np.zeros(dim), np.ones(dim) * 5)
        score = opti.particle_swarm_optimization(
            eval_func=self.__PSO_fitness_wrapper__,
            linkage=self.linkage,
            bounds=bounds,
            n_particles=50,
        ).swarm.best_cost
        self.assertAlmostEqual(score, 0.0, delta=1)


if __name__ == '__main__':
    unittest.main()
