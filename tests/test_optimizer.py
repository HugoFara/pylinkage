import unittest
import numpy as np

from pylinkage.geometry import bounding_box
from pylinkage.exceptions import UnbuildableError
from pylinkage import linkage as pl
from pylinkage import optimizer as opti
from pylinkage.utility import kinematic_minimization


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
    linkage = pl.Linkage(
        joints=(crank, pin),
        order=(crank, pin),
    )

    global init_pos
    init_pos = tuple(linkage.get_coords())
    return linkage

@kinematic_minimization
def fitness_func(loci, **kwargs):
    """
    Return how fit the locus is to describe a quarter of circle.

    It is a minimization problem.
    Theorical best score is 0.
    Worst score is float('inf').
    """
    # Locus of the Joint 'pin", mast in linkage order
    tip_locus = tuple(x[-1] for x in loci)
    # We get the bounding box
    curr_bb = bounding_box(tip_locus)
    # We set the reference bounding box
    ref_bb = (0, 5, 2, 3)
    # Our score is the square sum of the edges distances
    return sum((pos - ref_pos) ** 2 for pos, ref_pos in zip(curr_bb, ref_bb))


class TestGenerateBounds(unittest.TestCase):
    """Test various things about the generate_bounds function."""

    def test_function(self):
        """Test is the function runs simply."""
        center = [1, 2, 3]
        bounds = opti.generate_bounds(center=center, min_ratio=2, max_factor=2)
        self.assertTrue(np.all(bounds[0] == [.5, 1, 1.5]))
        self.assertTrue(np.all(bounds[1] == [2, 4, 6]))


class TestEvaluation(unittest.TestCase):
    """Test if a linkage can properly be evaluated."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_score(self):
        """Test if score is well returned."""
        score = fitness_func(self.linkage, self.constraints)
        self.assertAlmostEqual(score, 3, delta=.1)


class TestVariator(unittest.TestCase):

    def test_length(self):
        """Test that the length is about the number of subdivisions."""
        sequence = [1]
        bounds = ([0], [3])
        for divisions in (1, 2, 3, 5, 6, 13, 30):
            length = sum(1 for x in opti.variator(sequence, divisions, bounds))
            self.assertAlmostEqual(length, divisions, delta=1)


class TestTrialsAndErros(unittest.TestCase):
    linkage = prepare_linkage()

    def test_convergence(self):
        """Test if the output after some iterations is improved."""
        """print(opti.trials_and_errors_optimization(
            eval_func=lambda *x: -fitness_func(*x),
            linkage=self.linkage,
            divisions=5000,
            n_results=10,
        ))"""
        bounds = opti.generate_bounds(self.linkage.get_num_constraints(), 2, 2)
        score, dimensions, coord = opti.trials_and_errors_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            divisions=20,
            bounds=bounds,
            n_results=40,
            order_relation=min,
        )[0]
        self.assertAlmostEqual(score, 0.0, delta=0.3)


class TestPSO(unittest.TestCase):
    """Test the particle swarm optimization."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_convergence(self):
        """Test if the result is not to far from 0.0."""
        dim = len(self.constraints)
        bounds = (np.zeros(dim), np.ones(dim) * 5)
        score, dimensions, coord = opti.particle_swarm_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            bounds=bounds,
            n_particles=60,
            iters=50,
            order_relation=min,
        )[0]
        # Do not apply optimization problems
        self.assertAlmostEqual(score, 0.0, delta=1)


if __name__ == '__main__':
    unittest.main()
