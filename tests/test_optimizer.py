import unittest
import numpy as np

from pylinkage import linkage as pl
from pylinkage import optimizer as opti
from pylinkage.utility import kinematic_minimization


def prepare_linkage():
    """Return a simple four-bar linkage."""
    # Static points in space, belonging to the frame
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    # Close the loop
    pin = pl.Pivot(
        0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1
    )
    # Linkage definition
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def fitness_func(loci, **kwargs):
    """
    Return if the tip can go to the point (3, 1).

    Notes
    -----
    It is a minimization problem.
    The best possible score is 0.
    The worst score is float('inf').
    """
    # Locus of the Joint "pin"
    tip_locus = tuple(x[0] for x in loci)[0]
    return (tip_locus[0] - 3) ** 2 + tip_locus[1] ** 2


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
        self.assertAlmostEqual(score, 1, delta=.1)


class TestVariator(unittest.TestCase):

    def test_length(self):
        """Test that the length is about the number of subdivisions."""
        sequence = [1]
        bounds = ([0], [3])
        for divisions in (1, 2, 3, 5, 6, 13, 30):
            length = sum(
                1 for _ in opti.sequential_variator(sequence, divisions, bounds)
            )
            self.assertAlmostEqual(length, divisions, delta=1)
            length = sum(
                1 for _ in opti.fast_variator(divisions, bounds)
            )
            self.assertAlmostEqual(length, divisions)


class TestTrialsAndErrors(unittest.TestCase):
    linkage = prepare_linkage()

    def test_convergence(self):
        """Test if the output after some iterations is improved."""
        bounds = opti.generate_bounds(self.linkage.get_num_constraints(), 2, 2)
        score, dimensions, coord = opti.trials_and_errors_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            divisions=10,
            bounds=bounds,
            n_results=10,
            order_relation=min,
            verbose=False,
        )[0]
        self.assertAlmostEqual(score, 0.0, delta=0.3)


class TestPSO(unittest.TestCase):
    """Test the particle swarm optimization."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_convergence(self):
        """Test if the result is not too far from 0.0."""
        dim = len(self.constraints)
        bounds = (np.zeros(dim), np.ones(dim) * 5)
        score, dimensions, coord = opti.particle_swarm_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            bounds=bounds,
            n_particles=20,
            iters=50,
            order_relation=min,
            verbose=False,
        )[0]
        # Do not apply optimization problems
        self.assertAlmostEqual(score, 0.0, delta=.3)


if __name__ == '__main__':
    unittest.main()
