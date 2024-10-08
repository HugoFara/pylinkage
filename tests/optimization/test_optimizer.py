import unittest
import numpy as np

import pylinkage as pl
from pylinkage import optimization
from pylinkage.optimization.grid_search import fast_variator, sequential_variator
from pylinkage.optimization.utils import kinematic_minimization


def prepare_linkage():
    """Simple four-bar linkage.

    :returns: A four-bar linkage
    :rtype: pylinkage.Linkage
    """
    # Static points in space, belonging to the frame
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    # Close the loop
    pin = pl.Revolute(
        0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1
    )
    # Linkage definition
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def fitness_func(loci, **kwargs):
    """Return if the tip can go to the point (3, 1).

    :param loci:
    :param kwargs:

    .. notes
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
        bounds = optimization.generate_bounds(center=center, min_ratio=2, max_factor=2)
        self.assertTrue(np.all(bounds[0] == [.5, 1, 1.5]))
        self.assertTrue(np.all(bounds[1] == [2, 4, 6]))


class TestEvaluation(unittest.TestCase):
    """Test if a linkage can properly be evaluated."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_score(self):
        """Test if score is well returned."""
        score = fitness_func(self.linkage, self.constraints)
        self.assertAlmostEqual(1, score, delta=.1)


class TestVariator(unittest.TestCase):
    """Test case for the variation creators."""

    def test_length(self):
        """Test that the length is about the number of subdivisions."""
        sequence = [1]
        bounds = ((0, ), (3, ))
        for divisions in (1, 2, 3, 5, 6, 13, 30):
            length = sum(
                1 for _ in sequential_variator(sequence, divisions, bounds)
            )
            self.assertAlmostEqual(length, divisions, delta=1)
            length = sum(
                1 for _ in fast_variator(divisions, bounds)
            )
            self.assertAlmostEqual(length, divisions)


class TestTrialsAndErrors(unittest.TestCase):
    """Tests for the trials and errors optimization."""
    linkage = prepare_linkage()

    def test_convergence(self):
        """Test if the output after some iterations is improved."""
        bounds = optimization.generate_bounds(self.linkage.get_num_constraints(), 2, 2)
        score, dimensions, coord = optimization.trials_and_errors_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            divisions=10,
            bounds=bounds,
            n_results=10,
            order_relation=min,
            verbose=False,
        )[0]
        self.assertAlmostEqual(0.0, score, delta=0.3)


class TestPSO(unittest.TestCase):
    """Test the particle swarm optimization."""
    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_convergence(self):
        """Test if the result is not too far from 0.0."""
        delta = 0.3
        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5
        opti_kwargs = {
            "eval_func": fitness_func,
            "linkage": self.linkage,
            "bounds": bounds,
            "n_particles": 20,
            "iters": 30,
            "order_relation": min,
            "verbose": False
        }
        score, dimensions, coord = optimization.particle_swarm_optimization(**opti_kwargs)[0]
        if score > delta:
            # Try again with more agents
            opti_kwargs.update({
                "n_particles": 50,
                "iters": 100,
            })
            score, dimensions, coord = optimization.particle_swarm_optimization(**opti_kwargs)[0]
        # Do not apply optimization problems
        self.assertAlmostEqual(0.0, score, delta=delta)


if __name__ == '__main__':
    unittest.main()
