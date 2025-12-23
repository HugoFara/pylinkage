"""Tests for multi-objective optimization."""

import unittest

import numpy as np

import pylinkage as pl
from pylinkage import optimization
from pylinkage.exceptions import OptimizationError
from pylinkage.optimization.collections import ParetoFront, ParetoSolution
from pylinkage.optimization.utils import kinematic_minimization

# Check if pymoo is available
try:
    import pymoo  # noqa: F401

    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


def prepare_linkage():
    """Simple four-bar linkage for testing.

    Returns:
        A four-bar linkage.
    """
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    pin = pl.Revolute(
        0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1
    )
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def objective1(loci, **kwargs):
    """First objective: distance to point (3, 0)."""
    tip_locus = tuple(x[0] for x in loci)[0]
    return (tip_locus[0] - 3) ** 2 + tip_locus[1] ** 2


@kinematic_minimization
def objective2(loci, **kwargs):
    """Second objective: distance to point (0, 3)."""
    tip_locus = tuple(x[0] for x in loci)[0]
    return tip_locus[0] ** 2 + (tip_locus[1] - 3) ** 2


class TestParetoSolution(unittest.TestCase):
    """Tests for ParetoSolution class."""

    def test_creation(self):
        """Test creating a ParetoSolution."""
        sol = ParetoSolution(
            scores=(1.0, 2.0),
            dimensions=np.array([1.0, 2.0, 3.0]),
            init_positions=((0.0, 0.0), (1.0, 1.0)),
        )
        self.assertEqual(sol.scores, (1.0, 2.0))
        self.assertEqual(len(sol.dimensions), 3)

    def test_dominates(self):
        """Test Pareto dominance."""
        sol1 = ParetoSolution(
            scores=(1.0, 2.0),
            dimensions=np.array([1.0]),
            init_positions=(),
        )
        sol2 = ParetoSolution(
            scores=(2.0, 3.0),
            dimensions=np.array([1.0]),
            init_positions=(),
        )
        sol3 = ParetoSolution(
            scores=(0.5, 3.0),
            dimensions=np.array([1.0]),
            init_positions=(),
        )

        # sol1 dominates sol2 (better in both)
        self.assertTrue(sol1.dominates(sol2))
        # sol2 does not dominate sol1
        self.assertFalse(sol2.dominates(sol1))
        # sol1 and sol3 don't dominate each other (non-dominated)
        self.assertFalse(sol1.dominates(sol3))
        self.assertFalse(sol3.dominates(sol1))


class TestParetoFront(unittest.TestCase):
    """Tests for ParetoFront class."""

    def setUp(self):
        """Create sample Pareto front."""
        self.solutions = [
            ParetoSolution(
                scores=(1.0, 4.0),
                dimensions=np.array([1.0]),
                init_positions=(),
            ),
            ParetoSolution(
                scores=(2.0, 2.0),
                dimensions=np.array([2.0]),
                init_positions=(),
            ),
            ParetoSolution(
                scores=(4.0, 1.0),
                dimensions=np.array([3.0]),
                init_positions=(),
            ),
        ]
        self.front = ParetoFront(
            solutions=self.solutions,
            objective_names=("Obj1", "Obj2"),
        )

    def test_len(self):
        """Test length of Pareto front."""
        self.assertEqual(len(self.front), 3)

    def test_iter(self):
        """Test iteration over solutions."""
        sols = list(self.front)
        self.assertEqual(len(sols), 3)

    def test_getitem(self):
        """Test indexing solutions."""
        self.assertEqual(self.front[0].scores, (1.0, 4.0))
        self.assertEqual(self.front[2].scores, (4.0, 1.0))

    def test_n_objectives(self):
        """Test number of objectives."""
        self.assertEqual(self.front.n_objectives, 2)

    def test_scores_array(self):
        """Test scores as numpy array."""
        arr = self.front.scores_array()
        self.assertEqual(arr.shape, (3, 2))
        self.assertAlmostEqual(arr[0, 0], 1.0)
        self.assertAlmostEqual(arr[2, 1], 1.0)

    def test_filter(self):
        """Test filtering to fewer solutions."""
        filtered = self.front.filter(2)
        self.assertEqual(len(filtered), 2)
        # Should keep objective names
        self.assertEqual(filtered.objective_names, ("Obj1", "Obj2"))

    def test_filter_larger_than_front(self):
        """Test filtering with max larger than front size."""
        filtered = self.front.filter(10)
        self.assertEqual(len(filtered), 3)

    def test_best_compromise_equal_weights(self):
        """Test best compromise with equal weights."""
        best = self.front.best_compromise()
        # Middle solution (2, 2) should be best with equal weights
        self.assertEqual(best.scores, (2.0, 2.0))

    def test_best_compromise_custom_weights(self):
        """Test best compromise with custom weights."""
        # Weight objective 1 heavily
        best = self.front.best_compromise(weights=[0.9, 0.1])
        # First solution (1, 4) should be best when obj1 matters most
        self.assertEqual(best.scores, (1.0, 4.0))

    def test_best_compromise_empty_raises(self):
        """Test that empty front raises ValueError."""
        empty = ParetoFront(solutions=[], objective_names=())
        with self.assertRaises(ValueError):
            empty.best_compromise()

    @unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
    def test_hypervolume(self):
        """Test hypervolume computation."""
        hv = self.front.hypervolume(reference_point=[5.0, 5.0])
        self.assertGreater(hv, 0)

    def test_plot_empty_raises(self):
        """Test that plotting empty front raises ValueError."""
        empty = ParetoFront(solutions=[], objective_names=())
        with self.assertRaises(ValueError):
            empty.plot()

    def test_plot_2d(self):
        """Test 2D plotting."""
        import matplotlib.pyplot as plt

        fig = self.front.plot()
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_invalid_indices(self):
        """Test that invalid objective indices raise ValueError."""
        with self.assertRaises(ValueError):
            self.front.plot(objective_indices=(0, 5))


@unittest.skipUnless(PYMOO_AVAILABLE, "pymoo not installed")
class TestMultiObjectiveOptimization(unittest.TestCase):
    """Tests for multi_objective_optimization function."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_basic_optimization(self):
        """Test basic multi-objective optimization."""
        from pylinkage.optimization import multi_objective_optimization

        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        pareto = multi_objective_optimization(
            objectives=[objective1, objective2],
            linkage=self.linkage,
            bounds=bounds,
            n_generations=10,
            pop_size=20,
            verbose=False,
        )

        self.assertIsInstance(pareto, ParetoFront)
        self.assertGreater(len(pareto), 0)
        self.assertEqual(pareto.n_objectives, 2)

    def test_returns_pareto_front(self):
        """Test that result is a ParetoFront."""
        from pylinkage.optimization import multi_objective_optimization

        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        pareto = multi_objective_optimization(
            objectives=[objective1, objective2],
            linkage=self.linkage,
            bounds=bounds,
            n_generations=5,
            pop_size=10,
            verbose=False,
        )

        self.assertIsInstance(pareto, ParetoFront)
        for sol in pareto:
            self.assertIsInstance(sol, ParetoSolution)
            self.assertEqual(len(sol.scores), 2)

    def test_auto_bounds(self):
        """Test that bounds are auto-generated when not provided."""
        from pylinkage.optimization import multi_objective_optimization

        pareto = multi_objective_optimization(
            objectives=[objective1, objective2],
            linkage=self.linkage,
            bounds=None,
            n_generations=5,
            pop_size=10,
            verbose=False,
        )

        self.assertIsInstance(pareto, ParetoFront)

    def test_objective_names(self):
        """Test that objective names are stored."""
        from pylinkage.optimization import multi_objective_optimization

        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        pareto = multi_objective_optimization(
            objectives=[objective1, objective2],
            linkage=self.linkage,
            bounds=bounds,
            objective_names=["Path Error", "Transmission"],
            n_generations=5,
            pop_size=10,
            verbose=False,
        )

        self.assertEqual(pareto.objective_names, ("Path Error", "Transmission"))

    def test_nsga3(self):
        """Test NSGA-III algorithm."""
        from pylinkage.optimization import multi_objective_optimization

        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        pareto = multi_objective_optimization(
            objectives=[objective1, objective2],
            linkage=self.linkage,
            bounds=bounds,
            algorithm="nsga3",
            n_generations=5,
            pop_size=20,
            verbose=False,
        )

        self.assertIsInstance(pareto, ParetoFront)

    def test_invalid_algorithm_raises(self):
        """Test that invalid algorithm raises OptimizationError."""
        from pylinkage.optimization import multi_objective_optimization

        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        with self.assertRaises(OptimizationError):
            multi_objective_optimization(
                objectives=[objective1],
                linkage=self.linkage,
                bounds=bounds,
                algorithm="invalid",
                verbose=False,
            )

    def test_invalid_n_generations_raises(self):
        """Test that invalid n_generations raises OptimizationError."""
        from pylinkage.optimization import multi_objective_optimization

        with self.assertRaises(OptimizationError):
            multi_objective_optimization(
                objectives=[objective1],
                linkage=self.linkage,
                n_generations=-1,
                verbose=False,
            )

    def test_invalid_bounds_raises(self):
        """Test that invalid bounds raise OptimizationError."""
        from pylinkage.optimization import multi_objective_optimization

        with self.assertRaises(OptimizationError):
            multi_objective_optimization(
                objectives=[objective1],
                linkage=self.linkage,
                bounds=([0], [1, 2]),  # Mismatched dimensions
                verbose=False,
            )


class TestModuleExports(unittest.TestCase):
    """Test that multi-objective functions are properly exported."""

    def test_optimization_module_exports(self):
        """Test functions are accessible from optimization module."""
        self.assertTrue(hasattr(optimization, "multi_objective_optimization"))
        self.assertTrue(hasattr(optimization, "ParetoFront"))
        self.assertTrue(hasattr(optimization, "ParetoSolution"))

    def test_collections_exports(self):
        """Test classes are accessible from collections module."""
        from pylinkage.optimization import collections

        self.assertTrue(hasattr(collections, "ParetoFront"))
        self.assertTrue(hasattr(collections, "ParetoSolution"))


if __name__ == "__main__":
    unittest.main()
