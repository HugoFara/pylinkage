"""Tests for scipy-based optimization functions."""

import unittest

import numpy as np

import pylinkage as pl
from pylinkage import optimization
from pylinkage.exceptions import OptimizationError
from pylinkage.optimization.collections import Agent
from pylinkage.optimization.scipy_optimize import (
    differential_evolution_optimization,
    minimize_linkage,
)
from pylinkage.optimization.utils import kinematic_minimization


def prepare_linkage():
    """Simple four-bar linkage for testing.

    Returns:
        A four-bar linkage.
    """
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    pin = pl.Revolute(0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1)
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def fitness_func(loci, **kwargs):
    """Return distance to target point (3, 0).

    This is a minimization problem where the best possible score is 0.
    """
    tip_locus = tuple(x[0] for x in loci)[0]
    return (tip_locus[0] - 3) ** 2 + tip_locus[1] ** 2


class TestDifferentialEvolution(unittest.TestCase):
    """Tests for differential evolution optimization."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_convergence(self):
        """Test if differential evolution converges to near-optimal solution."""
        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        result = differential_evolution_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            bounds=bounds,
            maxiter=50,
            popsize=10,
            order_relation=min,
            verbose=False,
        )

        self.assertEqual(len(result), 1)
        score, dimensions, coord = result[0]
        self.assertAlmostEqual(0.0, score, delta=0.5)

    def test_returns_agent(self):
        """Test that result is a list of Agent objects."""
        dim = len(self.constraints)
        bounds = np.zeros(dim), np.ones(dim) * 5

        result = differential_evolution_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            bounds=bounds,
            maxiter=10,
            popsize=5,
            verbose=False,
        )

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], Agent)

    def test_auto_bounds(self):
        """Test that bounds are auto-generated when not provided."""
        result = differential_evolution_optimization(
            eval_func=fitness_func,
            linkage=self.linkage,
            bounds=None,  # Auto-generate bounds
            maxiter=10,
            popsize=5,
            verbose=False,
        )

        self.assertEqual(len(result), 1)

    def test_invalid_bounds_raises(self):
        """Test that invalid bounds raise OptimizationError."""
        with self.assertRaises(OptimizationError):
            differential_evolution_optimization(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=([0], [1, 2]),  # Mismatched dimensions
                verbose=False,
            )

    def test_invalid_maxiter_raises(self):
        """Test that invalid maxiter raises OptimizationError."""
        with self.assertRaises(OptimizationError):
            differential_evolution_optimization(
                eval_func=fitness_func,
                linkage=self.linkage,
                maxiter=-1,
                verbose=False,
            )


class TestMinimizeLinkage(unittest.TestCase):
    """Tests for local optimization via minimize."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_nelder_mead(self):
        """Test Nelder-Mead optimization."""
        result = minimize_linkage(
            eval_func=fitness_func,
            linkage=self.linkage,
            x0=list(self.constraints),
            method="Nelder-Mead",
            order_relation=min,
            verbose=False,
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Agent)

    def test_powell(self):
        """Test Powell optimization."""
        result = minimize_linkage(
            eval_func=fitness_func,
            linkage=self.linkage,
            x0=list(self.constraints),
            method="Powell",
            order_relation=min,
            verbose=False,
        )

        self.assertEqual(len(result), 1)

    def test_bounded_method(self):
        """Test bounded optimization with L-BFGS-B."""
        dim = len(self.constraints)
        bounds = np.ones(dim) * 0.1, np.ones(dim) * 10

        result = minimize_linkage(
            eval_func=fitness_func,
            linkage=self.linkage,
            x0=list(self.constraints),
            bounds=bounds,
            method="L-BFGS-B",
            order_relation=min,
            verbose=False,
        )

        self.assertEqual(len(result), 1)

    def test_auto_initial_guess(self):
        """Test that initial guess is auto-generated when not provided."""
        result = minimize_linkage(
            eval_func=fitness_func,
            linkage=self.linkage,
            x0=None,  # Auto-generate initial guess
            method="Nelder-Mead",
            order_relation=min,
            verbose=False,
        )

        self.assertEqual(len(result), 1)

    def test_invalid_x0_raises(self):
        """Test that invalid initial guess raises OptimizationError."""
        with self.assertRaises(OptimizationError):
            minimize_linkage(
                eval_func=fitness_func,
                linkage=self.linkage,
                x0=[1.0],  # Wrong number of dimensions
                verbose=False,
            )


class TestModuleExports(unittest.TestCase):
    """Test that scipy optimization functions are properly exported."""

    def test_optimization_module_exports(self):
        """Test functions are accessible from optimization module."""
        self.assertTrue(hasattr(optimization, "differential_evolution_optimization"))
        self.assertTrue(hasattr(optimization, "minimize_linkage"))

    def test_async_exports(self):
        """Test async functions are accessible from optimization module."""
        self.assertTrue(hasattr(optimization, "differential_evolution_optimization_async"))
        self.assertTrue(hasattr(optimization, "minimize_linkage_async"))


if __name__ == "__main__":
    unittest.main()
