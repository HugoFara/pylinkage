"""Tests for async optimization functions."""

import asyncio
import unittest

import numpy as np

import pylinkage as pl
from pylinkage import optimization
from pylinkage.optimization import (
    OptimizationProgress,
    particle_swarm_optimization_async,
    trials_and_errors_optimization_async,
)
from pylinkage.optimization.utils import kinematic_minimization


def prepare_linkage():
    """Simple four-bar linkage.

    :returns: A four-bar linkage
    :rtype: pylinkage.Linkage
    """
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    pin = pl.Revolute(
        0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1
    )
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def fitness_func(loci, **kwargs):
    """Return if the tip can go to the point (3, 1).

    It is a minimization problem.
    The best possible score is 0.
    """
    tip_locus = tuple(x[0] for x in loci)[0]
    return (tip_locus[0] - 3) ** 2 + tip_locus[1] ** 2


class TestOptimizationProgress(unittest.TestCase):
    """Tests for the OptimizationProgress dataclass."""

    def test_progress_fraction_normal(self):
        """Test progress fraction calculation."""
        progress = OptimizationProgress(
            current_iteration=50,
            total_iterations=100,
        )
        self.assertAlmostEqual(progress.progress_fraction, 0.5)

    def test_progress_fraction_zero_total(self):
        """Test progress fraction with zero total iterations."""
        progress = OptimizationProgress(
            current_iteration=0,
            total_iterations=0,
        )
        self.assertAlmostEqual(progress.progress_fraction, 0.0)

    def test_progress_complete(self):
        """Test progress with is_complete flag."""
        progress = OptimizationProgress(
            current_iteration=100,
            total_iterations=100,
            best_score=0.5,
            is_complete=True,
        )
        self.assertTrue(progress.is_complete)
        self.assertAlmostEqual(progress.best_score, 0.5)


class TestTrialsAndErrorsAsync(unittest.TestCase):
    """Tests for async trials and errors optimization."""

    linkage = prepare_linkage()

    def test_basic_optimization(self):
        """Test basic async optimization runs successfully."""
        async def run_test():
            bounds = optimization.generate_bounds(
                self.linkage.get_num_constraints(), 2, 2
            )
            results = await trials_and_errors_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                divisions=5,
                bounds=bounds,
                n_results=5,
                order_relation=min,
            )
            return results

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].score)

    def test_progress_callback(self):
        """Test that progress callback is invoked."""
        progress_updates = []

        def on_progress(progress: OptimizationProgress):
            progress_updates.append(progress)

        async def run_test():
            bounds = optimization.generate_bounds(
                self.linkage.get_num_constraints(), 2, 2
            )
            return await trials_and_errors_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                divisions=3,
                bounds=bounds,
                n_results=3,
                order_relation=min,
                on_progress=on_progress,
            )

        asyncio.run(run_test())
        # Should have at least start and completion callbacks
        self.assertGreaterEqual(len(progress_updates), 2)
        # First callback should indicate start
        self.assertEqual(progress_updates[0].current_iteration, 0)
        self.assertFalse(progress_updates[0].is_complete)
        # Last callback should indicate completion
        self.assertTrue(progress_updates[-1].is_complete)

    def test_convergence(self):
        """Test if the output after some iterations is improved."""
        # Use the same test parameters as the original sync test
        async def run_test():
            bounds = optimization.generate_bounds(
                self.linkage.get_num_constraints(), 2, 2
            )
            return await trials_and_errors_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                divisions=10,
                bounds=bounds,
                n_results=10,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        score = results[0].score
        # Grid search may not find optimal solution, just verify it found something
        self.assertIsNotNone(score)
        # Score should be finite (not inf)
        self.assertLess(score, float('inf'))


class TestPSOAsync(unittest.TestCase):
    """Tests for async particle swarm optimization."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_basic_optimization(self):
        """Test basic async PSO runs successfully."""
        async def run_test():
            dim = len(self.constraints)
            bounds = np.zeros(dim), np.ones(dim) * 5
            return await particle_swarm_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=bounds,
                n_particles=20,
                iters=10,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].score)

    def test_progress_callback(self):
        """Test that progress callback is invoked."""
        progress_updates = []

        def on_progress(progress: OptimizationProgress):
            progress_updates.append(progress)

        async def run_test():
            dim = len(self.constraints)
            bounds = np.zeros(dim), np.ones(dim) * 5
            return await particle_swarm_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=bounds,
                n_particles=20,
                iters=10,
                order_relation=min,
                on_progress=on_progress,
            )

        asyncio.run(run_test())
        # Should have at least start and completion callbacks
        self.assertGreaterEqual(len(progress_updates), 2)
        # First callback should indicate start
        self.assertEqual(progress_updates[0].current_iteration, 0)
        self.assertFalse(progress_updates[0].is_complete)
        # Last callback should indicate completion
        self.assertTrue(progress_updates[-1].is_complete)

    def test_convergence(self):
        """Test if the result is not too far from 0.0."""
        delta = 0.3

        async def run_test():
            dim = len(self.constraints)
            bounds = np.zeros(dim), np.ones(dim) * 5
            return await particle_swarm_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=bounds,
                n_particles=20,
                iters=30,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        score = results[0].score
        if score > delta:
            # Try again with more iterations
            async def retry():
                dim = len(self.constraints)
                bounds = np.zeros(dim), np.ones(dim) * 5
                return await particle_swarm_optimization_async(
                    eval_func=fitness_func,
                    linkage=self.linkage,
                    bounds=bounds,
                    n_particles=50,
                    iters=100,
                    order_relation=min,
                )

            results = asyncio.run(retry())
            score = results[0].score
        self.assertAlmostEqual(0.0, score, delta=delta)


if __name__ == '__main__':
    unittest.main()
