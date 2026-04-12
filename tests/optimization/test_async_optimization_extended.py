"""Extended tests for async optimization -- targeting uncovered lines."""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pylinkage as pl
from pylinkage import optimization
from pylinkage.optimization import (
    OptimizationProgress,
    particle_swarm_optimization_async,
    trials_and_errors_optimization_async,
)
from pylinkage.optimization.async_optimization import (
    differential_evolution_optimization_async,
    minimize_linkage_async,
)
from pylinkage.optimization.utils import kinematic_minimization
from pylinkage.population import Ensemble


def prepare_linkage():
    """Simple four-bar linkage."""
    frame_first = pl.Static(0, 0)
    frame_second = pl.Static(3, 0)
    pin = pl.Revolute(0, 2, joint0=frame_first, joint1=frame_second, distance0=3, distance1=1)
    return pl.Linkage(joints=[pin], order=[pin])


@kinematic_minimization
def fitness_func(loci, **kwargs):
    """Minimize distance from tip to (3, 1)."""
    tip_locus = tuple(x[0] for x in loci)[0]
    return (tip_locus[0] - 3) ** 2 + tip_locus[1] ** 2


class TestPSOAsyncWithExecutor(unittest.TestCase):
    """Test PSO async with explicit executor (line 155)."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_with_explicit_executor(self):
        """Test PSO async with an explicitly provided ThreadPoolExecutor."""
        async def run_test():
            dim = len(self.constraints)
            bounds = np.zeros(dim), np.ones(dim) * 5
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await particle_swarm_optimization_async(
                    eval_func=fitness_func,
                    linkage=self.linkage,
                    bounds=bounds,
                    n_particles=20,
                    iters=10,
                    order_relation=min,
                    executor=executor,
                )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)


class TestTrialsAsyncWithExecutor(unittest.TestCase):
    """Test trials async with explicit executor (line 260)."""

    linkage = prepare_linkage()

    def test_with_explicit_executor(self):
        """Test trials async with explicit executor."""
        async def run_test():
            bounds = optimization.generate_bounds(self.linkage.get_num_constraints(), 2, 2)
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await trials_and_errors_optimization_async(
                    eval_func=fitness_func,
                    linkage=self.linkage,
                    divisions=3,
                    bounds=bounds,
                    n_results=3,
                    order_relation=min,
                    executor=executor,
                )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)

    def test_with_explicit_parameters(self):
        """Test trials async when parameters are explicitly provided (line 231)."""
        async def run_test():
            constraints = list(self.linkage.get_num_constraints(flat=True))
            bounds = optimization.generate_bounds(self.linkage.get_num_constraints(), 2, 2)
            return await trials_and_errors_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                parameters=constraints,
                divisions=3,
                bounds=bounds,
                n_results=3,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)


class TestDifferentialEvolutionAsync(unittest.TestCase):
    """Test differential_evolution_optimization_async (lines 323-374)."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_basic_de_async(self):
        """Test basic differential evolution async optimization."""
        async def run_test():
            dim = len(self.constraints)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            return await differential_evolution_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=bounds,
                maxiter=5,
                popsize=5,
                seed=42,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        self.assertIsInstance(results, Ensemble)
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].score)

    def test_de_async_with_progress(self):
        """Test DE async with progress callback."""
        progress_updates = []

        def on_progress(progress: OptimizationProgress):
            progress_updates.append(progress)

        async def run_test():
            dim = len(self.constraints)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            return await differential_evolution_optimization_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                bounds=bounds,
                maxiter=5,
                popsize=5,
                seed=42,
                order_relation=min,
                on_progress=on_progress,
            )

        asyncio.run(run_test())
        self.assertGreaterEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates[0].current_iteration, 0)
        self.assertFalse(progress_updates[0].is_complete)
        self.assertTrue(progress_updates[-1].is_complete)

    def test_de_async_with_executor(self):
        """Test DE async with explicit executor."""
        async def run_test():
            dim = len(self.constraints)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await differential_evolution_optimization_async(
                    eval_func=fitness_func,
                    linkage=self.linkage,
                    bounds=bounds,
                    maxiter=5,
                    popsize=5,
                    seed=42,
                    order_relation=min,
                    executor=executor,
                )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)


class TestMinimizeLinkageAsync(unittest.TestCase):
    """Test minimize_linkage_async (lines 415-465)."""

    linkage = prepare_linkage()
    constraints = tuple(linkage.get_num_constraints())

    def test_basic_minimize_async(self):
        """Test basic minimize_linkage_async."""
        async def run_test():
            x0 = list(self.constraints)
            dim = len(x0)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            return await minimize_linkage_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                x0=x0,
                bounds=bounds,
                method="Nelder-Mead",
                maxiter=20,
                order_relation=min,
            )

        results = asyncio.run(run_test())
        self.assertIsInstance(results, Ensemble)
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].score)

    def test_minimize_async_with_progress(self):
        """Test minimize async with progress callback."""
        progress_updates = []

        def on_progress(progress: OptimizationProgress):
            progress_updates.append(progress)

        async def run_test():
            x0 = list(self.constraints)
            dim = len(x0)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            return await minimize_linkage_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                x0=x0,
                bounds=bounds,
                method="Nelder-Mead",
                maxiter=20,
                order_relation=min,
                on_progress=on_progress,
            )

        asyncio.run(run_test())
        self.assertGreaterEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates[0].current_iteration, 0)
        self.assertFalse(progress_updates[0].is_complete)
        self.assertTrue(progress_updates[-1].is_complete)
        self.assertEqual(progress_updates[0].total_iterations, 20)

    def test_minimize_async_default_maxiter(self):
        """Test minimize async with default maxiter (None)."""
        progress_updates = []

        def on_progress(progress: OptimizationProgress):
            progress_updates.append(progress)

        async def run_test():
            x0 = list(self.constraints)
            dim = len(x0)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            return await minimize_linkage_async(
                eval_func=fitness_func,
                linkage=self.linkage,
                x0=x0,
                bounds=bounds,
                method="Nelder-Mead",
                order_relation=min,
                on_progress=on_progress,
            )

        asyncio.run(run_test())
        # Default maxiter should be 1000
        self.assertEqual(progress_updates[0].total_iterations, 1000)

    def test_minimize_async_with_executor(self):
        """Test minimize async with explicit executor."""
        async def run_test():
            x0 = list(self.constraints)
            dim = len(x0)
            bounds = (np.zeros(dim).tolist(), (np.ones(dim) * 5).tolist())
            with ThreadPoolExecutor(max_workers=1) as executor:
                return await minimize_linkage_async(
                    eval_func=fitness_func,
                    linkage=self.linkage,
                    x0=x0,
                    bounds=bounds,
                    method="Nelder-Mead",
                    maxiter=10,
                    order_relation=min,
                    executor=executor,
                )

        results = asyncio.run(run_test())
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
