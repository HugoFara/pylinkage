"""Integration tests for co-optimization pipeline."""

from __future__ import annotations

import pytest

from pylinkage.optimization.co_optimization_types import (
    CoOptimizationConfig,
    MixedChromosome,
)
from pylinkage.synthesis._types import PrecisionPoint

# Skip all tests if pymoo is not installed
pymoo = pytest.importorskip("pymoo")

import numpy as np  # noqa: E402

from pylinkage.optimization.mixed_variable import co_optimize  # noqa: E402
from pylinkage.optimization.warm_start import (  # noqa: E402
    synthesis_to_chromosomes,
    warm_start_co_optimization,
)
from pylinkage.synthesis.ranking import compute_compactness, compute_path_accuracy  # noqa: E402
from pylinkage.topology.catalog import load_catalog  # noqa: E402


@pytest.fixture()
def simple_points() -> list[PrecisionPoint]:
    return [(0.0, 0.0), (1.0, 2.0), (2.0, 3.0), (3.0, 2.0), (4.0, 0.0)]


class TestCoOptimize:
    """Integration tests for co_optimize."""

    def test_returns_result(self, simple_points: list[PrecisionPoint]) -> None:
        """Should return a CoOptimizationResult even with tiny config."""
        def accuracy(linkage):
            return compute_path_accuracy(linkage, simple_points)

        config = CoOptimizationConfig(
            max_links=6,
            n_generations=3,
            pop_size=10,
            verbose=False,
            seed=42,
        )
        result = co_optimize(
            objectives=[accuracy],
            precision_points=simple_points,
            config=config,
            objective_names=["Path Error"],
        )
        assert result.pareto_front is not None
        assert result.config == config

    def test_multi_objective(self, simple_points: list[PrecisionPoint]) -> None:
        """Should handle multiple objectives."""
        def accuracy(linkage):
            return compute_path_accuracy(linkage, simple_points)

        def compact(linkage):
            return compute_compactness(linkage)

        config = CoOptimizationConfig(
            max_links=6,
            n_generations=3,
            pop_size=10,
            verbose=False,
            seed=42,
        )
        result = co_optimize(
            objectives=[accuracy, compact],
            precision_points=simple_points,
            config=config,
        )
        assert result.pareto_front is not None
        # Solutions should have 2 scores each
        for sol in result.solutions:
            assert len(sol.scores) == 2

    def test_with_initial_population(self, simple_points: list[PrecisionPoint]) -> None:
        """Should accept initial population without crashing."""
        def accuracy(linkage):
            return compute_path_accuracy(linkage, simple_points)

        seeds = [
            MixedChromosome(
                topology_idx=0,
                dimensions=np.array([1.0, 3.0, 3.0]),
            ),
        ]

        config = CoOptimizationConfig(
            max_links=6,
            n_generations=2,
            pop_size=8,
            verbose=False,
            seed=42,
        )
        result = co_optimize(
            objectives=[accuracy],
            precision_points=simple_points,
            config=config,
            initial_population=seeds,
        )
        assert result.pareto_front is not None


class TestSynthesisToChromosomes:
    """Tests for synthesis_to_chromosomes."""

    def test_converts_four_bar_solutions(self, simple_points: list[PrecisionPoint]) -> None:
        """Should convert synthesis results to chromosomes."""
        from pylinkage.synthesis.multi_topology import synthesize

        solutions = synthesize(
            simple_points,
            max_links=4,
            max_total_solutions=3,
            n_orientation_samples=6,
        )

        if not solutions:
            pytest.skip("No synthesis solutions found")

        catalog = load_catalog()
        chromosomes = synthesis_to_chromosomes(solutions, catalog)
        assert len(chromosomes) > 0
        for ch in chromosomes:
            assert ch.topology_idx >= 0
            assert len(ch.dimensions) > 0
            assert all(d > 0 for d in ch.dimensions)


class TestWarmStartPipeline:
    """Tests for warm_start_co_optimization."""

    def test_runs_end_to_end(self, simple_points: list[PrecisionPoint]) -> None:
        """Full pipeline should complete without error."""
        def accuracy(linkage):
            return compute_path_accuracy(linkage, simple_points)

        config = CoOptimizationConfig(
            max_links=4,
            n_generations=2,
            pop_size=8,
            verbose=False,
            seed=42,
        )
        result = warm_start_co_optimization(
            precision_points=simple_points,
            objectives=[accuracy],
            config=config,
            objective_names=["Path Error"],
            max_synthesis_solutions=5,
            n_orientation_samples=4,
        )
        assert result.pareto_front is not None
        assert result.config == config
