"""Tests for the mixed-variable co-optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from pylinkage.optimization.co_optimization_types import (
    CoOptimizationConfig,
    CoOptSolution,
    MixedChromosome,
)
from pylinkage.optimization.operators import (
    MixedCrossover,
    MixedMutation,
    warm_start_sampling,
)
from pylinkage.synthesis._types import PrecisionPoint
from pylinkage.topology.catalog import load_catalog


class TestMixedChromosome:
    """Tests for MixedChromosome."""

    def test_copy_is_independent(self) -> None:
        ch = MixedChromosome(
            topology_idx=0,
            dimensions=np.array([1.0, 2.0, 3.0]),
        )
        copy = ch.copy()
        copy.topology_idx = 5
        copy.dimensions[0] = 99.0
        assert ch.topology_idx == 0
        assert ch.dimensions[0] == 1.0


class TestMixedCrossover:
    """Tests for the crossover operator."""

    def test_produces_two_children(self) -> None:
        rng = np.random.default_rng(42)
        crossover = MixedCrossover(crossover_prob=1.0)
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([1.0, 4.0, 5.0, 6.0])
        xl = np.array([0.0, 0.1, 0.1, 0.1])
        xu = np.array([1.0, 10.0, 10.0, 10.0])

        c1, c2 = crossover(a, b, xl, xu, n_topologies=2, rng=rng)
        assert c1.shape == a.shape
        assert c2.shape == b.shape

    def test_no_crossover_returns_parents(self) -> None:
        rng = np.random.default_rng(42)
        crossover = MixedCrossover(crossover_prob=0.0)
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([1.0, 4.0, 5.0])
        xl = np.zeros(3)
        xu = np.ones(3) * 10

        c1, c2 = crossover(a, b, xl, xu, n_topologies=2, rng=rng)
        np.testing.assert_array_equal(c1, a)
        np.testing.assert_array_equal(c2, b)

    def test_children_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        crossover = MixedCrossover(crossover_prob=1.0)
        xl = np.array([0.0, 0.5, 0.5])
        xu = np.array([5.0, 10.0, 10.0])
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 8.0, 7.0])

        for _ in range(20):
            c1, c2 = crossover(a, b, xl, xu, n_topologies=6, rng=rng)
            # Dimensions (index 1+) should be within bounds
            assert np.all(c1[1:] >= xl[1:])
            assert np.all(c1[1:] <= xu[1:])
            assert np.all(c2[1:] >= xl[1:])
            assert np.all(c2[1:] <= xu[1:])


class TestMixedMutation:
    """Tests for the mutation operator."""

    def test_mutates_in_place(self) -> None:
        rng = np.random.default_rng(42)
        mutation = MixedMutation(
            topology_rate=1.0,
            dimension_sigma=0.3,
            dimension_rate=1.0,
        )
        x = np.array([0.0, 5.0, 5.0])
        xl = np.array([0.0, 0.1, 0.1])
        xu = np.array([2.0, 10.0, 10.0])
        original = x.copy()

        mutation(x, xl, xu, n_topologies=3, neighbor_map={0: [1, 2]}, rng=rng)
        # At least some values should have changed
        assert not np.array_equal(x, original)

    def test_stays_within_bounds(self) -> None:
        rng = np.random.default_rng(42)
        mutation = MixedMutation(
            topology_rate=0.5,
            dimension_sigma=0.5,
            dimension_rate=1.0,
        )
        xl = np.array([0.0, 1.0, 1.0])
        xu = np.array([5.0, 10.0, 10.0])

        for _ in range(50):
            x = np.array([2.0, 5.0, 5.0])
            mutation(x, xl, xu, n_topologies=6, neighbor_map={2: [0, 1, 3]}, rng=rng)
            assert np.all(x[1:] >= xl[1:])
            assert np.all(x[1:] <= xu[1:])


class TestWarmStartSampling:
    """Tests for warm_start_sampling."""

    def test_includes_seeds(self) -> None:
        rng = np.random.default_rng(42)
        seeds = [
            MixedChromosome(topology_idx=0, dimensions=np.array([1.0, 2.0])),
            MixedChromosome(topology_idx=1, dimensions=np.array([3.0, 4.0])),
        ]
        xl = np.array([0.0, 0.1, 0.1])
        xu = np.array([5.0, 10.0, 10.0])

        pop = warm_start_sampling(seeds, pop_size=10, n_dim=3, xl=xl, xu=xu, n_topologies=6, rng=rng)
        assert pop.shape == (10, 3)
        # First two rows should match seeds
        assert pop[0, 0] == 0.0
        assert pop[0, 1] == 1.0
        assert pop[1, 0] == 1.0
        assert pop[1, 1] == 3.0

    def test_fills_to_pop_size(self) -> None:
        rng = np.random.default_rng(42)
        xl = np.zeros(4)
        xu = np.ones(4) * 10

        pop = warm_start_sampling([], pop_size=20, n_dim=4, xl=xl, xu=xu, n_topologies=3, rng=rng)
        assert pop.shape == (20, 4)

    def test_clips_to_bounds(self) -> None:
        rng = np.random.default_rng(42)
        seeds = [
            MixedChromosome(topology_idx=0, dimensions=np.array([100.0])),
        ]
        xl = np.array([0.0, 0.1])
        xu = np.array([5.0, 10.0])

        pop = warm_start_sampling(seeds, pop_size=5, n_dim=2, xl=xl, xu=xu, n_topologies=6, rng=rng)
        assert pop[0, 1] <= xu[1]


class TestCoOptimizationConfig:
    """Tests for CoOptimizationConfig defaults."""

    def test_defaults(self) -> None:
        cfg = CoOptimizationConfig()
        assert cfg.max_links == 8
        assert cfg.algorithm == "nsga2"
        assert cfg.n_generations == 200
        assert cfg.pop_size == 100

    def test_custom_values(self) -> None:
        cfg = CoOptimizationConfig(
            max_links=6,
            algorithm="nsga3",
            n_generations=50,
            pop_size=30,
        )
        assert cfg.max_links == 6
        assert cfg.algorithm == "nsga3"


class TestCatalogIndexMethods:
    """Tests for topology_index and topology_by_index."""

    def test_round_trip(self) -> None:
        catalog = load_catalog()
        for entry in catalog:
            idx = catalog.topology_index(entry.id)
            assert catalog.topology_by_index(idx).id == entry.id

    def test_unknown_raises_key_error(self) -> None:
        catalog = load_catalog()
        with pytest.raises(KeyError):
            catalog.topology_index("nonexistent")

    def test_out_of_range_raises_index_error(self) -> None:
        catalog = load_catalog()
        with pytest.raises(IndexError):
            catalog.topology_by_index(999)
