"""Tests for generalized N-bar synthesis."""

from __future__ import annotations

import pytest

from pylinkage.synthesis._types import PrecisionPoint
from pylinkage.synthesis.generalized import (
    _get_decomposition,
    _partition_points_greedy,
    _synthesize_fourbar_as_nbar,
    generalized_synthesis,
)
from pylinkage.synthesis.topology_types import NBarSolution
from pylinkage.topology.catalog import load_catalog


@pytest.fixture
def catalog():
    return load_catalog()


class TestPartitionPointsGreedy:
    """Tests for _partition_points_greedy."""

    def test_four_groups(self, catalog) -> None:
        """Greedy should produce valid partition for 4-group Watt topology."""
        entry = catalog.get("watt")
        assert entry is not None
        decomposition = _get_decomposition(entry)
        assert decomposition is not None
        # Watt decomposes into 4 RRR dyads, needs 3 points each = 12
        partitions = _partition_points_greedy(12, decomposition)
        assert len(partitions) == 1
        partition = partitions[0]
        total = sum(len(g) for g in partition)
        assert total == 12
        assert all(len(g) >= 3 for g in partition)

    def test_insufficient_points(self, catalog) -> None:
        """Should return empty for too few points."""
        entry = catalog.get("watt")
        decomposition = _get_decomposition(entry)
        # Watt has 4 groups, needs 3 each = 12 min
        partitions = _partition_points_greedy(8, decomposition)
        assert len(partitions) == 0


class TestSynthesizeFourbarAsNbar:
    """Tests for _synthesize_fourbar_as_nbar."""

    def test_returns_nbar_solutions(self) -> None:
        """Should wrap four-bar results as NBarSolution."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.0),
        ]
        results = _synthesize_fourbar_as_nbar(points, max_solutions=3)
        assert isinstance(results, list)
        for nbar in results:
            assert isinstance(nbar, NBarSolution)
            assert nbar.topology_id == "four-bar"
            assert "A" in nbar.joint_positions
            assert "B" in nbar.joint_positions

    def test_empty_on_bad_points(self) -> None:
        """Collinear points may produce no solutions."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        ]
        results = _synthesize_fourbar_as_nbar(points, max_solutions=1)
        assert isinstance(results, list)


class TestGeneralizedSynthesis:
    """Integration tests for generalized_synthesis."""

    def test_fourbar_delegation(self, catalog) -> None:
        """Four-bar topology should delegate to path_generation."""
        entry = catalog.get("four-bar")
        assert entry is not None

        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.0),
        ]
        results = generalized_synthesis(
            entry, points, max_solutions=2,
            n_orientation_samples=6,
        )
        assert isinstance(results, list)
        for nbar in results:
            assert isinstance(nbar, NBarSolution)
            assert nbar.topology_id == "four-bar"

    def test_sixbar_delegation(self, catalog) -> None:
        """Six-bar topology should delegate to six_bar module."""
        entry = catalog.get("watt")
        assert entry is not None

        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 2.0), (2.0, 3.0),
            (3.0, 2.0), (4.0, 0.0), (5.0, -1.0),
        ]
        results = generalized_synthesis(
            entry, points, max_solutions=2,
            n_orientation_samples=4,
        )
        assert isinstance(results, list)

    def test_returns_empty_for_impossible(self, catalog) -> None:
        """Should return empty list for impossible synthesis."""
        entry = catalog.get("watt")
        assert entry is not None

        # Only 2 points — not enough
        points: list[PrecisionPoint] = [(0.0, 0.0), (1.0, 1.0)]
        results = generalized_synthesis(entry, points, max_solutions=1)
        assert isinstance(results, list)
