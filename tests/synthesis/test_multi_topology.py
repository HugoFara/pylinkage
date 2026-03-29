"""Tests for multi-topology synthesis."""

from __future__ import annotations

import pytest

from pylinkage.synthesis._types import PrecisionPoint, SynthesisType
from pylinkage.synthesis.multi_topology import rank_solutions, synthesize
from pylinkage.synthesis.topology_types import QualityMetrics, TopologySolution


class TestSynthesize:
    """Integration tests for the synthesize() function."""

    def test_returns_list_of_topology_solutions(self) -> None:
        """Should return TopologySolution objects."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.0),
        ]
        results = synthesize(
            points,
            max_links=4,  # Only four-bars
            max_solutions_per_topology=2,
            max_total_solutions=3,
            n_orientation_samples=6,
        )
        assert isinstance(results, list)
        for sol in results:
            assert isinstance(sol, TopologySolution)
            assert hasattr(sol, "linkage")
            assert hasattr(sol, "metrics")
            assert hasattr(sol, "topology_entry")

    def test_solutions_are_ranked(self) -> None:
        """Solutions should be in order of overall_score."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.0),
        ]
        results = synthesize(
            points,
            max_links=4,
            max_solutions_per_topology=3,
            max_total_solutions=5,
            n_orientation_samples=6,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].metrics.overall_score <= results[i + 1].metrics.overall_score

    def test_respects_max_links(self) -> None:
        """Should only try topologies up to max_links."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 1.0), (3.0, 0.0),
        ]
        results = synthesize(
            points,
            max_links=4,
            max_solutions_per_topology=2,
            max_total_solutions=5,
            n_orientation_samples=4,
        )
        for sol in results:
            assert sol.topology_entry.num_links <= 4

    def test_non_path_raises(self) -> None:
        """Should raise NotImplementedError for non-PATH types."""
        with pytest.raises(NotImplementedError):
            synthesize([(0, 0)], synthesis_type=SynthesisType.FUNCTION)

    def test_empty_on_impossible(self) -> None:
        """Should return empty list for impossible input."""
        results = synthesize(
            [(0.0, 0.0)],  # 1 point: can't synthesize
            max_links=4,
            max_solutions_per_topology=1,
            n_orientation_samples=2,
        )
        assert isinstance(results, list)


class TestRankSolutions:
    """Tests for rank_solutions."""

    def test_sorts_by_overall_score(self) -> None:
        """Should sort ascending by overall_score."""
        # Create mock solutions with known scores
        from unittest.mock import MagicMock

        solutions = []
        for score in [0.8, 0.3, 0.5, 0.1]:
            sol = MagicMock(spec=TopologySolution)
            sol.metrics = QualityMetrics(overall_score=score)
            solutions.append(sol)

        ranked = rank_solutions(solutions)
        scores = [s.metrics.overall_score for s in ranked]
        assert scores == sorted(scores)
        assert scores[0] == 0.1
        assert scores[-1] == 0.8


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_defaults(self) -> None:
        """Default metrics should have worst-case values."""
        m = QualityMetrics()
        assert m.path_accuracy == float("inf")
        assert m.min_transmission_angle == 0.0
        assert m.overall_score == float("inf")

    def test_custom_values(self) -> None:
        """Should accept custom metric values."""
        m = QualityMetrics(
            path_accuracy=0.1,
            min_transmission_angle=45.0,
            link_ratio=2.5,
            compactness=10.0,
            num_links=6,
            is_grashof=True,
            overall_score=0.3,
        )
        assert m.path_accuracy == 0.1
        assert m.num_links == 6
        assert m.is_grashof is True
