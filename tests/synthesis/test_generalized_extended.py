"""Extended tests for generalized N-bar synthesis covering uncovered branches."""

from __future__ import annotations

import pytest

from pylinkage.synthesis._types import PrecisionPoint
from pylinkage.synthesis.generalized import (
    _compute_link_lengths,
    _get_decomposition,
    _partition_points_exhaustive,
    _partition_points_greedy,
    _synthesize_chain,
    _synthesize_dyad_group,
    _synthesize_fourbar_as_nbar,
    _synthesize_sixbar_as_nbar,
    _synthesize_triad_group,
    generalized_synthesis,
)
from pylinkage.synthesis.topology_types import NBarSolution
from pylinkage.topology.catalog import load_catalog


@pytest.fixture
def catalog():
    return load_catalog()


class TestPartitionExhaustive:
    def test_insufficient_points(self, catalog):
        entry = catalog.get("watt")
        dec = _get_decomposition(entry)
        # Watt has 4 groups, min 3 each = 12. 8 points is too few.
        result = _partition_points_exhaustive(8, dec)
        assert result == []

    def test_one_group(self, catalog):
        entry = catalog.get("four-bar")
        dec = _get_decomposition(entry)
        assert dec is not None
        # four-bar has 1 group
        result = _partition_points_exhaustive(4, dec)
        assert len(result) == 1

    def test_one_group_too_few(self, catalog):
        entry = catalog.get("four-bar")
        dec = _get_decomposition(entry)
        # Below min_per_group
        result = _partition_points_exhaustive(2, dec)
        assert result == []

    def test_one_group_too_many(self, catalog):
        entry = catalog.get("four-bar")
        dec = _get_decomposition(entry)
        # Above max_per_group
        result = _partition_points_exhaustive(10, dec)
        assert result == []

    def test_four_groups(self, catalog):
        entry = catalog.get("watt")
        dec = _get_decomposition(entry)
        # Watt has 4 groups; 12 points = 3 each
        result = _partition_points_exhaustive(12, dec)
        assert len(result) > 0
        # All partitions should fully cover 12 points
        for partition in result:
            total = sum(len(g) for g in partition)
            assert total == 12


class TestPartitionGreedyExtended:
    def test_fourbar_single_group(self, catalog):
        entry = catalog.get("four-bar")
        dec = _get_decomposition(entry)
        partitions = _partition_points_greedy(4, dec)
        assert len(partitions) == 1
        assert sum(len(g) for g in partitions[0]) == 4

    def test_empty_decomposition_returns_empty(self):
        # Construct a mock empty decomposition
        class FakeDec:
            groups = []
        result = _partition_points_greedy(10, FakeDec())  # type: ignore[arg-type]
        assert result == []


class TestGetDecomposition:
    def test_returns_none_for_invalid_topology(self):
        # Create a fake topology entry that will fail
        class FakeEntry:
            family = "invalid"
            id = "bogus"
            def to_graph(self):
                raise ValueError("bad graph")
        result = _get_decomposition(FakeEntry())  # type: ignore[arg-type]
        assert result is None


class TestSynthesizeFourbarAsNbarExtra:
    def test_with_good_points(self):
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 2.0), (3.0, 1.5), (4.0, 0.0),
        ]
        results = _synthesize_fourbar_as_nbar(points, max_solutions=3)
        assert isinstance(results, list)
        for nbar in results:
            assert nbar.topology_id == "four-bar"
            assert "A" in nbar.joint_positions


class TestSynthesizeSixbarAsNbar:
    def test_watt(self):
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.0), (2.0, 1.5),
            (3.0, 1.0), (4.0, 0.5), (5.0, 0.0),
        ]
        results = _synthesize_sixbar_as_nbar(
            points, "watt", max_solutions=2, n_orientation_samples=4
        )
        assert isinstance(results, list)
        for nbar in results:
            assert nbar.topology_id == "watt"

    def test_stephenson(self):
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.0), (2.0, 1.5),
            (3.0, 1.0), (4.0, 0.5), (5.0, 0.0),
        ]
        results = _synthesize_sixbar_as_nbar(
            points, "stephenson", max_solutions=2, n_orientation_samples=4
        )
        assert isinstance(results, list)
        for nbar in results:
            assert nbar.topology_id == "stephenson"


class TestGeneralizedSynthesisEightBar:
    def test_eight_bar_topology_returns_list(self, catalog):
        # Use an eight-bar topology and test that generalized_synthesis
        # returns a list (may be empty if synthesis fails)
        entry = catalog.get("eight-bar-01")
        assert entry is not None

        # 21 points for 7 groups @ 3 each
        points = [(float(i), float(i % 3)) for i in range(21)]
        results = generalized_synthesis(
            entry, points, max_solutions=1, n_orientation_samples=4
        )
        assert isinstance(results, list)

    def test_eight_bar_with_exhaustive(self, catalog):
        entry = catalog.get("eight-bar-01")
        assert entry is not None
        points = [(float(i), float(i % 2)) for i in range(21)]
        results = generalized_synthesis(
            entry, points, max_solutions=1, n_orientation_samples=2,
            partition_strategy="exhaustive",
        )
        assert isinstance(results, list)

    def test_insufficient_points(self, catalog):
        entry = catalog.get("eight-bar-01")
        assert entry is not None
        points: list[PrecisionPoint] = [(0.0, 0.0), (1.0, 1.0)]
        results = generalized_synthesis(entry, points, max_solutions=1)
        assert results == []


class TestSynthesizeDyadGroup:
    def test_insufficient_points_returns_empty(self, catalog):
        entry = catalog.get("four-bar")
        dec = _get_decomposition(entry)
        group = dec.groups[0]
        # Only 2 points
        result = _synthesize_dyad_group(
            group_index=0,
            group=group,
            assigned_points=[(0.0, 0.0), (1.0, 1.0)],
            known_positions={},
            orientations=[0.0, 0.1],
            prev_fourbar=None,
        )
        assert result == []


class TestSynthesizeTriadGroup:
    def test_insufficient_points_returns_empty(self, catalog):
        entry = catalog.get("stephenson")
        dec = _get_decomposition(entry)
        group = dec.groups[0]  # triad
        result = _synthesize_triad_group(
            group_index=0,
            group=group,
            assigned_points=[(0.0, 0.0)],
            known_positions={},
            max_solutions=1,
        )
        assert result == []

    def test_triad_optimization_runs(self, catalog):
        entry = catalog.get("stephenson")
        dec = _get_decomposition(entry)
        group = dec.groups[0]  # triad
        result = _synthesize_triad_group(
            group_index=0,
            group=group,
            assigned_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)],
            known_positions={},
            max_solutions=1,
        )
        assert len(result) == 1
        gsr, new_positions, fourbar = result[0]
        assert gsr.group_index == 0
        assert fourbar is None
        assert len(new_positions) == len(group.internal_nodes)


class TestComputeLinkLengths:
    def test_empty_positions(self, catalog):
        entry = catalog.get("four-bar")
        lengths = _compute_link_lengths({}, entry)
        assert lengths == {}

    def test_complete_positions(self, catalog):
        entry = catalog.get("four-bar")
        positions = {
            "J0_2": (0.0, 0.0),
            "J1_3": (4.0, 0.0),
            "J0_0": (0.0, 0.0),
            "J1_1": (4.0, 0.0),
            "J2_3": (2.0, 2.0),
        }
        lengths = _compute_link_lengths(positions, entry)
        # Every edge whose both nodes are in positions should have a length
        for v in lengths.values():
            assert v >= 0.0


class TestSynthesizeChainDirect:
    def test_with_watt_partition(self, catalog):
        entry = catalog.get("watt")
        dec = _get_decomposition(entry)
        assert dec is not None
        # Watt has 4 groups, so 12 points (3 each)
        points: list[PrecisionPoint] = [(float(i), float(i % 3)) for i in range(12)]
        partition = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
        orientations = [0.1, 0.2, 0.3]
        results = _synthesize_chain(
            dec, entry, points, partition, orientations,
            max_candidates_per_group=1,
        )
        assert isinstance(results, list)
        # Every result should be an NBarSolution
        for nbar in results:
            assert isinstance(nbar, NBarSolution)


class TestGeneralizedSynthesisNoDecomposition:
    def test_returns_empty_on_bad_topology(self):
        class FakeEntry:
            family = "not-a-family"
            id = "bogus"
            def to_graph(self):
                raise ValueError("nope")
        points: list[PrecisionPoint] = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]
        results = generalized_synthesis(FakeEntry(), points, max_solutions=1)  # type: ignore[arg-type]
        assert results == []
