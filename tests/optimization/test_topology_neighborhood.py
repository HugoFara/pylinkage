"""Tests for topology neighborhood graph."""

from __future__ import annotations

from pylinkage.optimization.topology_neighborhood import (
    TopologyNeighbor,
    build_neighborhood_graph,
    topology_distance,
    topology_neighbors,
)
from pylinkage.topology.catalog import load_catalog


class TestBuildNeighborhoodGraph:
    """Tests for build_neighborhood_graph."""

    def test_all_topologies_have_entries(self) -> None:
        """Every topology in the catalog should appear in the graph."""
        catalog = load_catalog()
        graph = build_neighborhood_graph(catalog)
        for entry in catalog:
            assert entry.id in graph

    def test_four_bar_has_add_dyad_neighbors(self) -> None:
        """Four-bar should connect to six-bars via add_dyad."""
        catalog = load_catalog()
        graph = build_neighborhood_graph(catalog)
        four_bar_neighbors = graph["four-bar"]
        add_dyad_targets = [
            n.target_id for n in four_bar_neighbors if n.operation == "add_dyad"
        ]
        # Four-bar should connect to at least one six-bar
        assert len(add_dyad_targets) > 0
        for tid in add_dyad_targets:
            entry = catalog.get(tid)
            assert entry is not None
            assert entry.num_links == 6

    def test_watt_has_remove_dyad_to_four_bar(self) -> None:
        """Watt six-bar should connect back to four-bar via remove_dyad.

        Stephenson decomposes into a single triad (RRRRRR), not dyads,
        so it won't match the four-bar's (RRR,) as a decomposition prefix.
        Only Watt (which decomposes into RRR dyads) should connect.
        """
        catalog = load_catalog()
        graph = build_neighborhood_graph(catalog)
        watt_neighbors = graph.get("watt", [])
        remove_targets = [
            n.target_id for n in watt_neighbors if n.operation == "remove_dyad"
        ]
        assert "four-bar" in remove_targets

    def test_swap_variant_within_family(self) -> None:
        """Watt and Stephenson should be swap_variant neighbors."""
        catalog = load_catalog()
        graph = build_neighborhood_graph(catalog)
        watt_neighbors = graph.get("watt", [])
        swap_targets = [
            n.target_id for n in watt_neighbors if n.operation == "swap_variant"
        ]
        assert "stephenson" in swap_targets

    def test_neighbor_objects_well_formed(self) -> None:
        """All neighbor objects should have required fields."""
        catalog = load_catalog()
        graph = build_neighborhood_graph(catalog)
        for topo_id, neighbors in graph.items():
            for n in neighbors:
                assert isinstance(n, TopologyNeighbor)
                assert n.target_id in catalog
                assert n.operation in (
                    "add_dyad", "remove_dyad", "swap_variant", "restructure"
                )
                assert len(n.description) > 0


class TestTopologyNeighbors:
    """Tests for topology_neighbors function."""

    def test_returns_list(self) -> None:
        catalog = load_catalog()
        result = topology_neighbors("four-bar", catalog)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_unknown_topology_returns_empty(self) -> None:
        catalog = load_catalog()
        result = topology_neighbors("nonexistent", catalog)
        assert result == []


class TestTopologyDistance:
    """Tests for topology_distance."""

    def test_self_distance_is_zero(self) -> None:
        catalog = load_catalog()
        assert topology_distance("four-bar", "four-bar", catalog) == 0

    def test_four_bar_to_six_bar_is_one(self) -> None:
        """Four-bar to Watt should be distance 1 (add_dyad)."""
        catalog = load_catalog()
        d = topology_distance("four-bar", "watt", catalog)
        assert d == 1

    def test_graph_is_connected(self) -> None:
        """All topologies should be reachable from four-bar."""
        catalog = load_catalog()
        neighborhood = build_neighborhood_graph(catalog)
        for entry in catalog:
            d = topology_distance("four-bar", entry.id, catalog, neighborhood)
            assert d >= 0, f"four-bar cannot reach {entry.id}"

    def test_unknown_returns_negative(self) -> None:
        catalog = load_catalog()
        assert topology_distance("four-bar", "nonexistent", catalog) == -1
