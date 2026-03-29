"""Tests for the topology catalog."""

import tempfile
from pathlib import Path

from pylinkage.topology.analysis import compute_dof
from pylinkage.topology.catalog import (
    CatalogEntry,
    TopologyCatalog,
    generate_catalog,
    load_catalog,
)
from pylinkage.topology.isomorphism import canonical_form


class TestLoadCatalog:
    """Tests for loading the built-in catalog."""

    def test_load_succeeds(self):
        catalog = load_catalog()
        assert isinstance(catalog, TopologyCatalog)

    def test_total_count(self):
        """Built-in catalog has 19 entries (1 + 2 + 16)."""
        catalog = load_catalog()
        assert len(catalog) == 19

    def test_four_bar_exists(self):
        catalog = load_catalog()
        assert "four-bar" in catalog

    def test_watt_exists(self):
        catalog = load_catalog()
        assert "watt" in catalog

    def test_stephenson_exists(self):
        catalog = load_catalog()
        assert "stephenson" in catalog


class TestCatalogEntry:
    """Tests for CatalogEntry."""

    def test_four_bar_metadata(self):
        entry = load_catalog().get("four-bar")
        assert entry is not None
        assert entry.num_links == 4
        assert entry.num_joints == 4
        assert entry.dof == 1
        assert entry.family == "four-bar"

    def test_to_graph_returns_valid_graph(self):
        entry = load_catalog().get("four-bar")
        g = entry.to_graph()
        assert len(g.nodes) == 4
        assert compute_dof(g) == 1

    def test_all_entries_have_dof_1(self):
        for entry in load_catalog():
            g = entry.to_graph()
            assert compute_dof(g) == 1, f"{entry.id} has DOF={compute_dof(g)}"

    def test_all_entries_have_unique_graphs(self):
        catalog = load_catalog()
        forms = set()
        for entry in catalog:
            form = canonical_form(entry.to_graph())
            assert form not in forms, f"Duplicate graph for {entry.id}"
            forms.add(form)


class TestCatalogQueries:
    """Tests for catalog query methods."""

    def test_by_num_links_4(self):
        entries = load_catalog().by_num_links(4)
        assert len(entries) == 1
        assert entries[0].id == "four-bar"

    def test_by_num_links_6(self):
        entries = load_catalog().by_num_links(6)
        assert len(entries) == 2

    def test_by_num_links_8(self):
        entries = load_catalog().by_num_links(8)
        assert len(entries) == 16

    def test_by_family(self):
        entries = load_catalog().by_family("six-bar")
        assert len(entries) == 2
        assert all(e.family == "six-bar" for e in entries)

    def test_compatible_topologies(self):
        entries = load_catalog().compatible_topologies(max_links=6)
        assert len(entries) == 3  # 1 four-bar + 2 six-bars

    def test_all_graphs(self):
        graphs = load_catalog().all_graphs()
        assert len(graphs) == 19
        for g in graphs:
            assert compute_dof(g) == 1

    def test_iteration(self):
        entries = list(load_catalog())
        assert len(entries) == 19
        assert all(isinstance(e, CatalogEntry) for e in entries)


class TestCatalogSerialization:
    """Tests for JSON round-trip."""

    def test_json_round_trip(self):
        catalog = load_catalog()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_catalog.json"
            catalog.to_json(path)
            loaded = TopologyCatalog.from_json(path)
            assert len(loaded) == len(catalog)
            for entry in catalog:
                loaded_entry = loaded.get(entry.id)
                assert loaded_entry is not None
                assert loaded_entry.num_links == entry.num_links
                assert loaded_entry.num_joints == entry.num_joints
                assert loaded_entry.link_assortment == entry.link_assortment


class TestGenerateCatalog:
    """Tests for catalog generation."""

    def test_generate_four_bars_only(self):
        catalog = generate_catalog(max_links=4)
        assert len(catalog) == 1
        assert "four-bar" in catalog

    def test_generate_up_to_six(self):
        catalog = generate_catalog(max_links=6)
        assert len(catalog) == 3
