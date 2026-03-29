"""Tests for systematic topology enumeration."""

import pytest

from pylinkage.topology.analysis import compute_dof
from pylinkage.topology.enumeration import enumerate_all, enumerate_topologies
from pylinkage.topology.isomorphism import are_isomorphic


class TestEnumerateTopologies:
    """Tests for enumerate_topologies."""

    def test_four_bar_count(self):
        """There is exactly 1 four-bar topology."""
        topos = enumerate_topologies(4)
        assert len(topos) == 1

    def test_four_bar_dof(self):
        """The four-bar topology has DOF=1."""
        topos = enumerate_topologies(4)
        assert compute_dof(topos[0]) == 1

    def test_four_bar_structure(self):
        """The four-bar has 4 joints, 3 edges, no hyperedges."""
        g = enumerate_topologies(4)[0]
        assert len(g.nodes) == 4
        assert len(g.edges) == 3
        assert len(g.hyperedges) == 0

    def test_six_bar_count(self):
        """There are exactly 2 six-bar topologies (Watt, Stephenson)."""
        topos = enumerate_topologies(6)
        assert len(topos) == 2

    def test_six_bar_dof(self):
        """Both six-bar topologies have DOF=1."""
        for g in enumerate_topologies(6):
            assert compute_dof(g) == 1

    def test_six_bar_not_isomorphic(self):
        """The two six-bar topologies are distinct."""
        topos = enumerate_topologies(6)
        assert not are_isomorphic(topos[0], topos[1])

    def test_six_bar_have_ternary_links(self):
        """Both six-bar topologies have ternary links (hyperedges)."""
        for g in enumerate_topologies(6):
            assert len(g.hyperedges) > 0

    def test_eight_bar_count(self):
        """There are exactly 16 eight-bar topologies (Mruthyunjaya 1984)."""
        topos = enumerate_topologies(8)
        assert len(topos) == 16

    def test_eight_bar_dof(self):
        """All eight-bar topologies have DOF=1."""
        for g in enumerate_topologies(8):
            assert compute_dof(g) == 1

    def test_eight_bar_pairwise_distinct(self):
        """All 16 eight-bar topologies are pairwise non-isomorphic."""
        topos = enumerate_topologies(8)
        for i in range(len(topos)):
            for j in range(i + 1, len(topos)):
                assert not are_isomorphic(topos[i], topos[j]), (
                    f"Topologies {i} and {j} are isomorphic"
                )

    def test_invalid_link_count_raises(self):
        """Odd link counts that don't yield integer joints raise ValueError."""
        with pytest.raises(ValueError):
            enumerate_topologies(5)

    def test_too_few_links_raises(self):
        """Fewer than 4 links raises ValueError."""
        with pytest.raises(ValueError):
            enumerate_topologies(3)


    def test_four_bar_decomposes_to_rrr_dyad(self):
        """The four-bar topology can be Assur-decomposed into 1 RRR dyad."""
        from pylinkage.assur.decomposition import decompose_assur_groups
        from pylinkage.assur.hypergraph_conversion import from_hypergraph

        g = enumerate_topologies(4)[0]
        ag = from_hypergraph(g)
        result = decompose_assur_groups(ag)
        assert len(result.groups) == 1
        assert result.groups[0].joint_signature == "RRR"

    def test_six_bars_decompose(self):
        """Both six-bar topologies can be Assur-decomposed."""
        from pylinkage.assur.decomposition import decompose_assur_groups
        from pylinkage.assur.hypergraph_conversion import from_hypergraph

        for g in enumerate_topologies(6):
            ag = from_hypergraph(g)
            result = decompose_assur_groups(ag)
            assert len(result.groups) >= 1

    def test_four_bar_has_correct_roles(self):
        """The four-bar has 2 GROUND nodes, 1 DRIVER, 1 DRIVEN."""
        from pylinkage.hypergraph import NodeRole

        g = enumerate_topologies(4)[0]
        roles = [n.role for n in g.nodes.values()]
        assert roles.count(NodeRole.GROUND) == 2
        assert roles.count(NodeRole.DRIVER) == 1
        assert roles.count(NodeRole.DRIVEN) == 1


class TestEnumerateAll:
    """Tests for enumerate_all."""

    def test_default_max_links(self):
        """enumerate_all returns 4, 6, and 8 link topologies."""
        result = enumerate_all()
        assert set(result.keys()) == {4, 6, 8}

    def test_total_count(self):
        """Total count is 1 + 2 + 16 = 19."""
        result = enumerate_all()
        total = sum(len(v) for v in result.values())
        assert total == 19

    def test_max_links_4(self):
        """With max_links=4, only four-bars are returned."""
        result = enumerate_all(max_links=4)
        assert set(result.keys()) == {4}
        assert len(result[4]) == 1

    def test_max_links_6(self):
        """With max_links=6, four-bars and six-bars are returned."""
        result = enumerate_all(max_links=6)
        assert set(result.keys()) == {4, 6}
        assert len(result[4]) == 1
        assert len(result[6]) == 2
