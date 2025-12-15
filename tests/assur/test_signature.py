"""Tests for Assur signature parsing and hypergraph generation."""

import pytest

from pylinkage._types import JointType
from pylinkage.assur import DyadPRR, DyadRPR, DyadRRP, DyadRRR
from pylinkage.assur.signature import (
    AssurGroupClass,
    parse_signature,
    signature_to_group_class,
    signature_to_hypergraph,
)


class TestParseSignature:
    """Tests for signature parsing."""

    def test_parse_rrr(self):
        """Test parsing RRR signature."""
        sig = parse_signature("RRR")
        assert sig.joints == (JointType.REVOLUTE,) * 3
        assert sig.group_class == AssurGroupClass.DYAD

    def test_parse_with_separators(self):
        """Test parsing with underscores."""
        sig = parse_signature("R_R_R")
        assert sig.joints == (JointType.REVOLUTE,) * 3
        assert sig.group_class == AssurGroupClass.DYAD

    def test_parse_with_spaces(self):
        """Test parsing with spaces (ignored)."""
        sig = parse_signature("R R R")
        assert sig.joints == (JointType.REVOLUTE,) * 3

    def test_parse_mixed_case(self):
        """Test case insensitivity."""
        sig = parse_signature("rRr")
        assert sig.joints == (JointType.REVOLUTE,) * 3

    def test_parse_prismatic_p(self):
        """Test P maps to PRISMATIC."""
        sig = parse_signature("RPR")
        assert sig.joints[1] == JointType.PRISMATIC

    def test_parse_prismatic_t(self):
        """Test T maps to PRISMATIC (alias)."""
        sig = parse_signature("RTR")
        assert sig.joints[1] == JointType.PRISMATIC

    def test_parse_p_and_t_equivalent(self):
        """Test P and T produce equivalent signatures."""
        sig_p = parse_signature("RPR")
        sig_t = parse_signature("RTR")
        assert sig_p.joints == sig_t.joints
        assert sig_p.canonical_form == sig_t.canonical_form

    def test_parse_all_dyad_variants(self):
        """Test all valid dyad signatures."""
        for sig_str in ["RRR", "RRP", "RPR", "PRR", "RPP", "PRP", "PPR", "PPP"]:
            sig = parse_signature(sig_str)
            assert sig.group_class == AssurGroupClass.DYAD
            assert len(sig.joints) == 3

    def test_parse_invalid_char_raises(self):
        """Test invalid character raises ValueError."""
        with pytest.raises(ValueError, match="Invalid character"):
            parse_signature("RXR")

    def test_parse_empty_raises(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_signature("")

    def test_parse_only_separators_raises(self):
        """Test only separators raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_signature("___")

    def test_parse_wrong_length_raises(self):
        """Test wrong joint count raises ValueError."""
        with pytest.raises(ValueError, match="Invalid joint count"):
            parse_signature("RR")  # Only 2 joints

        with pytest.raises(ValueError, match="Invalid joint count"):
            parse_signature("RRRR")  # 4 joints (invalid)

        with pytest.raises(ValueError, match="Invalid joint count"):
            parse_signature("RRRRR")  # 5 joints (invalid)

    def test_parse_preserves_raw_string(self):
        """Test raw string is preserved."""
        sig = parse_signature("R_R_R")
        assert sig.raw_string == "R_R_R"


class TestAssurSignature:
    """Tests for AssurSignature properties."""

    def test_canonical_form_uses_p(self):
        """Test canonical form uses P not T."""
        sig = parse_signature("RTR")
        assert sig.canonical_form == "RPR"

    def test_canonical_form_all_p(self):
        """Test canonical form with all prismatic."""
        sig = parse_signature("TTT")
        assert sig.canonical_form == "PPP"

    def test_num_internal_nodes_dyad(self):
        """Test dyad has 1 internal node."""
        sig = parse_signature("RRR")
        assert sig.num_internal_nodes == 1

    def test_num_anchor_nodes_dyad(self):
        """Test dyad has 2 anchor nodes."""
        sig = parse_signature("RRR")
        assert sig.num_anchor_nodes == 2

    def test_num_links_dyad(self):
        """Test dyad has 2 links."""
        sig = parse_signature("RRR")
        assert sig.num_links == 2

    def test_signature_hashable(self):
        """Test signatures can be used as dict keys."""
        sig1 = parse_signature("RRR")
        sig2 = parse_signature("RRR")
        d = {sig1: "value"}
        assert d[sig2] == "value"

    def test_signature_frozen(self):
        """Test signature is immutable."""
        sig = parse_signature("RRR")
        with pytest.raises(AttributeError):
            sig.joints = (JointType.PRISMATIC,) * 3  # type: ignore

    def test_signature_equality(self):
        """Test signature equality by content."""
        sig1 = parse_signature("RRR")
        sig2 = parse_signature("R_R_R")  # Same joints, different raw string
        # Frozen dataclass compares all fields
        assert sig1.joints == sig2.joints
        assert sig1.group_class == sig2.group_class


class TestSignatureToHypergraph:
    """Tests for hypergraph generation."""

    def test_rrr_generates_correct_node_count(self):
        """Test RRR generates 3 nodes."""
        graph = signature_to_hypergraph("RRR")
        assert len(graph.nodes) == 3

    def test_rrr_generates_correct_edge_count(self):
        """Test RRR generates 2 edges."""
        graph = signature_to_hypergraph("RRR")
        assert len(graph.edges) == 2

    def test_all_edges_have_no_distance(self):
        """Test generated graph has no distance constraints."""
        graph = signature_to_hypergraph("RRR")
        for edge in graph.edges.values():
            assert edge.distance is None

    def test_all_nodes_have_no_position(self):
        """Test generated nodes have no positions."""
        graph = signature_to_hypergraph("RRR")
        for node in graph.nodes.values():
            assert node.position == (None, None)

    def test_joint_types_match_signature_rrr(self):
        """Test node joint types match RRR signature."""
        graph = signature_to_hypergraph("RRR")
        assert graph.nodes["anchor_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["internal_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["anchor_1"].joint_type == JointType.REVOLUTE

    def test_joint_types_match_signature_rpr(self):
        """Test node joint types match RPR signature."""
        graph = signature_to_hypergraph("RPR")
        assert graph.nodes["anchor_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["internal_0"].joint_type == JointType.PRISMATIC
        assert graph.nodes["anchor_1"].joint_type == JointType.REVOLUTE

    def test_joint_types_match_signature_prr(self):
        """Test node joint types match PRR signature."""
        graph = signature_to_hypergraph("PRR")
        assert graph.nodes["anchor_0"].joint_type == JointType.PRISMATIC
        assert graph.nodes["internal_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["anchor_1"].joint_type == JointType.REVOLUTE

    def test_joint_types_match_signature_rrp(self):
        """Test node joint types match RRP signature."""
        graph = signature_to_hypergraph("RRP")
        assert graph.nodes["anchor_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["internal_0"].joint_type == JointType.REVOLUTE
        assert graph.nodes["anchor_1"].joint_type == JointType.PRISMATIC

    def test_prefix_applies_to_node_ids(self):
        """Test prefix is applied to node IDs."""
        graph = signature_to_hypergraph("RRR", prefix="leg_")
        assert "leg_anchor_0" in graph.nodes
        assert "leg_anchor_1" in graph.nodes
        assert "leg_internal_0" in graph.nodes
        assert "anchor_0" not in graph.nodes

    def test_prefix_applies_to_edge_ids(self):
        """Test prefix is applied to edge IDs."""
        graph = signature_to_hypergraph("RRR", prefix="leg_")
        assert "leg_link_0" in graph.edges
        assert "leg_link_1" in graph.edges
        assert "link_0" not in graph.edges

    def test_name_parameter(self):
        """Test custom name is used."""
        graph = signature_to_hypergraph("RRR", name="MyDyad")
        assert graph.name == "MyDyad"

    def test_default_name(self):
        """Test default name includes signature."""
        graph = signature_to_hypergraph("RRR")
        assert "RRR" in graph.name

    def test_accepts_signature_object(self):
        """Test accepts AssurSignature directly."""
        sig = parse_signature("RRR")
        graph = signature_to_hypergraph(sig)
        assert len(graph.nodes) == 3

    def test_edge_connectivity(self):
        """Test edges connect correct nodes."""
        graph = signature_to_hypergraph("RRR")
        link0 = graph.edges["link_0"]
        link1 = graph.edges["link_1"]

        # link_0 connects anchor_0 to internal_0
        assert link0.source == "anchor_0"
        assert link0.target == "internal_0"

        # link_1 connects internal_0 to anchor_1
        assert link1.source == "internal_0"
        assert link1.target == "anchor_1"

    def test_triad_raises_not_implemented(self):
        """Test triad generation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="TRIAD"):
            signature_to_hypergraph("RRRRRR")

    def test_tetrad_raises_not_implemented(self):
        """Test tetrad generation raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="TETRAD"):
            signature_to_hypergraph("RRRRRRRRR")


class TestSignatureToGroupClass:
    """Tests for bridging to existing AssurGroup classes."""

    def test_rrr_maps_to_dyad_rrr(self):
        """Test RRR signature maps to DyadRRR."""
        sig = parse_signature("RRR")
        cls = signature_to_group_class(sig)
        assert cls is DyadRRR

    def test_rrp_maps_to_dyad_rrp(self):
        """Test RRP signature maps to DyadRRP."""
        sig = parse_signature("RRP")
        cls = signature_to_group_class(sig)
        assert cls is DyadRRP

    def test_rpr_maps_to_dyad_rpr(self):
        """Test RPR signature maps to DyadRPR."""
        sig = parse_signature("RPR")
        cls = signature_to_group_class(sig)
        assert cls is DyadRPR

    def test_prr_maps_to_dyad_prr(self):
        """Test PRR signature maps to DyadPRR."""
        sig = parse_signature("PRR")
        cls = signature_to_group_class(sig)
        assert cls is DyadPRR

    def test_accepts_string(self):
        """Test accepts string directly."""
        cls = signature_to_group_class("RRR")
        assert cls is DyadRRR

    def test_t_alias_works(self):
        """Test T alias maps correctly."""
        cls = signature_to_group_class("RRT")  # Same as RRP
        assert cls is DyadRRP

    def test_unknown_returns_none(self):
        """Test signature without corresponding class returns None."""
        # PPP has no corresponding class in DYAD_TYPES
        cls = signature_to_group_class("PPP")
        assert cls is None


class TestAssurGroupClass:
    """Tests for AssurGroupClass enum."""

    def test_dyad_value(self):
        """Test DYAD has value 1."""
        assert AssurGroupClass.DYAD == 1

    def test_triad_value(self):
        """Test TRIAD has value 2."""
        assert AssurGroupClass.TRIAD == 2

    def test_tetrad_value(self):
        """Test TETRAD has value 3."""
        assert AssurGroupClass.TETRAD == 3
