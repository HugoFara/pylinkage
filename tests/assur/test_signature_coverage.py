"""Additional tests for signature.py to increase coverage.

Covers: AssurSignature validation, parse_isomer_signature, isomer_to_canonical,
signature_to_group_class for triad and unknown, triad hypergraph generation,
and edge cases.
"""

import pytest

from pylinkage._types import JointType
from pylinkage.assur import Triad
from pylinkage.assur.groups import DyadPP
from pylinkage.assur.signature import (
    _JOINTS_PER_CLASS,
    AssurGroupClass,
    AssurSignature,
    isomer_to_canonical,
    parse_isomer_signature,
    parse_signature,
    signature_to_group_class,
    signature_to_hypergraph,
)

# ---------------------------------------------------------------------------
# AssurSignature validation
# ---------------------------------------------------------------------------

class TestAssurSignatureValidation:
    def test_wrong_joint_count_raises(self):
        """Constructing with wrong joint count for group class raises."""
        with pytest.raises(ValueError, match="requires"):
            AssurSignature(
                joints=(JointType.REVOLUTE, JointType.REVOLUTE),
                group_class=AssurGroupClass.DYAD,
                raw_string="RR",
            )

    def test_triad_needs_six_joints(self):
        with pytest.raises(ValueError, match="requires"):
            AssurSignature(
                joints=(JointType.REVOLUTE,) * 3,
                group_class=AssurGroupClass.TRIAD,
                raw_string="RRR",
            )

    def test_correct_joint_count_ok(self):
        sig = AssurSignature(
            joints=(JointType.REVOLUTE,) * 6,
            group_class=AssurGroupClass.TRIAD,
            raw_string="RRRRRR",
        )
        assert sig.group_class == AssurGroupClass.TRIAD


# ---------------------------------------------------------------------------
# AssurSignature properties for triad
# ---------------------------------------------------------------------------

class TestAssurSignatureTriadProperties:
    def test_triad_num_internal_nodes(self):
        sig = parse_signature("RRRRRR")
        assert sig.num_internal_nodes == 2

    def test_triad_num_anchor_nodes(self):
        sig = parse_signature("RRRRRR")
        assert sig.num_anchor_nodes == 3

    def test_triad_num_links(self):
        sig = parse_signature("RRRRRR")
        assert sig.num_links == 4

    def test_triad_canonical_form(self):
        sig = parse_signature("RRRRRR")
        assert sig.canonical_form == "RRRRRR"

    def test_triad_mixed_canonical(self):
        sig = parse_signature("RRRRRP")
        assert sig.canonical_form == "RRRRRP"


# ---------------------------------------------------------------------------
# parse_isomer_signature
# ---------------------------------------------------------------------------

class TestParseIsomerSignature:
    def test_basic_rrr(self):
        sig, roles = parse_isomer_signature("RRR")
        assert sig == "RRR"
        assert roles == ("revolute", "revolute", "revolute")

    def test_rt_underscore_r(self):
        sig, roles = parse_isomer_signature("RT_R")
        assert roles == ("revolute", "slider", "guide", "revolute")

    def test_t_underscore_r_underscore_t(self):
        sig, roles = parse_isomer_signature("T_R_T")
        assert roles == ("slider", "guide", "revolute", "guide", "slider")

    def test_p_treated_as_prismatic(self):
        sig, roles = parse_isomer_signature("RPR")
        assert roles == ("revolute", "prismatic", "revolute")

    def test_spaces_are_skipped(self):
        sig, roles = parse_isomer_signature("R R R")
        assert roles == ("revolute", "revolute", "revolute")

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid character"):
            parse_isomer_signature("RXR")

    def test_lowercase_normalized_to_upper(self):
        sig, roles = parse_isomer_signature("rrr")
        assert sig == "RRR"
        assert roles == ("revolute", "revolute", "revolute")


# ---------------------------------------------------------------------------
# isomer_to_canonical
# ---------------------------------------------------------------------------

class TestIsomerToCanonical:
    def test_rrr_unchanged(self):
        assert isomer_to_canonical("RRR") == "RRR"

    def test_rt_underscore_r(self):
        assert isomer_to_canonical("RT_R") == "RPR"

    def test_t_underscore_r_underscore_t(self):
        assert isomer_to_canonical("T_R_T") == "PRP"

    def test_t_without_following_guide(self):
        assert isomer_to_canonical("TRT") == "PRP"

    def test_underscore_t_pair(self):
        """Guide followed by slider -> P."""
        assert isomer_to_canonical("_TR") == "PR"

    def test_standalone_underscore_skipped(self):
        """Underscore not followed by T is skipped."""
        assert isomer_to_canonical("R_R") == "RR"

    def test_p_preserved(self):
        assert isomer_to_canonical("RPR") == "RPR"

    def test_spaces_ignored(self):
        assert isomer_to_canonical("R R R") == "RRR"

    def test_lowercase(self):
        assert isomer_to_canonical("rrr") == "RRR"

    def test_unknown_char_skipped(self):
        # Unknown characters just get skipped
        assert isomer_to_canonical("R!R") == "RR"


# ---------------------------------------------------------------------------
# signature_to_group_class for triad and edge cases
# ---------------------------------------------------------------------------

class TestSignatureToGroupClassExtended:
    def test_triad_returns_triad_class(self):
        cls = signature_to_group_class("RRRRRR")
        assert cls is Triad

    def test_pp_as_two_chars_raises(self):
        """PP (2 chars) is not a valid signature -- needs 3 for dyad."""
        with pytest.raises(ValueError, match="Invalid joint count"):
            signature_to_group_class("PP")

    def test_ppr_maps_to_dyad_pp(self):
        """PPR maps to DyadPP in DYAD_TYPES."""
        cls = signature_to_group_class("PPR")
        assert cls is DyadPP

    def test_prp_maps_to_dyad_pp(self):
        cls = signature_to_group_class("PRP")
        assert cls is DyadPP

    def test_tetrad_returns_none(self):
        """Tetrad (9 joints) has no specific class yet."""
        cls = signature_to_group_class("RRRRRRRRR")
        assert cls is None


# ---------------------------------------------------------------------------
# signature_to_hypergraph edge cases
# ---------------------------------------------------------------------------

class TestSignatureToHypergraphExtended:
    def test_triad_with_prefix(self):
        graph = signature_to_hypergraph("RRRRRR", prefix="t_")
        assert "t_anchor_0" in graph.nodes
        assert "t_internal_0" in graph.nodes
        assert "t_internal_1" in graph.nodes

    def test_triad_with_name(self):
        graph = signature_to_hypergraph("RRRRRR", name="MyTriad")
        assert graph.name == "MyTriad"

    def test_triad_default_name(self):
        graph = signature_to_hypergraph("RRRRRR")
        assert "RRRRRR" in graph.name

    def test_triad_edge_connectivity(self):
        graph = signature_to_hypergraph("RRRRRR")
        # link_0: anchor_0 -> internal_0
        assert graph.edges["link_0"].source == "anchor_0"
        assert graph.edges["link_0"].target == "internal_0"
        # link_1: internal_0 -> anchor_1
        assert graph.edges["link_1"].source == "internal_0"
        assert graph.edges["link_1"].target == "anchor_1"
        # link_2: internal_0 -> internal_1
        assert graph.edges["link_2"].source == "internal_0"
        assert graph.edges["link_2"].target == "internal_1"
        # link_3: internal_1 -> anchor_2
        assert graph.edges["link_3"].source == "internal_1"
        assert graph.edges["link_3"].target == "anchor_2"

    def test_triad_mixed_joint_types(self):
        graph = signature_to_hypergraph("RRPRRR")
        assert graph.nodes["anchor_2"].joint_type == JointType.PRISMATIC

    def test_dyad_no_prefix(self):
        graph = signature_to_hypergraph("RRR", prefix="")
        assert "anchor_0" in graph.nodes

    def test_dyad_all_prismatic(self):
        graph = signature_to_hypergraph("PPP")
        for nid in ("anchor_0", "internal_0", "anchor_1"):
            assert graph.nodes[nid].joint_type == JointType.PRISMATIC


# ---------------------------------------------------------------------------
# _JOINTS_PER_CLASS constant
# ---------------------------------------------------------------------------

class TestJointsPerClass:
    def test_dyad_has_3(self):
        assert _JOINTS_PER_CLASS[AssurGroupClass.DYAD] == 3

    def test_triad_has_6(self):
        assert _JOINTS_PER_CLASS[AssurGroupClass.TRIAD] == 6

    def test_tetrad_has_9(self):
        assert _JOINTS_PER_CLASS[AssurGroupClass.TETRAD] == 9
