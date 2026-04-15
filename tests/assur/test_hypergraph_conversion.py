"""Tests for assur/hypergraph_conversion.py (topology only)."""

import unittest

from pylinkage.assur.graph import Edge as AssurEdge
from pylinkage.assur.graph import JointType, LinkageGraph, NodeRole
from pylinkage.assur.graph import Node as AssurNode
from pylinkage.assur.hypergraph_conversion import from_hypergraph, to_hypergraph
from pylinkage._types import JointType as HyperJointType
from pylinkage._types import NodeRole as HyperNodeRole
from pylinkage.hypergraph.core import Edge as HyperEdge
from pylinkage.hypergraph.core import Node as HyperNode
from pylinkage.hypergraph.graph import HypergraphLinkage


class TestFromHypergraph(unittest.TestCase):
    """Tests for from_hypergraph conversion."""

    def test_simple_hypergraph_to_assur(self):
        """Test converting a simple hypergraph to Assur graph."""
        hg = HypergraphLinkage(name="Test")

        # Add nodes (topology only - no positions)
        hg.add_node(
            HyperNode(
                id="ground",
                role=HyperNodeRole.GROUND,
                joint_type=HyperJointType.REVOLUTE,
            )
        )
        hg.add_node(
            HyperNode(
                id="driver",
                role=HyperNodeRole.DRIVER,
                joint_type=HyperJointType.REVOLUTE,
            )
        )

        # Add edge (topology only - no distance)
        hg.add_edge(HyperEdge(id="e1", source="ground", target="driver"))

        # Convert
        assur = from_hypergraph(hg)

        self.assertEqual(assur.name, "Test")
        self.assertEqual(len(assur.nodes), 2)
        self.assertEqual(len(assur.edges), 1)

        # Check node types converted correctly
        self.assertEqual(assur.nodes["ground"].role, NodeRole.GROUND)
        self.assertEqual(assur.nodes["driver"].role, NodeRole.DRIVER)

    def test_hypergraph_with_driven_node(self):
        """Test converting hypergraph with driven node."""
        hg = HypergraphLinkage(name="FourBar")

        hg.add_node(HyperNode(id="g1", role=HyperNodeRole.GROUND))
        hg.add_node(HyperNode(id="g2", role=HyperNodeRole.GROUND))
        hg.add_node(HyperNode(id="crank", role=HyperNodeRole.DRIVER))
        hg.add_node(HyperNode(id="coupler", role=HyperNodeRole.DRIVEN))

        hg.add_edge(HyperEdge(id="e1", source="g1", target="crank"))
        hg.add_edge(HyperEdge(id="e2", source="crank", target="coupler"))
        hg.add_edge(HyperEdge(id="e3", source="g2", target="coupler"))

        assur = from_hypergraph(hg)

        self.assertEqual(len(assur.nodes), 4)
        self.assertEqual(assur.nodes["coupler"].role, NodeRole.DRIVEN)

    def test_hypergraph_with_prismatic_joint(self):
        """Test converting hypergraph with prismatic joint."""
        hg = HypergraphLinkage(name="Slider")

        hg.add_node(HyperNode(id="ground", role=HyperNodeRole.GROUND))
        hg.add_node(
            HyperNode(
                id="slider",
                role=HyperNodeRole.DRIVEN,
                joint_type=HyperJointType.PRISMATIC,
            )
        )

        hg.add_edge(HyperEdge(id="e1", source="ground", target="slider"))

        assur = from_hypergraph(hg)

        self.assertEqual(assur.nodes["slider"].joint_type, JointType.PRISMATIC)


class TestToHypergraph(unittest.TestCase):
    """Tests for to_hypergraph conversion."""

    def test_simple_assur_to_hypergraph(self):
        """Test converting a simple Assur graph to hypergraph."""
        assur = LinkageGraph(name="Test")

        assur.add_node(AssurNode(id="ground", role=NodeRole.GROUND))
        assur.add_node(AssurNode(id="driver", role=NodeRole.DRIVER))

        assur.add_edge(AssurEdge(id="e1", source="ground", target="driver"))

        hg = to_hypergraph(assur)

        self.assertEqual(hg.name, "Test")
        self.assertEqual(len(hg.nodes), 2)
        self.assertEqual(len(hg.edges), 1)

        # Check node roles converted correctly
        self.assertEqual(hg.nodes["ground"].role, HyperNodeRole.GROUND)
        self.assertEqual(hg.nodes["driver"].role, HyperNodeRole.DRIVER)

    def test_assur_with_driven_node(self):
        """Test converting Assur graph with driven node."""
        assur = LinkageGraph(name="FourBar")

        assur.add_node(AssurNode(id="g1", role=NodeRole.GROUND))
        assur.add_node(AssurNode(id="g2", role=NodeRole.GROUND))
        assur.add_node(AssurNode(id="crank", role=NodeRole.DRIVER))
        assur.add_node(AssurNode(id="coupler", role=NodeRole.DRIVEN))

        assur.add_edge(AssurEdge(id="e1", source="g1", target="crank"))
        assur.add_edge(AssurEdge(id="e2", source="crank", target="coupler"))
        assur.add_edge(AssurEdge(id="e3", source="g2", target="coupler"))

        hg = to_hypergraph(assur)

        self.assertEqual(len(hg.nodes), 4)
        self.assertEqual(hg.nodes["coupler"].role, HyperNodeRole.DRIVEN)

    def test_assur_with_prismatic_joint(self):
        """Test converting Assur graph with prismatic joint."""
        assur = LinkageGraph(name="Slider")

        assur.add_node(AssurNode(id="ground", role=NodeRole.GROUND))
        assur.add_node(
            AssurNode(
                id="slider",
                role=NodeRole.DRIVEN,
                joint_type=JointType.PRISMATIC,
            )
        )

        assur.add_edge(AssurEdge(id="e1", source="ground", target="slider"))

        hg = to_hypergraph(assur)

        self.assertEqual(hg.nodes["slider"].joint_type, HyperJointType.PRISMATIC)


class TestRoundTrip(unittest.TestCase):
    """Test round-trip conversions."""

    def test_hypergraph_to_assur_to_hypergraph(self):
        """Test hypergraph -> assur -> hypergraph preserves structure."""
        original = HypergraphLinkage(name="RoundTrip")

        original.add_node(HyperNode(id="g1", role=HyperNodeRole.GROUND))
        original.add_node(HyperNode(id="d1", role=HyperNodeRole.DRIVER))
        original.add_node(HyperNode(id="driven", role=HyperNodeRole.DRIVEN))

        original.add_edge(HyperEdge(id="e1", source="g1", target="d1"))
        original.add_edge(HyperEdge(id="e2", source="d1", target="driven"))

        # Convert to Assur
        assur = from_hypergraph(original)

        # Convert back to hypergraph
        restored = to_hypergraph(assur)

        # Verify structure preserved
        self.assertEqual(restored.name, original.name)
        self.assertEqual(len(restored.nodes), len(original.nodes))

    def test_assur_to_hypergraph_to_assur(self):
        """Test assur -> hypergraph -> assur preserves structure."""
        original = LinkageGraph(name="RoundTrip")

        original.add_node(AssurNode(id="g1", role=NodeRole.GROUND))
        original.add_node(AssurNode(id="d1", role=NodeRole.DRIVER))
        original.add_node(AssurNode(id="driven", role=NodeRole.DRIVEN))

        original.add_edge(AssurEdge(id="e1", source="g1", target="d1"))
        original.add_edge(AssurEdge(id="e2", source="d1", target="driven"))

        # Convert to hypergraph
        hg = to_hypergraph(original)

        # Convert back to assur
        restored = from_hypergraph(hg)

        # Verify structure preserved
        self.assertEqual(restored.name, original.name)
        self.assertEqual(len(restored.nodes), len(original.nodes))


if __name__ == "__main__":
    unittest.main()
