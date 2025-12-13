"""Tests for the conversion module."""

import pylinkage as pl
from pylinkage.assur import (
    Edge,
    LinkageGraph,
    Node,
    NodeRole,
    graph_to_linkage,
    linkage_to_graph,
)


class TestLinkageToGraph:
    """Tests for the linkage_to_graph function."""

    def test_convert_simple_crank(self):
        """Test converting a simple crank to graph."""
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="crank")
        linkage = pl.Linkage(joints=[crank], order=[crank])

        graph = linkage_to_graph(linkage)

        # Should have 2 nodes: ground anchor and crank
        assert len(graph.nodes) == 2

        # Check crank node
        crank_node = graph.nodes.get("crank")
        assert crank_node is not None
        assert crank_node.role == NodeRole.DRIVER
        assert crank_node.angle == 0.31

        # Check edge exists
        assert len(graph.edges) == 1

    def test_convert_four_bar(self):
        """Test converting a four-bar linkage to graph."""
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="B")
        pin = pl.Revolute(
            3, 2,
            joint0=crank,
            joint1=(3, 0),
            distance0=3,
            distance1=1,
            name="C"
        )
        linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin], name="Four-bar")

        graph = linkage_to_graph(linkage)

        assert graph.name == "Four-bar"

        # Should have nodes for: crank anchor, crank, pin anchor, pin
        # Note: tuple anchors become implicit ground nodes
        assert len(graph.nodes) >= 2

        # Check edges
        assert len(graph.edges) >= 2

    def test_convert_with_explicit_static(self):
        """Test converting linkage with explicit Static joints."""
        anchor0 = pl.Static(0, 0, name="anchor0")
        anchor1 = pl.Static(3, 0, name="anchor1")
        crank = pl.Crank(0, 1, joint0=anchor0, angle=0.31, distance=1, name="crank")
        pin = pl.Revolute(
            3, 2,
            joint0=crank,
            joint1=anchor1,
            distance0=3,
            distance1=1,
            name="pin"
        )
        linkage = pl.Linkage(
            joints=[anchor0, anchor1, crank, pin],
            order=[crank, pin]
        )

        graph = linkage_to_graph(linkage)

        # Check Static joints become GROUND nodes
        assert graph.nodes["anchor0"].role == NodeRole.GROUND
        assert graph.nodes["anchor1"].role == NodeRole.GROUND

        # Check Crank becomes DRIVER
        assert graph.nodes["crank"].role == NodeRole.DRIVER

        # Check Revolute becomes DRIVEN
        assert graph.nodes["pin"].role == NodeRole.DRIVEN


class TestGraphToLinkage:
    """Tests for the graph_to_linkage function."""

    def test_convert_simple_graph(self):
        """Test converting a simple graph to linkage."""
        graph = LinkageGraph(name="Four-bar")

        # Ground points
        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))

        # Driver
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))

        # Driven
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))

        # Edges
        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        graph.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        linkage = graph_to_linkage(graph)

        assert linkage.name == "Four-bar"

        # Check we have the right joint types
        joint_names = [j.name for j in linkage.joints]
        assert "A" in joint_names or "D" in joint_names  # At least one ground

    def test_roundtrip_conversion(self):
        """Test that linkage -> graph -> linkage preserves structure."""
        # Create original linkage
        crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="B")
        pin = pl.Revolute(
            3, 2,
            joint0=crank,
            joint1=(3, 0),
            distance0=3,
            distance1=1,
            name="C"
        )
        original = pl.Linkage(joints=[crank, pin], order=[crank, pin], name="Test")

        # Convert to graph
        graph = linkage_to_graph(original)

        # Convert back to linkage
        restored = graph_to_linkage(graph)

        # Check name preserved
        assert restored.name == original.name

        # Check joint count (may differ due to implicit anchors becoming explicit)
        assert len(restored.joints) >= len(original.joints)

    def test_converted_linkage_can_simulate(self):
        """Test that converted linkage can run simulation."""
        graph = LinkageGraph(name="Four-bar")

        graph.add_node(Node("A", role=NodeRole.GROUND, position=(0.0, 0.0)))
        graph.add_node(Node("D", role=NodeRole.GROUND, position=(3.0, 0.0)))
        graph.add_node(Node("B", role=NodeRole.DRIVER, position=(0.0, 1.0), angle=0.31))
        graph.add_node(Node("C", role=NodeRole.DRIVEN, position=(3.0, 2.0)))

        graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))
        graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))
        graph.add_edge(Edge("CD", source="C", target="D", distance=1.0))

        linkage = graph_to_linkage(graph)

        # Should be able to step without error
        step_count = 0
        for coords in linkage.step(iterations=10):
            step_count += 1
            assert len(coords) == len(linkage.joints)

        assert step_count == 10
