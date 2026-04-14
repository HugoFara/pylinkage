"""Conversion from a joint-based Linkage to the Assur graph representation.

This module provides ``linkage_to_graph()`` — extracting a graph + a
``Dimensions`` object from an existing ``Linkage``. For conversion in
the other direction, use ``graph_to_mechanism()`` from
``assur.mechanism_conversion``.
"""

from math import tau
from typing import TYPE_CHECKING

from ..dimensions import Dimensions, DriverAngle
from ._types import JointType, NodeId, NodeRole
from .graph import Edge, LinkageGraph, Node

if TYPE_CHECKING:
    from ..linkage.linkage import Linkage


def linkage_to_graph(linkage: "Linkage") -> tuple[LinkageGraph, Dimensions]:
    """Convert an existing Linkage to graph representation and Dimensions.

    Maps existing joint types to graph nodes:
    - Static -> GROUND node
    - Crank -> DRIVER node + edge to its anchor
    - Revolute -> DRIVEN node (RRR dyad internal)
    - Linear -> DRIVEN node (RRP dyad internal)
    - Fixed -> DRIVEN node (deterministic constraint)

    Args:
        linkage: The Linkage to convert.

    Returns:
        Tuple of (LinkageGraph, Dimensions) - topology and dimensional data.

    Example:
        >>> linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin])
        >>> graph, dims = linkage_to_graph(linkage)
        >>> print(f"Nodes: {list(graph.nodes.keys())}")
    """
    from ..joints.crank import Crank
    from ..joints.fixed import Fixed
    from ..joints.joint import _StaticBase
    from ..joints.prismatic import Prismatic
    from ..joints.revolute import Revolute

    graph = LinkageGraph(name=linkage.name)

    # Dimensional data
    node_positions: dict[str, tuple[float, float]] = {}
    driver_angles: dict[str, DriverAngle] = {}
    edge_distances: dict[str, float] = {}

    # Track joint -> node mapping using object id
    joint_to_node: dict[int, NodeId] = {}
    edge_counter = 0

    def get_or_create_anchor_node(
        joint: object, coord: tuple[float | None, float | None]
    ) -> NodeId:
        """Get existing node for joint or create anchor node."""
        nonlocal edge_counter

        joint_id = id(joint)
        if joint_id in joint_to_node:
            return joint_to_node[joint_id]

        # Create implicit ground node for tuple-defined anchor
        node_id = f"anchor_{edge_counter}"
        edge_counter += 1

        anchor_node = Node(
            id=node_id,
            role=NodeRole.GROUND,
            name=node_id,
        )
        graph.add_node(anchor_node)
        joint_to_node[joint_id] = node_id

        # Store position in dimensions
        if coord[0] is not None and coord[1] is not None:
            node_positions[node_id] = (coord[0], coord[1])

        return node_id

    # First pass: create nodes for all joints in the linkage
    for i, joint in enumerate(linkage.joints):
        node_id = joint.name if joint.name else f"joint_{i}"

        # Ensure unique node IDs
        if node_id in graph.nodes:
            node_id = f"{node_id}_{i}"

        # Determine node properties based on joint type
        # Use _StaticBase to match both user-created Static and internal _StaticBase
        if isinstance(joint, _StaticBase):
            role = NodeRole.GROUND
            joint_type = JointType.REVOLUTE
        elif isinstance(joint, Crank):
            role = NodeRole.DRIVER
            joint_type = JointType.REVOLUTE
        elif isinstance(joint, Prismatic):
            # Prismatic has a prismatic component, but the joint itself is revolute
            role = NodeRole.DRIVEN
            joint_type = JointType.REVOLUTE
        else:
            role = NodeRole.DRIVEN
            joint_type = JointType.REVOLUTE

        node = Node(
            id=node_id,
            joint_type=joint_type,
            role=role,
            name=joint.name,
        )

        # Store position in dimensions
        coord = joint.coord()
        if coord[0] is not None and coord[1] is not None:
            node_positions[node_id] = (coord[0], coord[1])

        # Store driver angle in dimensions
        if isinstance(joint, Crank) and joint.angle is not None:
            # joint.angle is the initial angle
            driver_angles[node_id] = DriverAngle(
                angular_velocity=tau / 360,
                initial_angle=joint.angle,
            )

        graph.add_node(node)
        joint_to_node[id(joint)] = node_id

    # Second pass: create edges from joint relationships
    for joint in linkage.joints:
        node_id = joint_to_node[id(joint)]

        if isinstance(joint, Crank):
            # Crank has connection to joint0 with distance r
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                if joint.r is not None:
                    edge_distances[edge_id] = joint.r
                edge_counter += 1

        elif isinstance(joint, Revolute):
            # Revolute has two parent connections
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                if joint.r0 is not None:
                    edge_distances[edge_id] = joint.r0
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint1, joint.joint1.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                if joint.r1 is not None:
                    edge_distances[edge_id] = joint.r1
                edge_counter += 1

        elif isinstance(joint, Fixed):
            # Fixed has two parent connections (distance + angle)
            # We store distance as edge constraint; angle is implicit
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                if joint.r is not None:
                    edge_distances[edge_id] = joint.r
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint1, joint.joint1.coord())

                # Edge to joint1 for orientation reference (no distance constraint)
                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Prismatic):
            # Prismatic has revolute connection + line constraint
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                if joint.revolute_radius is not None:
                    edge_distances[edge_id] = joint.revolute_radius
                edge_counter += 1

            # Line-defining joints (joint1, joint2)
            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint1, joint.joint1.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                edge_counter += 1

            if hasattr(joint, "joint2") and joint.joint2 is not None:
                parent_id = joint_to_node.get(id(joint.joint2))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint2, joint.joint2.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                )
                graph.add_edge(edge)
                edge_counter += 1

    dimensions = Dimensions(
        node_positions=node_positions,
        driver_angles=driver_angles,
        edge_distances=edge_distances,
        name=linkage.name,
    )

    return graph, dimensions
