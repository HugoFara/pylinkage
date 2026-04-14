"""Conversion between hypergraph and Linkage representations.

This module provides ``from_linkage()`` — converting a joint-based
``Linkage`` into the hypergraph representation plus a ``Dimensions``
object. For conversion in the other direction, use ``to_mechanism()``
from ``hypergraph.mechanism_conversion``.

The hypergraph is the foundational mathematical layer. For conversion
to/from the Assur graph representation, use the
``assur.hypergraph_conversion`` module.
"""

from math import tau
from typing import TYPE_CHECKING

from ..dimensions import Dimensions, DriverAngle
from ._types import JointType, NodeId, NodeRole
from .core import Edge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..linkage.linkage import Linkage


def from_linkage(linkage: "Linkage") -> tuple[HypergraphLinkage, Dimensions]:
    """Convert a joint-based Linkage to a HypergraphLinkage and Dimensions.

    This converts an existing Linkage to the hypergraph representation
    for analysis or manipulation.

    Args:
        linkage: The Linkage to convert.

    Returns:
        A tuple of (HypergraphLinkage, Dimensions).

    Example:
        >>> linkage = Linkage(joints=[...], order=[...])
        >>> hg, dims = from_linkage(linkage)
    """
    from ..joints.crank import Crank
    from ..joints.fixed import Fixed
    from ..joints.joint import Static
    from ..joints.prismatic import Prismatic
    from ..joints.revolute import Revolute

    hypergraph = HypergraphLinkage(name=linkage.name)

    # Dimensional data
    node_positions: dict[str, tuple[float, float]] = {}
    driver_angles: dict[str, DriverAngle] = {}
    edge_distances: dict[str, float] = {}

    # Track joint -> node mapping
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

        # Create implicit ground node
        node_id = f"anchor_{edge_counter}"
        edge_counter += 1

        anchor_node = Node(
            id=node_id,
            role=NodeRole.GROUND,
            name=node_id,
        )
        hypergraph.add_node(anchor_node)
        joint_to_node[joint_id] = node_id

        # Store position in dimensions
        if coord[0] is not None and coord[1] is not None:
            node_positions[node_id] = (coord[0], coord[1])

        return node_id

    # First pass: create nodes for all joints
    for i, joint in enumerate(linkage.joints):
        node_id = joint.name if joint.name else f"joint_{i}"

        # Ensure unique node IDs
        if node_id in hypergraph.nodes:
            node_id = f"{node_id}_{i}"

        # Determine node properties
        if isinstance(joint, Static) and not isinstance(joint, Crank):
            role = NodeRole.GROUND
            joint_type = JointType.REVOLUTE
        elif isinstance(joint, Crank):
            role = NodeRole.DRIVER
            joint_type = JointType.REVOLUTE
        elif isinstance(joint, Prismatic):
            role = NodeRole.DRIVEN
            joint_type = JointType.PRISMATIC
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

        if isinstance(joint, Crank) and joint.angle is not None:
            driver_angles[node_id] = DriverAngle(
                angular_velocity=tau / 360,
                initial_angle=joint.angle,
            )

        hypergraph.add_node(node)
        joint_to_node[id(joint)] = node_id

    # Second pass: create edges from joint relationships
    for joint in linkage.joints:
        node_id = joint_to_node[id(joint)]

        if isinstance(joint, Crank):
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
                hypergraph.add_edge(edge)
                if joint.r is not None:
                    edge_distances[edge_id] = joint.r
                edge_counter += 1

        elif isinstance(joint, Revolute):
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
                hypergraph.add_edge(edge)
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
                hypergraph.add_edge(edge)
                if joint.r1 is not None:
                    edge_distances[edge_id] = joint.r1
                edge_counter += 1

        elif isinstance(joint, Fixed):
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
                hypergraph.add_edge(edge)
                if joint.r is not None:
                    edge_distances[edge_id] = joint.r
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
                hypergraph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Prismatic):
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
                hypergraph.add_edge(edge)
                if joint.revolute_radius is not None:
                    edge_distances[edge_id] = joint.revolute_radius
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
                hypergraph.add_edge(edge)
                edge_counter += 1

            if hasattr(joint, "joint2") and joint.joint2 is not None:
                parent_id = joint_to_node.get(id(joint.joint2))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint2, joint.joint2.coord())

                edge_id = f"edge_{edge_counter}"
                edge = Edge(
                    id=edge_id,
                    source=parent_id,
                    target=node_id,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

    dimensions = Dimensions(
        node_positions=node_positions,
        driver_angles=driver_angles,
        edge_distances=edge_distances,
        name=linkage.name,
    )

    return hypergraph, dimensions
