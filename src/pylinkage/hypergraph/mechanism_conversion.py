"""Direct conversion between HypergraphLinkage and Mechanism.

This module provides direct conversion between the hypergraph representation
and the mechanism model, bypassing the legacy Linkage class.

This is the preferred conversion path for new code:
- HypergraphLinkage is the abstract mathematical representation
- Mechanism is the concrete simulation model with Links + Joints

For backward compatibility with legacy Linkage, use conversion.py instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._types import JointType, NodeId, NodeRole
from .core import Edge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..mechanism import Mechanism


def to_mechanism(hypergraph: HypergraphLinkage) -> Mechanism:
    """Convert a HypergraphLinkage directly to a Mechanism.

    This converts the hypergraph to the mechanism model without going
    through the legacy Linkage class. This is the preferred conversion
    path for new code.

    Args:
        hypergraph: The HypergraphLinkage to convert.

    Returns:
        A Mechanism instance ready for simulation.

    Raises:
        ValueError: If the hypergraph is underconstrained or has
            disconnected components that cannot be solved.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> # ... add nodes, edges, hyperedges ...
        >>> mechanism = to_mechanism(hg)
        >>> for positions in mechanism.step():
        ...     print(positions)
    """
    from ..mechanism import (
        DriverLink,
        GroundJoint,
        GroundLink,
        Link,
        Mechanism,
        PrismaticJoint,
        RevoluteJoint,
    )
    from ..mechanism.joint import Joint

    # First expand hyperedges to simple graph
    simple = hypergraph.to_simple_graph()

    joints: list[Joint] = []
    links: list[Link] = []
    node_to_joint: dict[NodeId, Joint] = {}
    ground_joints: list[GroundJoint] = []

    # Create ground joints first
    for node in simple.ground_nodes():
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        ground = GroundJoint(
            id=node.id,
            position=(x, y),
            name=node.name,
        )
        joints.append(ground)
        node_to_joint[node.id] = ground
        ground_joints.append(ground)

    # Create ground link from ground joints
    ground_link: GroundLink | None = None
    if ground_joints:
        ground_link = GroundLink(
            id="ground",
            joints=list(ground_joints),
            name="ground",
        )
        links.append(ground_link)

    # Create driver joints and links
    for node in simple.driver_nodes():
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        # Create the output joint for the crank
        output_joint = RevoluteJoint(
            id=node.id,
            position=(x, y),
            name=node.name,
        )
        joints.append(output_joint)
        node_to_joint[node.id] = output_joint

        # Find ground connection to create driver link
        neighbors = simple.neighbors(node.id)
        motor_joint: GroundJoint | None = None
        radius: float | None = None

        for neighbor_id in neighbors:
            neighbor = simple.nodes.get(neighbor_id)
            if neighbor and neighbor.role == NodeRole.GROUND:
                motor = node_to_joint.get(neighbor_id)
                if isinstance(motor, GroundJoint):
                    motor_joint = motor
                    edge = simple.get_edge_between(node.id, neighbor_id)
                    if edge:
                        radius = edge.distance
                    break

        if motor_joint:
            # Compute initial angle from motor to output
            mx, my = motor_joint.position
            if mx is not None and my is not None and x is not None and y is not None:
                initial_angle = math.atan2(y - my, x - mx)
            else:
                initial_angle = 0.0

            driver = DriverLink(
                id=f"{node.id}_crank",
                joints=[motor_joint, output_joint],
                name=f"{node.id}_crank",
                motor_joint=motor_joint,
                angular_velocity=node.angle if node.angle else 0.1,
                initial_angle=initial_angle,
            )
            links.append(driver)

    # Create driven joints - build dependency order
    driven = simple.driven_nodes()
    solved_nodes: set[NodeId] = set()
    for node in simple.ground_nodes():
        solved_nodes.add(node.id)
    for node in simple.driver_nodes():
        solved_nodes.add(node.id)

    # Iteratively add nodes whose parents are solved
    remaining = {n.id: n for n in driven}
    max_iterations = len(remaining) * 2

    for _ in range(max_iterations):
        if not remaining:
            break

        for node_id, node in list(remaining.items()):
            neighbors = simple.neighbors(node_id)
            parent_ids = [n for n in neighbors if n in solved_nodes]

            # Need at least 2 solved parents for a driven joint
            if len(parent_ids) >= 2:
                pos = node.position
                x = pos[0] if pos[0] is not None else 0.0
                y = pos[1] if pos[1] is not None else 0.0

                created_joint: Joint

                if node.joint_type == JointType.PRISMATIC:
                    # Prismatic joint - needs circle center + line
                    created_joint = PrismaticJoint(
                        id=node_id,
                        position=(x, y),
                        name=node.name,
                    )
                else:
                    # Revolute joint (most common)
                    created_joint = RevoluteJoint(
                        id=node_id,
                        position=(x, y),
                        name=node.name,
                    )

                joints.append(created_joint)
                node_to_joint[node_id] = created_joint

                # Create links to parent joints
                for i, parent_id in enumerate(parent_ids[:2]):
                    parent_joint = node_to_joint[parent_id]
                    edge = simple.get_edge_between(node_id, parent_id)

                    link = Link(
                        id=f"{node_id}_link{i}",
                        joints=[parent_joint, created_joint],
                        name=f"{node_id}_link{i}",
                    )
                    # Note: Link length is determined by edge distance in the graph
                    # The mechanism will use the joint positions to compute distances
                    links.append(link)

                solved_nodes.add(node_id)
                del remaining[node_id]
                break

    if remaining:
        raise ValueError(
            f"Could not determine solve order for nodes: {list(remaining.keys())}. "
            "The hypergraph may be underconstrained or have disconnected components."
        )

    return Mechanism(
        name=hypergraph.name,
        joints=joints,
        links=links,
        ground=ground_link,
    )


def from_mechanism(mechanism: Mechanism) -> HypergraphLinkage:
    """Convert a Mechanism to a HypergraphLinkage.

    This converts an existing Mechanism to the hypergraph representation
    for analysis, visualization, or manipulation.

    Args:
        mechanism: The Mechanism to convert.

    Returns:
        A HypergraphLinkage representation.

    Example:
        >>> mechanism = Mechanism(joints=[...], links=[...])
        >>> hg = from_mechanism(mechanism)
        >>> # Analyze or modify the hypergraph
    """
    from ..mechanism import DriverLink, GroundJoint, PrismaticJoint

    hypergraph = HypergraphLinkage(name=mechanism.name)

    # Track joint -> node mapping
    joint_to_node: dict[str, NodeId] = {}
    edge_counter = 0

    # Create nodes for all joints
    for joint in mechanism.joints:
        node_id = joint.id

        # Ensure unique node IDs
        if node_id in hypergraph.nodes:
            node_id = f"{node_id}_{id(joint)}"

        # Determine node properties
        if isinstance(joint, GroundJoint):
            role = NodeRole.GROUND
            joint_type = JointType.REVOLUTE
        elif isinstance(joint, PrismaticJoint):
            role = NodeRole.DRIVEN
            joint_type = JointType.PRISMATIC
        else:
            # Check if this joint is a driver output
            is_driver_output = False
            for link in mechanism.links:
                if isinstance(link, DriverLink):
                    if link.output_joint and link.output_joint.id == joint.id:
                        is_driver_output = True
                        break

            role = NodeRole.DRIVER if is_driver_output else NodeRole.DRIVEN
            joint_type = JointType.REVOLUTE

        pos = joint.position
        node = Node(
            id=node_id,
            joint_type=joint_type,
            role=role,
            position=pos,
            name=joint.name,
        )

        # Add driver-specific attributes
        if role == NodeRole.DRIVER:
            for link in mechanism.links:
                if isinstance(link, DriverLink):
                    if link.output_joint and link.output_joint.id == joint.id:
                        node.angle = link.angular_velocity
                        break

        hypergraph.add_node(node)
        joint_to_node[joint.id] = node_id

    # Create edges from links
    for link in mechanism.links:
        if len(link.joints) >= 2:
            for i in range(len(link.joints)):
                for j in range(i + 1, len(link.joints)):
                    joint_a = link.joints[i]
                    joint_b = link.joints[j]

                    node_a = joint_to_node.get(joint_a.id)
                    node_b = joint_to_node.get(joint_b.id)

                    if node_a and node_b:
                        # Check if edge already exists
                        existing = hypergraph.get_edge_between(node_a, node_b)
                        if existing is None:
                            distance = link.get_distance(joint_a, joint_b)
                            edge = Edge(
                                id=f"edge_{edge_counter}",
                                source=node_a,
                                target=node_b,
                                distance=distance,
                            )
                            hypergraph.add_edge(edge)
                            edge_counter += 1

    return hypergraph
