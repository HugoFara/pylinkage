"""Direct conversion between LinkageGraph and Mechanism.

This module provides direct conversion between the Assur graph representation
and the mechanism model, bypassing the legacy Linkage class.

This is the preferred conversion path for new code:
- LinkageGraph is the formal kinematic graph representation (Assur theory)
- Mechanism is the concrete simulation model with Links + Joints

For backward compatibility with legacy Linkage, use conversion.py instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._types import JointType, NodeId, NodeRole
from .decomposition import decompose_assur_groups
from .graph import Edge, LinkageGraph, Node

if TYPE_CHECKING:
    from ..mechanism import Mechanism


def graph_to_mechanism(graph: LinkageGraph) -> Mechanism:
    """Convert a LinkageGraph directly to a Mechanism.

    This converts the Assur graph to the mechanism model without going
    through the legacy Linkage class. This is the preferred conversion
    path for new code.

    The conversion:
    1. Decomposes the graph into Assur groups
    2. Creates appropriate Joint and Link instances for each element
    3. Constructs a Mechanism with correct topology

    Args:
        graph: The LinkageGraph to convert.

    Returns:
        A Mechanism instance ready for simulation.

    Raises:
        ValueError: If the graph cannot be decomposed into valid Assur groups.
        NotImplementedError: If an unsupported Assur group type is encountered.

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
        >>> # ... add more nodes and edges ...
        >>> mechanism = graph_to_mechanism(graph)
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

    # Decompose to get solving order
    decomposition = decompose_assur_groups(graph)

    joints: list[Joint] = []
    links: list[Link] = []
    node_to_joint: dict[NodeId, Joint] = {}
    ground_joints: list[GroundJoint] = []

    # Create ground joints
    for node_id in decomposition.ground:
        node = graph.nodes[node_id]
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        ground = GroundJoint(
            id=node_id,
            position=(x, y),
            name=node.name,
        )
        joints.append(ground)
        node_to_joint[node_id] = ground
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
    for node_id in decomposition.drivers:
        node = graph.nodes[node_id]
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        # Create the output joint for the crank
        output_joint = RevoluteJoint(
            id=node_id,
            position=(x, y),
            name=node.name,
        )
        joints.append(output_joint)
        node_to_joint[node_id] = output_joint

        # Find ground connection to create driver link
        neighbors = graph.neighbors(node_id)
        motor_joint: GroundJoint | None = None
        radius: float | None = None

        for neighbor_id in neighbors:
            if neighbor_id in decomposition.ground:
                motor = node_to_joint.get(neighbor_id)
                if isinstance(motor, GroundJoint):
                    motor_joint = motor
                    edge = graph.get_edge_between(node_id, neighbor_id)
                    if edge is not None:
                        radius = edge.distance
                    break

        if motor_joint:
            # Compute initial angle from motor to output
            mx, my = motor_joint.position
            if mx is not None and my is not None:
                initial_angle = math.atan2(y - my, x - mx)
            else:
                initial_angle = 0.0

            driver = DriverLink(
                id=f"{node_id}_crank",
                joints=[motor_joint, output_joint],
                name=f"{node_id}_crank",
                motor_joint=motor_joint,
                angular_velocity=node.angle if node.angle else 0.1,
                initial_angle=initial_angle,
            )
            links.append(driver)

    # Create joints for each Assur group
    for group in decomposition.groups:
        if group.joint_signature == "RRR":
            # RRR dyad - create RevoluteJoint with two Links
            if len(group.internal_nodes) != 1:
                raise ValueError(
                    f"RRR group should have 1 internal node, got {len(group.internal_nodes)}"
                )

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = node.position
            x = pos[0] if pos[0] is not None else 0.0
            y = pos[1] if pos[1] is not None else 0.0

            revolute_joint = RevoluteJoint(
                id=internal_id,
                position=(x, y),
                name=node.name,
            )
            joints.append(revolute_joint)
            node_to_joint[internal_id] = revolute_joint

            # Create links to anchor joints
            anchor0 = node_to_joint.get(group.anchor_nodes[0])
            if anchor0:
                link0 = Link(
                    id=f"{internal_id}_link0",
                    joints=[anchor0, revolute_joint],
                    name=f"{internal_id}_link0",
                )
                links.append(link0)

            if len(group.anchor_nodes) > 1:
                anchor1 = node_to_joint.get(group.anchor_nodes[1])
                if anchor1:
                    link1 = Link(
                        id=f"{internal_id}_link1",
                        joints=[anchor1, revolute_joint],
                        name=f"{internal_id}_link1",
                    )
                    links.append(link1)

        elif group.joint_signature == "RRP":
            # RRP dyad - create PrismaticJoint with links
            if len(group.internal_nodes) != 1:
                raise ValueError(
                    f"RRP group should have 1 internal node, got {len(group.internal_nodes)}"
                )

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = node.position
            x = pos[0] if pos[0] is not None else 0.0
            y = pos[1] if pos[1] is not None else 0.0

            # Determine axis from line nodes if available
            line_node1 = getattr(group, "line_node1", None)
            line_node2 = getattr(group, "line_node2", None)
            axis = (1.0, 0.0)  # Default horizontal

            if line_node1 and line_node2:
                n1 = graph.nodes.get(line_node1)
                n2 = graph.nodes.get(line_node2)
                if n1 and n2 and n1.position[0] is not None and n2.position[0] is not None:
                    dx = n2.position[0] - n1.position[0]
                    dy = (n2.position[1] or 0.0) - (n1.position[1] or 0.0)
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 1e-10:
                        axis = (dx / length, dy / length)

            prismatic_joint = PrismaticJoint(
                id=internal_id,
                position=(x, y),
                name=node.name,
                axis=axis,
            )
            joints.append(prismatic_joint)
            node_to_joint[internal_id] = prismatic_joint

            # Create link to revolute anchor
            if group.anchor_nodes:
                anchor = node_to_joint.get(group.anchor_nodes[0])
                if anchor:
                    link = Link(
                        id=f"{internal_id}_link0",
                        joints=[anchor, prismatic_joint],
                        name=f"{internal_id}_link0",
                    )
                    links.append(link)

        else:
            raise NotImplementedError(
                f"Conversion for {group.joint_signature} group not implemented"
            )

    return Mechanism(
        name=graph.name,
        joints=joints,
        links=links,
        ground=ground_link,
    )


def mechanism_to_graph(mechanism: Mechanism) -> LinkageGraph:
    """Convert a Mechanism to a LinkageGraph.

    This converts an existing Mechanism to the Assur graph representation
    for analysis, decomposition, or visualization.

    Args:
        mechanism: The Mechanism to convert.

    Returns:
        A LinkageGraph representation.

    Example:
        >>> mechanism = Mechanism(joints=[...], links=[...])
        >>> graph = mechanism_to_graph(mechanism)
        >>> decomposition = decompose_assur_groups(graph)
    """
    from ..mechanism import DriverLink, GroundJoint, PrismaticJoint

    graph = LinkageGraph(name=mechanism.name)

    # Track joint -> node mapping
    joint_to_node: dict[str, NodeId] = {}
    edge_counter = 0

    # Create nodes for all joints
    for joint in mechanism.joints:
        node_id = joint.id

        # Ensure unique node IDs
        if node_id in graph.nodes:
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

        graph.add_node(node)
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
                        existing = graph.get_edge_between(node_a, node_b)
                        if existing is None:
                            distance = link.get_distance(joint_a, joint_b)
                            edge = Edge(
                                id=f"edge_{edge_counter}",
                                source=node_a,
                                target=node_b,
                                distance=distance,
                            )
                            graph.add_edge(edge)
                            edge_counter += 1

    return graph
