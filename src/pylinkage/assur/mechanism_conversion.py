"""Direct conversion between LinkageGraph and Mechanism.

This module provides direct conversion between the Assur graph representation
and the mechanism model, bypassing the legacy Linkage class.

This is the preferred conversion path for new code:
- LinkageGraph is the formal kinematic graph representation (Assur theory)
- Dimensions holds the geometric data (positions, distances, angles)
- Mechanism is the concrete simulation model with Links + Joints

For backward compatibility with legacy Linkage, use conversion.py instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._types import JointType, NodeId, NodeRole
from ..dimensions import Dimensions, DriverAngle
from .decomposition import decompose_assur_groups
from .graph import Edge, LinkageGraph, Node

if TYPE_CHECKING:
    from ..mechanism import Mechanism


def graph_to_mechanism(graph: LinkageGraph, dimensions: Dimensions) -> Mechanism:
    """Convert a LinkageGraph and Dimensions directly to a Mechanism.

    This converts the Assur graph and dimensions to the mechanism model
    without going through the legacy Linkage class. This is the preferred
    conversion path for new code.

    The conversion:
    1. Decomposes the graph into Assur groups
    2. Creates appropriate Joint and Link instances for each element
    3. Constructs a Mechanism with correct topology

    Args:
        graph: The LinkageGraph defining topology.
        dimensions: The Dimensions providing positions, distances, angles.

    Returns:
        A Mechanism instance ready for simulation.

    Raises:
        ValueError: If the graph cannot be decomposed into valid Assur groups.
        NotImplementedError: If an unsupported Assur group type is encountered.

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND))
        >>> dims = Dimensions(node_positions={"A": (0, 0)}, ...)
        >>> mechanism = graph_to_mechanism(graph, dims)
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
        pos = dimensions.get_node_position(node_id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

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
        pos = dimensions.get_node_position(node_id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

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

        for neighbor_id in neighbors:
            if neighbor_id in decomposition.ground:
                motor = node_to_joint.get(neighbor_id)
                if isinstance(motor, GroundJoint):
                    motor_joint = motor
                    break

        if motor_joint:
            # Compute initial angle from motor to output
            mx, my = motor_joint.position
            initial_angle = math.atan2(y - my, x - mx) if mx is not None and my is not None else 0.0

            # Get driver angle from dimensions
            driver_angle = dimensions.get_driver_angle(node_id)
            angular_velocity = driver_angle.angular_velocity if driver_angle else math.tau / 360

            driver = DriverLink(
                id=f"{node_id}_crank",
                joints=[motor_joint, output_joint],
                name=f"{node_id}_crank",
                motor_joint=motor_joint,
                angular_velocity=angular_velocity,
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
            pos = dimensions.get_node_position(internal_id)
            x = pos[0] if pos else 0.0
            y = pos[1] if pos else 0.0

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
            pos = dimensions.get_node_position(internal_id)
            x = pos[0] if pos else 0.0
            y = pos[1] if pos else 0.0

            # Determine axis from line nodes if available
            line_node1 = getattr(group, "line_node1", None)
            line_node2 = getattr(group, "line_node2", None)
            axis = (1.0, 0.0)  # Default horizontal

            if line_node1 and line_node2:
                pos1 = dimensions.get_node_position(line_node1)
                pos2 = dimensions.get_node_position(line_node2)
                if pos1 and pos2:
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
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

        elif group.group_class >= 2:
            # Triad (or higher) — 2+ internal nodes connected to anchors
            # Create a RevoluteJoint for each internal node
            for internal_id in group.internal_nodes:
                node = graph.nodes[internal_id]
                pos = dimensions.get_node_position(internal_id)
                x = pos[0] if pos else 0.0
                y = pos[1] if pos else 0.0

                revolute_joint = RevoluteJoint(
                    id=internal_id,
                    position=(x, y),
                    name=node.name,
                )
                joints.append(revolute_joint)
                node_to_joint[internal_id] = revolute_joint

            # Create a Link for each edge in edge_map
            edge_map = getattr(group, "edge_map", {})
            if not edge_map:
                # Fallback: create links from internal_edges and graph topology
                for edge_id in group.internal_edges:
                    edge = graph.edges.get(edge_id)
                    if edge is None:
                        continue
                    joint_a = node_to_joint.get(edge.source)
                    joint_b = node_to_joint.get(edge.target)
                    if joint_a and joint_b:
                        link = Link(
                            id=edge_id,
                            joints=[joint_a, joint_b],
                            name=edge_id,
                        )
                        links.append(link)
            else:
                for edge_id, (node_a, node_b) in edge_map.items():
                    joint_a = node_to_joint.get(node_a)
                    joint_b = node_to_joint.get(node_b)
                    if joint_a and joint_b:
                        link = Link(
                            id=edge_id,
                            joints=[joint_a, joint_b],
                            name=edge_id,
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


def mechanism_to_graph(mechanism: Mechanism) -> tuple[LinkageGraph, Dimensions]:
    """Convert a Mechanism to a LinkageGraph and Dimensions.

    This converts an existing Mechanism to the Assur graph representation
    for analysis, decomposition, or visualization.

    Args:
        mechanism: The Mechanism to convert.

    Returns:
        A tuple of (LinkageGraph, Dimensions).

    Example:
        >>> mechanism = Mechanism(joints=[...], links=[...])
        >>> graph, dims = mechanism_to_graph(mechanism)
        >>> decomposition = decompose_assur_groups(graph)
    """
    from ..mechanism import DriverLink, GroundJoint, PrismaticJoint

    graph = LinkageGraph(name=mechanism.name)

    # Track joint -> node mapping
    joint_to_node: dict[str, NodeId] = {}
    edge_counter = 0

    # Dimensional data
    node_positions: dict[str, tuple[float, float]] = {}
    driver_angles: dict[str, DriverAngle] = {}
    edge_distances: dict[str, float] = {}

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
                if (
                    isinstance(link, DriverLink)
                    and link.output_joint
                    and link.output_joint.id == joint.id
                ):
                    is_driver_output = True
                    break

            role = NodeRole.DRIVER if is_driver_output else NodeRole.DRIVEN
            joint_type = JointType.REVOLUTE

        node = Node(
            id=node_id,
            joint_type=joint_type,
            role=role,
            name=joint.name,
        )

        # Store position in dimensions
        pos = joint.position
        if pos[0] is not None and pos[1] is not None:
            node_positions[node_id] = (pos[0], pos[1])

        # Store driver angle in dimensions
        if role == NodeRole.DRIVER:
            for link in mechanism.links:
                if (
                    isinstance(link, DriverLink)
                    and link.output_joint
                    and link.output_joint.id == joint.id
                ):
                    driver_angles[node_id] = DriverAngle(
                        angular_velocity=link.angular_velocity,
                        initial_angle=link.initial_angle,
                    )
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
                            edge_id = f"edge_{edge_counter}"
                            edge = Edge(
                                id=edge_id,
                                source=node_a,
                                target=node_b,
                            )
                            graph.add_edge(edge)

                            # Store distance in dimensions
                            distance = link.get_distance(joint_a, joint_b)
                            if distance is not None:
                                edge_distances[edge_id] = distance

                            edge_counter += 1

    dimensions = Dimensions(
        node_positions=node_positions,
        driver_angles=driver_angles,
        edge_distances=edge_distances,
        name=mechanism.name,
    )

    return graph, dimensions
