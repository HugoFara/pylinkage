"""Direct conversion between HypergraphLinkage and Mechanism.

This module provides direct conversion between the hypergraph representation
and the mechanism model, bypassing the legacy Linkage class.

This is the preferred conversion path for new code:
- HypergraphLinkage is the pure topological representation
- Dimensions holds the geometric data (positions, distances, angles)
- Mechanism is the concrete simulation model with Links + Joints

For backward compatibility with legacy Linkage, use conversion.py instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._types import JointType, NodeId, NodeRole
from ..dimensions import Dimensions, DriverAngle
from .core import Edge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..mechanism import Mechanism


def _slider_anchor_and_line(
    hypergraph: HypergraphLinkage, node_id: NodeId
) -> tuple[NodeId, tuple[NodeId, NodeId]]:
    """Return the revolute anchor and the two line nodes of a slider node.

    Reads the ``RRPDyad`` encoding of :func:`from_sim_linkage`: one edge
    to the revolute anchor, one hyperedge over the slider and the two
    nodes that define its slide line.

    Raises:
        NotImplementedError: If the node is not a slider on a fixed line.
    """
    edges = hypergraph.get_edges_for_node(node_id)
    hyperedges = hypergraph.get_hyperedges_for_node(node_id)
    what = f"Prismatic node {node_id!r}"
    if len(edges) != 1:
        raise NotImplementedError(
            f"{what} has {len(edges)} edges; Mechanism only represents a slider "
            "with one revolute leg on a fixed line (an RRP dyad), not a "
            "double slider or a slider pinned to several links."
        )
    if len(hyperedges) != 1 or len(hyperedges[0].nodes) != 3:
        raise NotImplementedError(
            f"{what} needs exactly one hyperedge of three nodes to define its "
            f"slide line, found {[list(h.nodes) for h in hyperedges]}."
        )
    line_ids = tuple(n for n in hyperedges[0].nodes if n != node_id)
    moving = [n for n in line_ids if hypergraph.nodes[n].role != NodeRole.GROUND]
    if moving:
        raise NotImplementedError(
            f"{what} slides on a line through {moving}, which is not ground; "
            "Mechanism only represents a slide line fixed to the frame."
        )
    return edges[0].other_node(node_id), (line_ids[0], line_ids[1])


def to_mechanism(hypergraph: HypergraphLinkage, dimensions: Dimensions) -> Mechanism:
    """Convert a HypergraphLinkage and Dimensions to a Mechanism.

    This converts the hypergraph topology plus dimensional data to the
    mechanism model. This is the preferred conversion path for new code.

    A driven ``JointType.PRISMATIC`` node is a slider: it needs one
    edge, to the revolute anchor it is linked to, and one hyperedge of
    three nodes whose two other nodes lie on the slide line, which is
    how :func:`from_sim_linkage` encodes an ``RRPDyad``. The mechanism
    keeps the line as the :class:`~pylinkage.mechanism.PrismaticJoint`'s
    ``axis`` and ``line_point``, so the two line nodes must be ground.

    Args:
        hypergraph: The HypergraphLinkage defining the topology.
        dimensions: The Dimensions providing positions, distances, angles.

    Returns:
        A Mechanism instance ready for simulation.

    Raises:
        ValueError: If the hypergraph is underconstrained or has
            disconnected components that cannot be solved.
        NotImplementedError: If a prismatic node is not a slider on a
            fixed line (its guide is a moving link, it has no revolute
            leg or several, or it is a ``PPDyad``-style double slider),
            which :class:`~pylinkage.mechanism.Mechanism` cannot
            represent yet.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> hg.add_node(Node("A", role=NodeRole.GROUND))
        >>> hg.add_edge(Edge("AB", "A", "B"))
        >>> dims = Dimensions(
        ...     node_positions={"A": (0, 0), "B": (1, 0)},
        ...     edge_distances={"AB": 1.0},
        ... )
        >>> mechanism = to_mechanism(hg, dims)
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
        pos = dimensions.get_node_position(node.id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

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
        pos = dimensions.get_node_position(node.id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

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

        for neighbor_id in neighbors:
            neighbor = simple.nodes.get(neighbor_id)
            if neighbor and neighbor.role == NodeRole.GROUND:
                motor = node_to_joint.get(neighbor_id)
                if isinstance(motor, GroundJoint):
                    motor_joint = motor
                    break

        if motor_joint:
            # Compute initial angle from motor to output
            mx, my = motor_joint.position
            initial_angle = math.atan2(y - my, x - mx) if mx is not None and my is not None else 0.0

            # Get driver angle from dimensions
            driver_angle = dimensions.get_driver_angle(node.id)
            angular_velocity = driver_angle.angular_velocity if driver_angle else math.tau / 360

            driver = DriverLink(
                id=f"{node.id}_crank",
                joints=[motor_joint, output_joint],
                name=f"{node.id}_crank",
                motor_joint=motor_joint,
                angular_velocity=angular_velocity,
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
            pos = dimensions.get_node_position(node_id)
            x = pos[0] if pos else 0.0
            y = pos[1] if pos else 0.0

            created_joint: Joint

            if node.joint_type == JointType.PRISMATIC:
                # A slider: one link to its revolute anchor, and the
                # slide line taken from its guide hyperedge.
                anchor_id, line_ids = _slider_anchor_and_line(hypergraph, node_id)
                if anchor_id not in solved_nodes:
                    continue
                line_start = dimensions.get_node_position(line_ids[0])
                line_end = dimensions.get_node_position(line_ids[1])
                if line_start is None or line_end is None:
                    raise ValueError(
                        f"Prismatic node {node_id!r} slides on the line through "
                        f"{line_ids[0]!r} and {line_ids[1]!r}, which need positions."
                    )
                axis = (line_end[0] - line_start[0], line_end[1] - line_start[1])
                if math.hypot(*axis) < 1e-12:
                    raise ValueError(
                        f"Prismatic node {node_id!r}: line nodes {line_ids[0]!r} and "
                        f"{line_ids[1]!r} coincide, so they define no slide line."
                    )
                created_joint = PrismaticJoint(
                    id=node_id,
                    position=(x, y),
                    name=node.name,
                    axis=axis,
                    line_point=(line_start[0], line_start[1]),
                )
                parent_ids = [anchor_id]
            else:
                neighbors = simple.neighbors(node_id)
                parent_ids = [n for n in neighbors if n in solved_nodes][:2]
                # Need 2 solved parents for a revolute joint
                if len(parent_ids) < 2:
                    continue
                created_joint = RevoluteJoint(
                    id=node_id,
                    position=(x, y),
                    name=node.name,
                )

            joints.append(created_joint)
            node_to_joint[node_id] = created_joint

            # Create links to parent joints
            for i, parent_id in enumerate(parent_ids):
                parent_joint = node_to_joint[parent_id]

                link = Link(
                    id=f"{node_id}_link{i}",
                    joints=[parent_joint, created_joint],
                    name=f"{node_id}_link{i}",
                )
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


def from_mechanism(mechanism: Mechanism) -> tuple[HypergraphLinkage, Dimensions]:
    """Convert a Mechanism to a HypergraphLinkage and Dimensions.

    This converts an existing Mechanism to the hypergraph representation
    for analysis, visualization, or manipulation.

    Args:
        mechanism: The Mechanism to convert.

    Returns:
        A tuple of (HypergraphLinkage, Dimensions).

    Example:
        >>> mechanism = Mechanism(joints=[...], links=[...])
        >>> hg, dims = from_mechanism(mechanism)
        >>> # Analyze or modify the hypergraph
    """
    from ..mechanism import DriverLink, GroundJoint, PrismaticJoint

    hypergraph = HypergraphLinkage(name=mechanism.name)

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

        # Add driver-specific attributes to dimensions
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
                            edge_id = f"edge_{edge_counter}"
                            edge = Edge(
                                id=edge_id,
                                source=node_a,
                                target=node_b,
                            )
                            hypergraph.add_edge(edge)

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

    return hypergraph, dimensions
