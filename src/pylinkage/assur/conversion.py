"""Conversion between graph and joint-based representations.

This module provides functions to convert between the graph-based
Assur group representation and the existing joint-based Linkage class.

.. deprecated:: 0.8.0
    The ``graph_to_linkage()`` function converts to the legacy Linkage class.
    For new code, use ``graph_to_mechanism()`` from ``assur.mechanism_conversion``
    which converts directly to the new Mechanism model.

This enables interoperability between the two systems:
- Define linkages using graph syntax, convert to Linkage for simulation
- Analyze existing Linkages by converting to graph representation
"""

import warnings
from typing import TYPE_CHECKING

from ..dimensions import Dimensions, DriverAngle
from ..joints import Prismatic
from ._types import JointType, NodeId, NodeRole
from .decomposition import decompose_assur_groups
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
    from ..joints import Crank, Fixed, Prismatic, Revolute
    from ..joints.joint import _StaticBase

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
            # joint.angle is the initial angle, angular_velocity defaults to 0.1
            driver_angles[node_id] = DriverAngle(
                angular_velocity=0.1,
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


def graph_to_linkage(graph: LinkageGraph, dimensions: Dimensions) -> "Linkage":
    """Convert a graph representation and Dimensions to a Linkage.

    .. deprecated:: 0.8.0
        Use ``graph_to_mechanism()`` instead for direct conversion to the new
        Mechanism model. This function converts to the legacy Linkage class.

        Example migration::

            # Old (deprecated):
            from pylinkage.assur import graph_to_linkage
            linkage = graph_to_linkage(graph, dims)

            # New (preferred):
            from pylinkage.assur import graph_to_mechanism
            mechanism = graph_to_mechanism(graph, dims)

    This function:
    1. Decomposes the graph into Assur groups
    2. Creates appropriate Joint instances for each element
    3. Constructs a Linkage with correct solving order

    Args:
        graph: The LinkageGraph defining topology.
        dimensions: The Dimensions providing positions, distances, angles.

    Returns:
        A Linkage instance ready for simulation.

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND))
        >>> dims = Dimensions(node_positions={"A": (0, 0)}, ...)
        >>> linkage = graph_to_linkage(graph, dims)
        >>> for coords in linkage.step():
        ...     print(coords)
    """
    warnings.warn(
        "graph_to_linkage() is deprecated. Use graph_to_mechanism() from "
        "pylinkage.assur.mechanism_conversion for direct conversion "
        "to the new Mechanism model.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..joints import Crank, Revolute
    from ..joints.joint import Joint, _StaticBase
    from ..linkage.linkage import Linkage as LinkageClass

    # Decompose to get solving order
    decomposition = decompose_assur_groups(graph)

    joints: list[Joint] = []
    node_to_joint: dict[NodeId, Joint] = {}

    # Create ground joints (use _StaticBase to avoid deprecation warnings)
    for node_id in decomposition.ground:
        node = graph.nodes[node_id]
        pos = dimensions.get_node_position(node_id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

        joint = _StaticBase(
            x=x,
            y=y,
            name=node.name,
        )
        joints.append(joint)
        node_to_joint[node_id] = joint

    # Create driver joints (Cranks)
    for node_id in decomposition.drivers:
        node = graph.nodes[node_id]
        pos = dimensions.get_node_position(node_id)
        x = pos[0] if pos else 0.0
        y = pos[1] if pos else 0.0

        # Find the ground connection
        neighbors = graph.neighbors(node_id)
        parent_joint: Joint | None = None
        distance: float | None = None

        for neighbor_id in neighbors:
            if neighbor_id in decomposition.ground:
                parent_joint = node_to_joint.get(neighbor_id)
                edge = graph.get_edge_between(node_id, neighbor_id)
                if edge is not None:
                    distance = dimensions.get_edge_distance(edge.id)
                break

        driver_angle = dimensions.get_driver_angle(node_id)
        angle = driver_angle.angular_velocity if driver_angle else 0.1

        crank_joint = Crank(
            x=x,
            y=y,
            joint0=parent_joint,
            distance=distance,
            angle=angle,
            name=node.name,
        )
        joints.append(crank_joint)
        node_to_joint[node_id] = crank_joint

    # Create joints for each Assur group
    for group in decomposition.groups:
        if group.joint_signature == "RRR":
            # Create Revolute joint
            if len(group.internal_nodes) != 1:
                raise ValueError(
                    f"RRR group should have 1 internal node, got {len(group.internal_nodes)}"
                )

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = dimensions.get_node_position(internal_id)
            x = pos[0] if pos else 0.0
            y = pos[1] if pos else 0.0

            anchor0 = node_to_joint.get(group.anchor_nodes[0])
            anchor1 = (
                node_to_joint.get(group.anchor_nodes[1]) if len(group.anchor_nodes) > 1 else None
            )

            # Get distances from dimensions using edge IDs
            distance0: float | None = None
            distance1: float | None = None
            if len(group.internal_edges) > 0:
                distance0 = dimensions.get_edge_distance(group.internal_edges[0])
            if len(group.internal_edges) > 1:
                distance1 = dimensions.get_edge_distance(group.internal_edges[1])

            revolute_joint = Revolute(
                x=x,
                y=y,
                joint0=anchor0,
                joint1=anchor1,
                distance0=distance0,
                distance1=distance1,
                name=node.name,
            )
            joints.append(revolute_joint)
            node_to_joint[internal_id] = revolute_joint

        elif group.joint_signature == "RRP":
            # Create Prismatic joint
            if len(group.internal_nodes) != 1:
                raise ValueError(
                    f"RRP group should have 1 internal node, got {len(group.internal_nodes)}"
                )

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = dimensions.get_node_position(internal_id)
            x = pos[0] if pos else 0.0
            y = pos[1] if pos else 0.0

            anchor = node_to_joint.get(group.anchor_nodes[0]) if group.anchor_nodes else None
            line1 = node_to_joint.get(getattr(group, "line_node1", None) or "")
            line2 = node_to_joint.get(getattr(group, "line_node2", None) or "")

            # Get revolute distance from dimensions using edge ID
            revolute_distance: float | None = None
            if len(group.internal_edges) > 0:
                revolute_distance = dimensions.get_edge_distance(group.internal_edges[0])

            prismatic_joint = Prismatic(
                x=x,
                y=y,
                joint0=anchor,
                joint1=line1,
                joint2=line2,
                revolute_radius=revolute_distance,
                name=node.name,
            )
            joints.append(prismatic_joint)
            node_to_joint[internal_id] = prismatic_joint

        else:
            raise NotImplementedError(
                f"Conversion for {group.joint_signature} group not implemented"
            )

    # Build solve order from decomposition
    order: list[Joint] = []
    for node_id in decomposition.drivers:
        order.append(node_to_joint[node_id])
    for group in decomposition.groups:
        for internal_id in group.internal_nodes:
            order.append(node_to_joint[internal_id])

    return LinkageClass(
        joints=joints,
        order=order,
        name=graph.name,
    )
