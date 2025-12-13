"""Conversion between graph and joint-based representations.

This module provides functions to convert between the graph-based
Assur group representation and the existing joint-based Linkage class.

This enables interoperability between the two systems:
- Define linkages using graph syntax, convert to Linkage for simulation
- Analyze existing Linkages by converting to graph representation
"""

from typing import TYPE_CHECKING

from ..joints import Prismatic
from ._types import JointType, NodeId, NodeRole
from .decomposition import decompose_assur_groups
from .graph import Edge, LinkageGraph, Node

if TYPE_CHECKING:
    from ..linkage.linkage import Linkage


def linkage_to_graph(linkage: "Linkage") -> LinkageGraph:
    """Convert an existing Linkage to graph representation.

    Maps existing joint types to graph nodes:
    - Static -> GROUND node
    - Crank -> DRIVER node + edge to its anchor
    - Revolute -> DRIVEN node (RRR dyad internal)
    - Linear -> DRIVEN node (RRP dyad internal)
    - Fixed -> DRIVEN node (deterministic constraint)

    Args:
        linkage: The Linkage to convert.

    Returns:
        LinkageGraph representation of the linkage.

    Example:
        >>> linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin])
        >>> graph = linkage_to_graph(linkage)
        >>> print(f"Nodes: {list(graph.nodes.keys())}")
    """
    from ..joints import Crank, Fixed, Prismatic, Revolute, Static

    graph = LinkageGraph(name=linkage.name)

    # Track joint -> node mapping using object id
    joint_to_node: dict[int, NodeId] = {}
    edge_counter = 0

    def get_or_create_anchor_node(joint: object, coord: tuple[float | None, float | None]) -> NodeId:
        """Get existing node for joint or create anchor node."""
        nonlocal edge_counter

        joint_id = id(joint)
        if joint_id in joint_to_node:
            return joint_to_node[joint_id]

        # Create implicit ground node for tuple-defined anchor
        node_id = f"anchor_{edge_counter}"
        edge_counter += 1

        pos = (coord[0], coord[1]) if coord[0] is not None else (None, None)
        anchor_node = Node(
            id=node_id,
            role=NodeRole.GROUND,
            position=pos,
            name=node_id,
        )
        graph.add_node(anchor_node)
        joint_to_node[joint_id] = node_id
        return node_id

    # First pass: create nodes for all joints in the linkage
    for i, joint in enumerate(linkage.joints):
        node_id = joint.name if joint.name else f"joint_{i}"

        # Ensure unique node IDs
        if node_id in graph.nodes:
            node_id = f"{node_id}_{i}"

        # Determine node properties based on joint type
        if isinstance(joint, Static):
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
            position=joint.coord(),
            name=joint.name,
        )

        # Add driver-specific attributes
        if isinstance(joint, Crank):
            node.angle = joint.angle

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

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r,
                )
                graph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Revolute):
            # Revolute has two parent connections
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r0,
                )
                graph.add_edge(edge)
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint1, joint.joint1.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r1,
                )
                graph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Fixed):
            # Fixed has two parent connections (distance + angle)
            # We store distance as edge constraint; angle is implicit
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r,
                )
                graph.add_edge(edge)
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
                    distance=None,  # Orientation reference only
                )
                graph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Prismatic):
            # Prismatic has revolute connection + line constraint
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint0, joint.joint0.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.revolute_radius,
                )
                graph.add_edge(edge)
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
                    distance=None,  # Line constraint, not distance
                )
                graph.add_edge(edge)
                edge_counter += 1

            if hasattr(joint, 'joint2') and joint.joint2 is not None:
                parent_id = joint_to_node.get(id(joint.joint2))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(joint.joint2, joint.joint2.coord())

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=None,  # Line constraint, not distance
                )
                graph.add_edge(edge)
                edge_counter += 1

    return graph


def graph_to_linkage(graph: LinkageGraph) -> "Linkage":
    """Convert a graph representation to a Linkage.

    This function:
    1. Decomposes the graph into Assur groups
    2. Creates appropriate Joint instances for each element
    3. Constructs a Linkage with correct solving order

    Args:
        graph: The LinkageGraph to convert.

    Returns:
        A Linkage instance ready for simulation.

    Example:
        >>> graph = LinkageGraph(name="Four-bar")
        >>> graph.add_node(Node("A", role=NodeRole.GROUND, position=(0, 0)))
        >>> # ... add more nodes and edges ...
        >>> linkage = graph_to_linkage(graph)
        >>> for coords in linkage.step():
        ...     print(coords)
    """
    from ..joints import Crank, Revolute, Static
    from ..joints.joint import Joint
    from ..linkage.linkage import Linkage as LinkageClass

    # Decompose to get solving order
    decomposition = decompose_assur_groups(graph)

    joints: list[Joint] = []
    node_to_joint: dict[NodeId, Joint] = {}

    # Create ground joints (Static)
    for node_id in decomposition.ground:
        node = graph.nodes[node_id]
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        joint = Static(
            x=x,
            y=y,
            name=node.name,
        )
        joints.append(joint)
        node_to_joint[node_id] = joint

    # Create driver joints (Cranks)
    for node_id in decomposition.drivers:
        node = graph.nodes[node_id]
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        # Find the ground connection
        neighbors = graph.neighbors(node_id)
        parent_joint: Joint | None = None
        distance: float | None = None

        for neighbor_id in neighbors:
            if neighbor_id in decomposition.ground:
                parent_joint = node_to_joint.get(neighbor_id)
                edge = graph.get_edge_between(node_id, neighbor_id)
                if edge is not None:
                    distance = edge.distance
                break

        crank_joint = Crank(
            x=x,
            y=y,
            joint0=parent_joint,
            distance=distance,
            angle=node.angle,
            name=node.name,
        )
        joints.append(crank_joint)
        node_to_joint[node_id] = crank_joint

    # Create joints for each Assur group
    for group in decomposition.groups:
        if group.joint_signature == "RRR":
            # Create Revolute joint
            if len(group.internal_nodes) != 1:
                raise ValueError(f"RRR group should have 1 internal node, got {len(group.internal_nodes)}")

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = node.position
            x = pos[0] if pos[0] is not None else 0.0
            y = pos[1] if pos[1] is not None else 0.0

            anchor0 = node_to_joint.get(group.anchor_nodes[0])
            anchor1 = node_to_joint.get(group.anchor_nodes[1]) if len(group.anchor_nodes) > 1 else None

            # Get distances from the group
            distance0 = getattr(group, 'distance0', None)
            distance1 = getattr(group, 'distance1', None)

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
                raise ValueError(f"RRP group should have 1 internal node, got {len(group.internal_nodes)}")

            internal_id = group.internal_nodes[0]
            node = graph.nodes[internal_id]
            pos = node.position
            x = pos[0] if pos[0] is not None else 0.0
            y = pos[1] if pos[1] is not None else 0.0

            anchor = node_to_joint.get(group.anchor_nodes[0]) if group.anchor_nodes else None
            line1 = node_to_joint.get(getattr(group, 'line_node1', None) or '')
            line2 = node_to_joint.get(getattr(group, 'line_node2', None) or '')

            revolute_distance = getattr(group, 'revolute_distance', None)

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
