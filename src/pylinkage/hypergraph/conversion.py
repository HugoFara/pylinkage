"""Conversion between hypergraph and Linkage representations.

This module provides functions to convert between the hypergraph
representation and the joint-based Linkage class.

The hypergraph is the foundational mathematical layer. Conversion to
Linkage enables simulation. For conversion to/from Assur graph
representation, use the assur.hypergraph_conversion module.
"""

from typing import TYPE_CHECKING

from ..joints.joint import Joint
from ._types import JointType, NodeId, NodeRole
from .core import Edge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..linkage.linkage import Linkage


def to_linkage(hypergraph: HypergraphLinkage) -> "Linkage":
    """Convert a HypergraphLinkage to a joint-based Linkage.

    This converts the hypergraph directly to a Linkage that can be used
    for simulation. The conversion maps nodes to joints based on their
    roles and connections.

    Args:
        hypergraph: The HypergraphLinkage to convert.

    Returns:
        A Linkage instance ready for simulation.

    Example:
        >>> hg = HypergraphLinkage(name="Four-bar")
        >>> # ... add nodes, edges, hyperedges ...
        >>> linkage = to_linkage(hg)
        >>> for coords in linkage.step():
        ...     print(coords)
    """
    from ..joints import Crank, Prismatic, Revolute, Static
    from ..joints.joint import Joint
    from ..linkage.linkage import Linkage as LinkageClass

    # First expand hyperedges to simple graph
    simple = hypergraph.to_simple_graph()

    joints: list[Joint] = []
    node_to_joint: dict[NodeId, Joint] = {}
    solve_order: list[Joint] = []

    # Create ground joints (Static) first
    for node in simple.ground_nodes():
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        joint = Static(x=x, y=y, name=node.name)
        joints.append(joint)
        node_to_joint[node.id] = joint

    # Create driver joints (Cranks)
    for node in simple.driver_nodes():
        pos = node.position
        x = pos[0] if pos[0] is not None else 0.0
        y = pos[1] if pos[1] is not None else 0.0

        # Find ground connection
        neighbors = simple.neighbors(node.id)
        parent_joint: Joint | None = None
        distance: float | None = None

        for neighbor_id in neighbors:
            neighbor = simple.nodes.get(neighbor_id)
            if neighbor and neighbor.role == NodeRole.GROUND:
                parent_joint = node_to_joint.get(neighbor_id)
                edge = simple.get_edge_between(node.id, neighbor_id)
                if edge:
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
        node_to_joint[node.id] = crank_joint
        solve_order.append(crank_joint)

    # Create driven joints
    # Build dependency order based on connections
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

                # Determine joint type based on connections
                if node.joint_type == JointType.PRISMATIC:
                    # Prismatic joint - needs circle center + line
                    # Find revolute connection (with distance)
                    revolute_parent: Joint | None = None
                    revolute_dist: float | None = None
                    line_parents: list[Joint] = []

                    for pid in parent_ids:
                        edge = simple.get_edge_between(node_id, pid)
                        if edge and edge.distance is not None:
                            if revolute_parent is None:
                                revolute_parent = node_to_joint.get(pid)
                                revolute_dist = edge.distance
                            else:
                                line_parents.append(node_to_joint[pid])
                        else:
                            line_parents.append(node_to_joint[pid])

                    created_joint: Joint = Prismatic(
                        x=x,
                        y=y,
                        joint0=revolute_parent,
                        joint1=line_parents[0] if len(line_parents) > 0 else None,
                        joint2=line_parents[1] if len(line_parents) > 1 else None,
                        revolute_radius=revolute_dist,
                        name=node.name,
                    )
                else:
                    # Revolute joint (most common)
                    parent0 = node_to_joint.get(parent_ids[0])
                    parent1 = node_to_joint.get(parent_ids[1])

                    edge0 = simple.get_edge_between(node_id, parent_ids[0])
                    edge1 = simple.get_edge_between(node_id, parent_ids[1])

                    dist0 = edge0.distance if edge0 else None
                    dist1 = edge1.distance if edge1 else None

                    created_joint = Revolute(
                        x=x,
                        y=y,
                        joint0=parent0,
                        joint1=parent1,
                        distance0=dist0,
                        distance1=dist1,
                        name=node.name,
                    )

                joints.append(created_joint)
                node_to_joint[node_id] = created_joint
                solve_order.append(created_joint)
                solved_nodes.add(node_id)
                del remaining[node_id]
                break

    if remaining:
        raise ValueError(
            f"Could not determine solve order for nodes: {list(remaining.keys())}. "
            "The hypergraph may be underconstrained or have disconnected components."
        )

    return LinkageClass(
        joints=joints,
        order=solve_order,
        name=hypergraph.name,
    )


def from_linkage(linkage: "Linkage") -> HypergraphLinkage:
    """Convert a joint-based Linkage to a HypergraphLinkage.

    This converts an existing Linkage to the hypergraph representation
    for analysis or manipulation.

    Args:
        linkage: The Linkage to convert.

    Returns:
        A HypergraphLinkage representation.

    Example:
        >>> linkage = Linkage(joints=[...], order=[...])
        >>> hg = from_linkage(linkage)
    """
    from ..joints import Crank, Fixed, Prismatic, Revolute, Static

    hypergraph = HypergraphLinkage(name=linkage.name)

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

        pos = (coord[0], coord[1]) if coord[0] is not None else (None, None)
        anchor_node = Node(
            id=node_id,
            role=NodeRole.GROUND,
            position=pos,
            name=node_id,
        )
        hypergraph.add_node(anchor_node)
        joint_to_node[joint_id] = node_id
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
            position=joint.coord(),
            name=joint.name,
        )

        if isinstance(joint, Crank):
            node.angle = joint.angle

        hypergraph.add_node(node)
        joint_to_node[id(joint)] = node_id

    # Second pass: create edges from joint relationships
    for joint in linkage.joints:
        node_id = joint_to_node[id(joint)]

        if isinstance(joint, Crank):
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint0, joint.joint0.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Revolute):
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint0, joint.joint0.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r0,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint1, joint.joint1.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r1,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Fixed):
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint0, joint.joint0.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.r,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint1, joint.joint1.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=None,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

        elif isinstance(joint, Prismatic):
            if joint.joint0 is not None:
                parent_id = joint_to_node.get(id(joint.joint0))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint0, joint.joint0.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=joint.revolute_radius,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

            if joint.joint1 is not None:
                parent_id = joint_to_node.get(id(joint.joint1))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint1, joint.joint1.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=None,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

            if hasattr(joint, "joint2") and joint.joint2 is not None:
                parent_id = joint_to_node.get(id(joint.joint2))
                if parent_id is None:
                    parent_id = get_or_create_anchor_node(
                        joint.joint2, joint.joint2.coord()
                    )

                edge = Edge(
                    id=f"edge_{edge_counter}",
                    source=parent_id,
                    target=node_id,
                    distance=None,
                )
                hypergraph.add_edge(edge)
                edge_counter += 1

    return hypergraph
