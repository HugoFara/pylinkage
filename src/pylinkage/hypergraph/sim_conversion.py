"""Conversion between ``simulation.Linkage`` and ``HypergraphLinkage``.

``simulation.Linkage`` (a.k.a. "SimLinkage") is the joint-free
component/actuator/dyad API produced by ``pylinkage.synthesis`` and
``pylinkage.optimization.co_optimize``. This module converts one of
those objects into the topological ``HypergraphLinkage`` plus a
companion ``Dimensions`` record so downstream tooling
(``leggedsnake.Walker``, exporters) can consume synthesis output
directly.

Only the component types that the upstream synthesis / catalog
co-optimization actually emit are handled:

- ``Ground`` → ``NodeRole.GROUND``
- ``Crank`` / ``ArcCrank`` → ``NodeRole.DRIVER`` with a
  :class:`~pylinkage.dimensions.DriverAngle`
- ``LinearActuator`` → ``NodeRole.DRIVER`` (modelled as a
  prismatic-joint node; the prismatic axis lives on the output node)
- ``RRRDyad`` → two edges (one per distance constraint)
- ``FixedDyad`` → a hyperedge over the rigid triangle plus two edges
  carrying the two leg distances as numeric fallbacks
- ``RRPDyad`` → one edge for the revolute leg; the prismatic track is
  encoded as a hyperedge over the three line/anchor nodes
- ``PPDyad`` → a hyperedge over the four line anchors (no numeric
  distances — it is fully determined by the four line anchors)

Anything else raises :class:`NotImplementedError` so callers know to
extend the bridge when pylinkage grows new component types.
"""

from __future__ import annotations

from math import hypot, tau
from typing import TYPE_CHECKING, Any

from .._types import JointType, NodeRole
from ..dimensions import Dimensions, DriverAngle
from .core import Edge, Hyperedge, Node
from .graph import HypergraphLinkage

if TYPE_CHECKING:
    from ..simulation import Linkage as SimLinkage


def _anchor_parent(anchor: Any) -> Any:
    """Resolve an anchor (or ``_AnchorProxy``) to its underlying component."""
    return getattr(anchor, "_parent", anchor)


def from_sim_linkage(sim_linkage: SimLinkage) -> tuple[HypergraphLinkage, Dimensions]:
    """Convert a :class:`pylinkage.simulation.Linkage` to a hypergraph.

    Args:
        sim_linkage: A :class:`simulation.Linkage` built from the modern
            component/actuator/dyad API.

    Returns:
        A tuple ``(hypergraph, dimensions)`` suitable for feeding into
        :func:`pylinkage.hypergraph.to_mechanism` or into any
        hypergraph-native consumer.

    Raises:
        TypeError: If ``sim_linkage`` does not look like a
            ``simulation.Linkage`` (no ``.components``).
        NotImplementedError: If a component type is encountered that
            this bridge does not yet handle.
    """
    from ..actuators import ArcCrank, Crank, LinearActuator
    from ..components import Ground
    from ..dyads import FixedDyad, PPDyad, RRPDyad, RRRDyad

    components = getattr(sim_linkage, "components", None)
    if components is None:
        raise TypeError(
            f"Expected a simulation.Linkage, got {type(sim_linkage).__name__}"
        )

    name = getattr(sim_linkage, "name", "") or ""
    hg = HypergraphLinkage(name=name)

    node_positions: dict[str, tuple[float, float]] = {}
    edge_distances: dict[str, float] = {}
    driver_angles: dict[str, DriverAngle] = {}
    hyperedge_constraints: dict[str, dict[tuple[str, str], float]] = {}
    component_to_node: dict[int, str] = {}

    # Pass 1: one node per component.
    for i, comp in enumerate(components):
        if isinstance(comp, Ground):
            role = NodeRole.GROUND
            joint_type = JointType.REVOLUTE
        elif isinstance(comp, (Crank, ArcCrank)):
            role = NodeRole.DRIVER
            joint_type = JointType.REVOLUTE
        elif isinstance(comp, LinearActuator):
            role = NodeRole.DRIVER
            joint_type = JointType.PRISMATIC
        else:
            role = NodeRole.DRIVEN
            joint_type = JointType.REVOLUTE

        node_id = getattr(comp, "name", None) or f"n{i}"
        hg.add_node(Node(node_id, joint_type=joint_type, role=role, name=node_id))
        component_to_node[id(comp)] = node_id

        x = getattr(comp, "x", None)
        y = getattr(comp, "y", None)
        if x is not None and y is not None:
            node_positions[node_id] = (float(x), float(y))

    def _resolve(anchor: Any) -> str:
        parent = _anchor_parent(anchor)
        key = id(parent)
        if key not in component_to_node:
            raise NotImplementedError(
                f"Anchor {anchor!r} references a component not registered in "
                "the SimLinkage; cannot build the hypergraph."
            )
        return component_to_node[key]

    edge_counter = 0

    def _fresh_edge_id() -> str:
        nonlocal edge_counter
        eid = f"e{edge_counter}"
        edge_counter += 1
        return eid

    def _fresh_hyperedge_id() -> str:
        nonlocal edge_counter
        heid = f"he{edge_counter}"
        edge_counter += 1
        return heid

    # Pass 2: edges / hyperedges + driver angles.
    for comp in components:
        this_id = component_to_node[id(comp)]

        if isinstance(comp, Ground):
            continue

        if isinstance(comp, (Crank, ArcCrank)):
            anchor_id = _resolve(comp.anchor)
            eid = _fresh_edge_id()
            hg.add_edge(Edge(eid, anchor_id, this_id))
            radius_attr: Any = getattr(comp, "radius", None)
            if radius_attr is None:
                radius_attr = getattr(comp, "distance", 1.0)
            edge_distances[eid] = float(radius_attr)
            omega = float(getattr(comp, "angular_velocity", tau / 360.0))
            initial_angle = float(getattr(comp, "initial_angle", 0.0) or 0.0)
            driver_angles[this_id] = DriverAngle(
                angular_velocity=omega, initial_angle=initial_angle
            )
            continue

        if isinstance(comp, LinearActuator):
            anchor_id = _resolve(comp.anchor)
            eid = _fresh_edge_id()
            hg.add_edge(Edge(eid, anchor_id, this_id))
            stroke = float(getattr(comp, "stroke", 0.0) or 0.0)
            edge_distances[eid] = stroke
            # ``speed`` is the linear-actuator analogue of crank
            # angular_velocity; ``angle`` gives the prismatic-axis
            # direction.
            speed = float(getattr(comp, "speed", 0.0) or 0.0)
            driver_angles[this_id] = DriverAngle(
                angular_velocity=speed,
                initial_angle=float(getattr(comp, "angle", 0.0) or 0.0),
            )
            continue

        if isinstance(comp, RRRDyad):
            a1 = _resolve(comp.anchor1)
            a2 = _resolve(comp.anchor2)
            e1 = _fresh_edge_id()
            e2 = _fresh_edge_id()
            hg.add_edge(Edge(e1, a1, this_id))
            hg.add_edge(Edge(e2, a2, this_id))
            edge_distances[e1] = float(comp.distance1)
            edge_distances[e2] = float(comp.distance2)
            continue

        if isinstance(comp, FixedDyad):
            a1 = _resolve(comp.anchor1)
            a2 = _resolve(comp.anchor2)
            he_id = _fresh_hyperedge_id()
            hg.add_hyperedge(Hyperedge(he_id, (a1, a2, this_id)))
            # Leg distances. anchor1 → this is the declared ``distance``;
            # anchor2 → this is recovered from current positions so the
            # resulting Dimensions is self-consistent.
            e1 = _fresh_edge_id()
            e2 = _fresh_edge_id()
            hg.add_edge(Edge(e1, a1, this_id))
            hg.add_edge(Edge(e2, a2, this_id))
            d1 = float(getattr(comp, "distance", 0.0))
            edge_distances[e1] = d1
            a2_pos = node_positions.get(a2)
            this_pos = node_positions.get(this_id)
            if a2_pos is not None and this_pos is not None:
                edge_distances[e2] = hypot(
                    this_pos[0] - a2_pos[0], this_pos[1] - a2_pos[1]
                )
            hyperedge_constraints[he_id] = {
                (min(a1, this_id), max(a1, this_id)): d1,
            }
            continue

        if isinstance(comp, RRPDyad):
            rev = _resolve(comp.revolute_anchor)
            la1 = _resolve(comp.line_anchor1)
            la2 = _resolve(comp.line_anchor2)
            # Revolute leg carries the distance constraint.
            eid = _fresh_edge_id()
            hg.add_edge(Edge(eid, rev, this_id))
            edge_distances[eid] = float(comp.distance)
            # Prismatic track captured as a ternary hyperedge on the line.
            he_id = _fresh_hyperedge_id()
            hg.add_hyperedge(Hyperedge(he_id, (la1, la2, this_id)))
            continue

        if isinstance(comp, PPDyad):
            l1a = _resolve(comp.line1_anchor1)
            l1b = _resolve(comp.line1_anchor2)
            l2a = _resolve(comp.line2_anchor1)
            l2b = _resolve(comp.line2_anchor2)
            he_id = _fresh_hyperedge_id()
            hg.add_hyperedge(Hyperedge(he_id, (l1a, l1b, l2a, l2b, this_id)))
            continue

        raise NotImplementedError(
            f"Component type {type(comp).__name__!r} is not yet supported "
            "by from_sim_linkage(). Extend pylinkage.hypergraph.sim_conversion "
            "with the required topology."
        )

    dimensions = Dimensions(
        node_positions=node_positions,
        driver_angles=driver_angles,
        edge_distances=edge_distances,
        hyperedge_constraints=hyperedge_constraints,
        name=name,
    )
    return hg, dimensions
