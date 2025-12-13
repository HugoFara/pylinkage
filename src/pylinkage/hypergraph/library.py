"""Pre-defined components for common linkage patterns.

This module provides a component library with ready-to-use linkage building
blocks that can be instantiated and composed into larger mechanisms.
"""

from ._types import NodeRole
from .components import Component, ParameterMapping, ParameterSpec, Port
from .core import Edge, Node
from .graph import HypergraphLinkage

# Component registry - maps component IDs to Component instances
COMPONENT_LIBRARY: dict[str, Component] = {}


def register_component(component: Component) -> Component:
    """Register a component in the library.

    Args:
        component: The component to register.

    Returns:
        The same component (for decorator-style usage).

    Example:
        >>> @register_component
        ... def my_component():
        ...     return Component(...)
    """
    COMPONENT_LIBRARY[component.id] = component
    return component


def get_component(component_id: str) -> Component:
    """Get a component from the library by ID.

    Args:
        component_id: The component ID.

    Returns:
        The Component instance.

    Raises:
        KeyError: If component not found.
    """
    return COMPONENT_LIBRARY[component_id]


def list_components() -> list[str]:
    """List all registered component IDs.

    Returns:
        List of component IDs.
    """
    return list(COMPONENT_LIBRARY.keys())


# ============================================================================
# Pre-built components
# ============================================================================


def create_fourbar_component() -> Component:
    """Create a standard four-bar linkage component.

    The four-bar linkage is the most common planar linkage, consisting of:
    - Two ground joints (A, D)
    - One driver/crank joint (B)
    - One driven/coupler joint (C)

    Topology::

        A -------- B (driver)
        |          |
        |          |  <- coupler
        |          |
        D -------- C (output)

    Ports:
        - ground_left: Left ground point (A)
        - ground_right: Right ground point (D)
        - input: Crank/driver joint (B)
        - output: Coupler point (C)

    Parameters:
        - crank_length: Distance A-B (default 1.0)
        - coupler_length: Distance B-C (default 3.0)
        - rocker_length: Distance C-D (default 2.0)
        - ground_length: Distance A-D (default 3.0)

    Returns:
        A four-bar linkage Component.
    """
    graph = HypergraphLinkage(name="FourBar-Internal")

    # Add nodes
    graph.add_node(Node("A", position=(0.0, 0.0), role=NodeRole.GROUND, name="A"))
    graph.add_node(Node("D", position=(3.0, 0.0), role=NodeRole.GROUND, name="D"))
    graph.add_node(
        Node("B", position=(0.0, 1.0), role=NodeRole.DRIVER, angle=0.1, name="B")
    )
    graph.add_node(Node("C", position=(2.5, 1.5), role=NodeRole.DRIVEN, name="C"))

    # Add edges (links)
    graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))  # crank
    graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))  # coupler
    graph.add_edge(Edge("CD", source="C", target="D", distance=2.0))  # rocker

    return Component(
        id="fourbar",
        internal_graph=graph,
        ports={
            "ground_left": Port("ground_left", "A", "Left ground point"),
            "ground_right": Port("ground_right", "D", "Right ground point"),
            "input": Port("input", "B", "Crank/driver joint"),
            "output": Port("output", "C", "Coupler point"),
        },
        parameters={
            "crank_length": ParameterSpec(
                "crank_length", "Distance A-B", default=1.0, min_value=0.01
            ),
            "coupler_length": ParameterSpec(
                "coupler_length", "Distance B-C", default=3.0, min_value=0.01
            ),
            "rocker_length": ParameterSpec(
                "rocker_length", "Distance C-D", default=2.0, min_value=0.01
            ),
            "ground_length": ParameterSpec(
                "ground_length", "Distance A-D", default=3.0, min_value=0.01
            ),
        },
        parameter_mappings=[
            ParameterMapping("crank_length", edge_ids=["AB"]),
            ParameterMapping("coupler_length", edge_ids=["BC"]),
            ParameterMapping("rocker_length", edge_ids=["CD"]),
        ],
        name="Four-Bar Linkage",
    )


def create_crank_slider_component() -> Component:
    """Create a crank-slider linkage component.

    The crank-slider converts rotary motion to linear motion:
    - One ground joint (A)
    - One driver/crank joint (B)
    - One slider joint (C) that moves along a line

    Topology::

        A -------- B (driver)
                   |
                   |  <- connecting rod
                   |
        ========== C (slider on ground line)

    Ports:
        - ground: Ground/anchor point (A)
        - input: Crank joint (B)
        - slider: Slider output (C)

    Parameters:
        - crank_length: Distance A-B (default 1.0)
        - rod_length: Distance B-C (default 3.0)

    Returns:
        A crank-slider Component.
    """
    graph = HypergraphLinkage(name="CrankSlider-Internal")

    # Add nodes
    graph.add_node(Node("A", position=(0.0, 0.0), role=NodeRole.GROUND, name="A"))
    graph.add_node(
        Node("B", position=(0.0, 1.0), role=NodeRole.DRIVER, angle=0.1, name="B")
    )
    graph.add_node(Node("C", position=(3.0, 0.0), role=NodeRole.DRIVEN, name="C"))

    # Add edges
    graph.add_edge(Edge("AB", source="A", target="B", distance=1.0))  # crank
    graph.add_edge(Edge("BC", source="B", target="C", distance=3.0))  # connecting rod

    return Component(
        id="crank_slider",
        internal_graph=graph,
        ports={
            "ground": Port("ground", "A", "Ground anchor point"),
            "input": Port("input", "B", "Crank/driver joint"),
            "slider": Port("slider", "C", "Slider output"),
        },
        parameters={
            "crank_length": ParameterSpec(
                "crank_length", "Distance A-B", default=1.0, min_value=0.01
            ),
            "rod_length": ParameterSpec(
                "rod_length", "Distance B-C", default=3.0, min_value=0.01
            ),
        },
        parameter_mappings=[
            ParameterMapping("crank_length", edge_ids=["AB"]),
            ParameterMapping("rod_length", edge_ids=["BC"]),
        ],
        name="Crank-Slider Linkage",
    )


def create_dyad_component() -> Component:
    """Create a simple dyad (two-bar) component.

    A dyad is the simplest Assur group - two links connected at a joint,
    with the other ends connected to known points.

    Topology::

        anchor0 -------- internal -------- anchor1

    Ports:
        - anchor0: First anchor point
        - anchor1: Second anchor point
        - output: Internal joint

    Parameters:
        - distance0: Distance from anchor0 to internal (default 1.0)
        - distance1: Distance from anchor1 to internal (default 1.0)

    Returns:
        A dyad Component.
    """
    graph = HypergraphLinkage(name="Dyad-Internal")

    # Add nodes
    graph.add_node(
        Node("anchor0", position=(0.0, 0.0), role=NodeRole.GROUND, name="anchor0")
    )
    graph.add_node(
        Node("anchor1", position=(2.0, 0.0), role=NodeRole.GROUND, name="anchor1")
    )
    graph.add_node(
        Node("internal", position=(1.0, 0.866), role=NodeRole.DRIVEN, name="internal")
    )

    # Add edges
    graph.add_edge(Edge("e0", source="anchor0", target="internal", distance=1.0))
    graph.add_edge(Edge("e1", source="anchor1", target="internal", distance=1.0))

    return Component(
        id="dyad",
        internal_graph=graph,
        ports={
            "anchor0": Port("anchor0", "anchor0", "First anchor point"),
            "anchor1": Port("anchor1", "anchor1", "Second anchor point"),
            "output": Port("output", "internal", "Internal joint"),
        },
        parameters={
            "distance0": ParameterSpec(
                "distance0", "Distance from anchor0", default=1.0, min_value=0.01
            ),
            "distance1": ParameterSpec(
                "distance1", "Distance from anchor1", default=1.0, min_value=0.01
            ),
        },
        parameter_mappings=[
            ParameterMapping("distance0", edge_ids=["e0"]),
            ParameterMapping("distance1", edge_ids=["e1"]),
        ],
        name="Dyad (RRR)",
    )


# ============================================================================
# Register standard components
# ============================================================================

FOURBAR = register_component(create_fourbar_component())
CRANK_SLIDER = register_component(create_crank_slider_component())
DYAD = register_component(create_dyad_component())
