"""AssurMechanism: Wrapper providing Assur analysis for mechanisms.

This module provides the AssurMechanism class, which wraps a Mechanism
and adds formal Assur group analysis capabilities.

The AssurMechanism class represents the key design principle of the assur module:
Assur groups define logical/structural properties, not behaviors. Solving and
simulation are delegated to the Mechanism and solver modules.

Example:
    >>> from pylinkage.mechanism import Mechanism
    >>> from pylinkage.assur import AssurMechanism
    >>>
    >>> mechanism = Mechanism(joints=[...], links=[...])
    >>> assur = AssurMechanism(mechanism)
    >>>
    >>> # Access formal properties
    >>> print(f"DOF: {assur.degree_of_freedom}")
    >>> print(f"Groups: {[g.joint_signature for g in assur.assur_groups]}")
    >>>
    >>> # Simulation still works via delegation
    >>> for positions in assur.step():
    ...     print(positions)
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .decomposition import DecompositionResult, decompose_assur_groups, validate_decomposition
from .mechanism_conversion import graph_to_mechanism, mechanism_to_graph

if TYPE_CHECKING:
    from .._types import Coord, MaybeCoord
    from ..mechanism import Mechanism
    from .graph import LinkageGraph
    from .groups import AssurGroup


@dataclass
class AssurMechanism:
    """Wrapper that adds Assur group analysis to a Mechanism.

    This class provides formal kinematic analysis capabilities without
    modifying the underlying Mechanism's simulation behavior. It embodies
    the design principle that Assur groups define logical properties
    (structure, classification, constraints) rather than behaviors.

    The AssurMechanism:
    - Wraps an existing Mechanism instance
    - Computes and caches Assur group decomposition
    - Provides structural analysis properties (DOF, groups, validation)
    - Delegates simulation to the underlying Mechanism

    Attributes:
        mechanism: The wrapped Mechanism instance.

    Properties:
        decomposition: Cached DecompositionResult.
        assur_groups: List of AssurGroup objects in solving order.
        degree_of_freedom: Computed DOF of the mechanism.
    """

    mechanism: Mechanism
    _decomposition: DecompositionResult | None = field(default=None, repr=False)
    _graph: LinkageGraph | None = field(default=None, repr=False)

    @classmethod
    def from_mechanism(cls, mechanism: Mechanism) -> AssurMechanism:
        """Create AssurMechanism wrapper from an existing Mechanism.

        Args:
            mechanism: The Mechanism to wrap.

        Returns:
            AssurMechanism instance wrapping the mechanism.
        """
        return cls(mechanism=mechanism)

    @classmethod
    def from_graph(cls, graph: LinkageGraph) -> AssurMechanism:
        """Create AssurMechanism from a LinkageGraph.

        Converts the graph to a Mechanism and wraps it.

        Args:
            graph: The LinkageGraph to convert and wrap.

        Returns:
            AssurMechanism instance with the converted mechanism.
        """
        mechanism = graph_to_mechanism(graph)
        instance = cls(mechanism=mechanism)
        instance._graph = graph
        return instance

    @property
    def graph(self) -> LinkageGraph:
        """Get or compute the LinkageGraph representation.

        If created from a graph, returns that graph. Otherwise,
        converts the mechanism to a graph.
        """
        if self._graph is None:
            self._graph = mechanism_to_graph(self.mechanism)
        return self._graph

    @property
    def decomposition(self) -> DecompositionResult:
        """Get or compute the Assur group decomposition.

        The decomposition is computed once and cached. Use analyze()
        to force recomputation.

        Returns:
            DecompositionResult with groups in solving order.
        """
        if self._decomposition is None:
            self._decomposition = decompose_assur_groups(self.graph)
        return self._decomposition

    @property
    def assur_groups(self) -> list[AssurGroup]:
        """Return the Assur groups in solving order.

        Returns:
            List of AssurGroup instances representing the structural
            decomposition of the mechanism.
        """
        return self.decomposition.groups

    @property
    def degree_of_freedom(self) -> int:
        """Compute the degree of freedom using Gruebler's equation.

        For planar mechanisms:
        DOF = 3(n - 1) - 2*j1 - j2

        where:
        - n = number of links (including ground)
        - j1 = number of 1-DOF joints (revolute, prismatic)
        - j2 = number of 2-DOF joints

        Returns:
            The computed degree of freedom.
        """
        # Count links (each Link is one link, ground counts as 1)
        n_links = len(self.mechanism.links)

        # Count joints by DOF
        j1 = 0  # 1-DOF joints (revolute, prismatic)
        j2 = 0  # 2-DOF joints (we don't have these yet)

        for _joint in self.mechanism.joints:
            # All current joint types are 1-DOF
            j1 += 1

        # Gruebler's equation for planar mechanisms
        dof = 3 * (n_links - 1) - 2 * j1 - j2
        return dof

    @property
    def num_ground_nodes(self) -> int:
        """Return the number of ground nodes."""
        return len(self.decomposition.ground)

    @property
    def num_driver_nodes(self) -> int:
        """Return the number of driver nodes."""
        return len(self.decomposition.drivers)

    @property
    def num_assur_groups(self) -> int:
        """Return the number of Assur groups."""
        return len(self.decomposition.groups)

    def analyze(self) -> DecompositionResult:
        """Force recomputation of the decomposition.

        This clears the cached decomposition and graph, then
        recomputes the decomposition from the current mechanism state.

        Returns:
            The newly computed DecompositionResult.
        """
        self._graph = None
        self._decomposition = None
        return self.decomposition

    def validate(self) -> list[str]:
        """Validate the mechanism structure.

        Checks for common structural issues:
        - No ground nodes
        - No driver nodes
        - Nodes not accounted for in groups

        Returns:
            List of warning/error messages. Empty if valid.
        """
        return validate_decomposition(self.decomposition)

    def is_valid(self) -> bool:
        """Check if the mechanism structure is valid.

        Returns:
            True if validate() returns no messages.
        """
        return len(self.validate()) == 0

    # Delegation methods

    def step(self, dt: float = 1.0) -> Generator[tuple[MaybeCoord, ...], None, None]:
        """Simulate one full rotation cycle.

        Delegates to the underlying mechanism's step() method.

        Args:
            dt: Time step (default 1.0).

        Yields:
            Tuple of (x, y) positions for each joint at each step.
        """
        return self.mechanism.step(dt)

    def get_joint_positions(self) -> list[Coord]:
        """Get current positions of all joints.

        Delegates to the underlying mechanism.

        Returns:
            List of (x, y) positions for each joint.
        """
        return self.mechanism.get_joint_positions()

    def reset(self) -> None:
        """Reset the mechanism to initial state.

        Delegates to the underlying mechanism.
        """
        self.mechanism.reset()
