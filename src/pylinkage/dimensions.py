"""Dimensional data for linkage mechanisms.

This module provides the Dimensions class which holds all geometric data
(positions, distances, angles) separate from topology. This separation
allows pure topological analysis without geometric constraints.

The Dimensions class is shared by both the hypergraph and assur modules.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class DriverAngle:
    """Angular parameters for a driver joint.

    Attributes:
        angular_velocity: Rotation angle per simulation step (radians).
        initial_angle: Starting angle of the driver (radians).
    """

    angular_velocity: float
    initial_angle: float = 0.0


@dataclass
class Dimensions:
    """Geometric data for a linkage topology.

    Holds all dimensional information separate from the topological structure.
    This allows the same topology to be instantiated with different dimensions.

    Attributes:
        node_positions: Mapping from node ID to (x, y) coordinates.
        driver_angles: Mapping from driver node ID to angular parameters.
        edge_distances: Mapping from edge ID to link length.
        hyperedge_constraints: Mapping from hyperedge ID to pairwise distance
            constraints. Each value is a dict mapping (node1, node2) pairs
            to distances.
        name: Optional name for this dimension set.

    Example:
        >>> dims = Dimensions(
        ...     node_positions={"A": (0.0, 0.0), "B": (1.0, 0.0)},
        ...     driver_angles={"B": DriverAngle(0.1, 0.0)},
        ...     edge_distances={"AB": 1.0},
        ... )
    """

    node_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    driver_angles: dict[str, DriverAngle] = field(default_factory=dict)
    edge_distances: dict[str, float] = field(default_factory=dict)
    hyperedge_constraints: dict[str, dict[tuple[str, str], float]] = field(default_factory=dict)
    name: str = ""

    def copy(self) -> "Dimensions":
        """Create a deep copy of this Dimensions object.

        Returns:
            A new Dimensions object with copied data.
        """
        return Dimensions(
            node_positions=dict(self.node_positions),
            driver_angles={
                k: DriverAngle(v.angular_velocity, v.initial_angle)
                for k, v in self.driver_angles.items()
            },
            edge_distances=dict(self.edge_distances),
            hyperedge_constraints={k: dict(v) for k, v in self.hyperedge_constraints.items()},
            name=self.name,
        )

    def validate_against(
        self,
        node_ids: "Iterable[str]",
        edge_ids: "Iterable[str]",
        hyperedge_ids: "Iterable[str] | None" = None,
    ) -> list[str]:
        """Validate that dimensions are compatible with a topology.

        Checks that all referenced node/edge IDs exist in the provided
        topology ID sets.

        Args:
            node_ids: Valid node IDs from the topology.
            edge_ids: Valid edge IDs from the topology.
            hyperedge_ids: Valid hyperedge IDs from the topology (optional).

        Returns:
            List of error messages. Empty list means validation passed.

        Example:
            >>> dims = Dimensions(node_positions={"A": (0, 0), "X": (1, 1)})
            >>> errors = dims.validate_against(["A", "B"], ["AB"])
            >>> "X" in errors[0]  # "X" is not a valid node
            True
        """
        errors: list[str] = []
        node_set = set(node_ids)
        edge_set = set(edge_ids)
        hyperedge_set = set(hyperedge_ids) if hyperedge_ids is not None else set()

        # Check node positions
        for node_id in self.node_positions:
            if node_id not in node_set:
                errors.append(f"Unknown node '{node_id}' in node_positions")

        # Check driver angles
        for node_id in self.driver_angles:
            if node_id not in node_set:
                errors.append(f"Unknown node '{node_id}' in driver_angles")

        # Check edge distances
        for edge_id in self.edge_distances:
            if edge_id not in edge_set:
                errors.append(f"Unknown edge '{edge_id}' in edge_distances")

        # Check hyperedge constraints
        if hyperedge_ids is not None:
            for he_id in self.hyperedge_constraints:
                if he_id not in hyperedge_set:
                    errors.append(f"Unknown hyperedge '{he_id}' in hyperedge_constraints")

        return errors

    def get_node_position(self, node_id: str) -> tuple[float, float] | None:
        """Get the position of a node.

        Args:
            node_id: The node identifier.

        Returns:
            The (x, y) position, or None if not defined.
        """
        return self.node_positions.get(node_id)

    def get_edge_distance(self, edge_id: str) -> float | None:
        """Get the distance constraint for an edge.

        Args:
            edge_id: The edge identifier.

        Returns:
            The distance, or None if not defined.
        """
        return self.edge_distances.get(edge_id)

    def get_driver_angle(self, node_id: str) -> DriverAngle | None:
        """Get the driver angle parameters for a node.

        Args:
            node_id: The node identifier.

        Returns:
            The DriverAngle, or None if not defined.
        """
        return self.driver_angles.get(node_id)

    def get_hyperedge_distance(self, hyperedge_id: str, node1: str, node2: str) -> float | None:
        """Get a pairwise distance constraint from a hyperedge.

        Args:
            hyperedge_id: The hyperedge identifier.
            node1: First node in the pair.
            node2: Second node in the pair.

        Returns:
            The distance, or None if not defined.
        """
        constraints = self.hyperedge_constraints.get(hyperedge_id)
        if constraints is None:
            return None
        # Try both orderings
        key = (min(node1, node2), max(node1, node2))
        return constraints.get(key)
