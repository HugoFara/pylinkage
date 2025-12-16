"""MechanismBuilder - Links-first approach to mechanism definition.

This module provides a builder pattern for creating Mechanism objects
where users define links with their intrinsic properties (lengths, port
geometry) first, then connect them with joints. Joint positions are
computed automatically during assembly.

Example:
    >>> from pylinkage.mechanism import MechanismBuilder
    >>> mechanism = (
    ...     MechanismBuilder("four-bar")
    ...     .add_ground_link("ground", ports={"O1": (0, 0), "O2": (3, 0)})
    ...     .add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
    ...     .add_link("coupler", length=2.5)
    ...     .add_link("rocker", length=1.5)
    ...     .connect("crank.tip", "coupler.0")
    ...     .connect("coupler.1", "rocker.0")
    ...     .connect("rocker.1", "ground.O2")
    ...     .build()
    ... )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from typing_extensions import Self

from ..exceptions import UnbuildableError, UnderconstrainedError
from ..geometry.secants import (
    INTERSECTION_NONE,
    circle_intersect,
    circle_line_from_points_intersection,
)
from .joint import GroundJoint, PrismaticJoint, RevoluteJoint
from .link import DriverLink, GroundLink, Link
from .mechanism import Mechanism


@dataclass
class Port:
    """A connection point on a link.

    Ports define where joints can be placed on a link.
    For binary links: 2 ports (endpoints, named "0" and "1" or custom)
    For ternary links: 3 ports (triangle vertices)

    Attributes:
        id: Unique identifier within the link.
        local_position: Position relative to link's local frame.
            For binary links this is None (determined by length).
            For ternary+ links this is (x, y) in local coordinates.
    """

    id: str
    local_position: tuple[float, float] | None = None


@dataclass
class PendingLink:
    """A link awaiting assembly.

    Stores link definition before joint positions are computed.

    Attributes:
        id: Unique identifier for the link.
        ports: Dictionary mapping port ID to Port object.
        length: For binary links, the distance between ports.
        port_geometry: For ternary+ links, local coordinates of each port.
        is_driver: True if this link is motor-driven.
        motor_port: For driver links, the ground port where motor attaches.
        angular_velocity: For driver links, rotation rate in rad/step.
        initial_angle: For driver links, starting angle in radians.
    """

    id: str
    ports: dict[str, Port] = field(default_factory=dict)
    length: float | None = None
    port_geometry: dict[str, tuple[float, float]] | None = None
    is_driver: bool = False
    motor_port: str | None = None
    angular_velocity: float = 0.0
    initial_angle: float = 0.0

    def get_port_distance(self, port1_id: str, port2_id: str) -> float | None:
        """Get distance between two ports on this link.

        For binary links, returns the length.
        For ternary+ links, computes from port_geometry.
        """
        if self.length is not None and len(self.ports) == 2:
            return self.length

        if (
            self.port_geometry is not None
            and port1_id in self.port_geometry
            and port2_id in self.port_geometry
        ):
            p1 = self.port_geometry[port1_id]
            p2 = self.port_geometry[port2_id]
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        return None


@dataclass
class Connection:
    """A connection between two ports on different links.

    Attributes:
        port1: Full port identifier "link_id.port_id".
        port2: Full port identifier "link_id.port_id".
        joint_type: Type of joint ("revolute" or "prismatic").
    """

    port1: str
    port2: str
    joint_type: str = "revolute"


@dataclass
class SlideAxis:
    """Definition of a prismatic joint axis.

    Attributes:
        id: Unique identifier for the axis.
        point: A point on the slide line.
        direction: Direction vector (will be normalized).
    """

    id: str
    point: tuple[float, float]
    direction: tuple[float, float]

    def get_normalized_direction(self) -> tuple[float, float]:
        """Return normalized direction vector."""
        dx, dy = self.direction
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return (1.0, 0.0)
        return (dx / length, dy / length)

    def get_line_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return two points on the line for intersection computation."""
        x, y = self.point
        dx, dy = self.get_normalized_direction()
        return ((x, y), (x + dx * 100, y + dy * 100))


@dataclass
class PrismaticConnection:
    """A connection of a port to a slide axis.

    Attributes:
        port: Full port identifier "link_id.port_id".
        axis_id: ID of the SlideAxis.
    """

    port: str
    axis_id: str


@dataclass
class MechanismBuilder:
    """Builder for creating Mechanism objects using a links-first approach.

    This builder allows defining mechanisms by specifying link properties
    (lengths, port geometry) rather than joint positions. Joint positions
    are computed automatically during the build() step.

    Example:
        >>> builder = MechanismBuilder("four-bar")
        >>> builder.add_ground_link("ground", ports={"O1": (0, 0), "O2": (3, 0)})
        >>> builder.add_driver_link("crank", length=1.0, motor_port="O1")
        >>> builder.add_link("coupler", length=2.5)
        >>> builder.add_link("rocker", length=1.5)
        >>> builder.connect("crank.tip", "coupler.0")
        >>> builder.connect("coupler.1", "rocker.0")
        >>> builder.connect("rocker.1", "ground.O2")
        >>> mechanism = builder.build()
    """

    name: str = ""

    # Internal storage
    _ground_link_id: str | None = field(default=None, repr=False)
    _ground_ports: dict[str, tuple[float, float]] = field(
        default_factory=dict, repr=False
    )
    _pending_links: dict[str, PendingLink] = field(default_factory=dict, repr=False)
    _connections: list[Connection] = field(default_factory=list, repr=False)
    _prismatic_connections: list[PrismaticConnection] = field(
        default_factory=list, repr=False
    )
    _slide_axes: dict[str, SlideAxis] = field(default_factory=dict, repr=False)
    _configuration: dict[str, int] = field(default_factory=dict, repr=False)

    def add_ground_link(
        self,
        id: str,
        ports: dict[str, tuple[float, float]],
    ) -> Self:
        """Add the ground (frame) link with fixed port positions.

        The ground link represents the stationary frame of the mechanism.
        Each port on the ground link has a fixed position in the global
        coordinate system.

        Args:
            id: Unique identifier for the ground link.
            ports: Dictionary mapping port names to (x, y) positions.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_ground_link("ground", ports={"O1": (0, 0), "O2": (3, 0)})
        """
        self._ground_link_id = id
        self._ground_ports = dict(ports)
        return self

    def add_driver_link(
        self,
        id: str,
        length: float,
        motor_port: str,
        omega: float = 0.1,
        initial_angle: float = 0.0,
    ) -> Self:
        """Add a motor-driven link (crank).

        A driver link rotates around a ground port at a specified angular
        velocity. It has two ports: the motor port (at the ground) and
        the output port ("tip") at distance `length` from the motor.

        Args:
            id: Unique identifier for the link.
            length: Distance from motor to output (crank radius).
            motor_port: Name of the ground port where motor attaches.
            omega: Angular velocity in radians per step.
            initial_angle: Starting angle in radians (from positive x-axis).

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_driver_link("crank", length=1.0, motor_port="O1", omega=0.1)
        """
        link = PendingLink(
            id=id,
            ports={"motor": Port("motor"), "tip": Port("tip")},
            length=length,
            is_driver=True,
            motor_port=motor_port,
            angular_velocity=omega,
            initial_angle=initial_angle,
        )
        self._pending_links[id] = link
        return self

    def add_link(
        self,
        id: str,
        length: float,
    ) -> Self:
        """Add a binary link with given length.

        A binary link has two ports (connection points) named "0" and "1",
        separated by the specified length.

        Args:
            id: Unique identifier for the link.
            length: Distance between the two ports.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_link("coupler", length=2.5)
        """
        link = PendingLink(
            id=id,
            ports={"0": Port("0"), "1": Port("1")},
            length=length,
        )
        self._pending_links[id] = link
        return self

    def add_ternary_link(
        self,
        id: str,
        port_geometry: dict[str, tuple[float, float]],
    ) -> Self:
        """Add a ternary (3-port) link with triangle geometry.

        A ternary link has three ports arranged in a triangle. The geometry
        is specified as local coordinates for each port, from which all
        pairwise distances are derived.

        Args:
            id: Unique identifier for the link.
            port_geometry: Dictionary mapping port names to (x, y) positions
                in the link's local coordinate frame.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If port_geometry doesn't have exactly 3 ports.

        Example:
            >>> builder.add_ternary_link(
            ...     "coupler",
            ...     port_geometry={"A": (0, 0), "B": (3, 0), "P": (1.5, 1)}
            ... )
        """
        if len(port_geometry) != 3:
            raise ValueError(f"Ternary link must have exactly 3 ports, got {len(port_geometry)}")

        ports = {name: Port(name, pos) for name, pos in port_geometry.items()}
        link = PendingLink(
            id=id,
            ports=ports,
            port_geometry=port_geometry,
        )
        self._pending_links[id] = link
        return self

    def add_quaternary_link(
        self,
        id: str,
        port_geometry: dict[str, tuple[float, float]],
    ) -> Self:
        """Add a quaternary (4-port) link.

        A quaternary link has four ports. The geometry is specified as
        local coordinates for each port.

        Args:
            id: Unique identifier for the link.
            port_geometry: Dictionary mapping port names to (x, y) positions
                in the link's local coordinate frame.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If port_geometry doesn't have exactly 4 ports.
        """
        if len(port_geometry) != 4:
            raise ValueError(f"Quaternary link must have exactly 4 ports, got {len(port_geometry)}")

        ports = {name: Port(name, pos) for name, pos in port_geometry.items()}
        link = PendingLink(
            id=id,
            ports=ports,
            port_geometry=port_geometry,
        )
        self._pending_links[id] = link
        return self

    def add_slide_axis(
        self,
        id: str,
        through: tuple[float, float],
        direction: tuple[float, float],
    ) -> Self:
        """Define a slide axis for prismatic joints.

        A slide axis is a line along which a prismatic joint can translate.

        Args:
            id: Unique identifier for the axis.
            through: A point (x, y) that the line passes through.
            direction: Direction vector (dx, dy) of the line.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_slide_axis("rail", through=(0, 0), direction=(1, 0))
        """
        self._slide_axes[id] = SlideAxis(id, through, direction)
        return self

    def connect(self, port1: str, port2: str) -> Self:
        """Connect two ports with a revolute joint.

        Creates a pin joint between two ports on different links. The port
        identifiers use the format "link_id.port_id".

        Args:
            port1: First port identifier (e.g., "crank.tip").
            port2: Second port identifier (e.g., "coupler.0").

        Returns:
            Self for method chaining.

        Example:
            >>> builder.connect("crank.tip", "coupler.0")
        """
        self._connections.append(Connection(port1, port2, "revolute"))
        return self

    def connect_prismatic(self, port: str, axis: str) -> Self:
        """Connect a port to a slide axis with a prismatic joint.

        Creates a slider joint that constrains the port to move along
        the specified axis.

        Args:
            port: Port identifier (e.g., "rod.1").
            axis: Slide axis identifier.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.connect_prismatic("rod.1", "rail")
        """
        self._prismatic_connections.append(PrismaticConnection(port, axis))
        return self

    def set_branch(self, joint: str, branch: int) -> Self:
        """Set the assembly branch for a joint with two solutions.

        When computing joint positions via circle-circle intersection,
        there are typically two solutions. This method allows selecting
        which solution to use.

        Args:
            joint: Joint identifier (typically "link_id.port_id").
            branch: 0 for first solution, 1 for second solution.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_branch("coupler.1", 1)  # Use "lower" configuration
        """
        self._configuration[joint] = branch
        return self

    def build(self) -> Mechanism:
        """Assemble and return the Mechanism.

        Validates the link definitions, computes all joint positions
        from constraints, and creates a Mechanism object.

        Returns:
            Assembled Mechanism ready for simulation.

        Raises:
            UnbuildableError: If geometry is impossible (links can't reach).
            UnderconstrainedError: If system is underconstrained.
            ValueError: If link definitions are invalid.
        """
        self._validate()
        positions = self._assemble()
        return self._create_mechanism(positions)

    def _validate(self) -> None:
        """Validate all link definitions before assembly."""
        errors: list[str] = []

        # Check ground link exists
        if not self._ground_ports:
            errors.append("No ground link defined (use add_ground_link)")

        # Check all connections reference valid ports
        all_ports = self._get_all_port_ids()
        for conn in self._connections:
            if conn.port1 not in all_ports:
                errors.append(f"Unknown port: {conn.port1}")
            if conn.port2 not in all_ports:
                errors.append(f"Unknown port: {conn.port2}")

        # Check prismatic connections reference valid ports and axes
        for pconn in self._prismatic_connections:
            if pconn.port not in all_ports:
                errors.append(f"Unknown port: {pconn.port}")
            if pconn.axis_id not in self._slide_axes:
                errors.append(f"Unknown slide axis: {pconn.axis_id}")

        # Check driver links reference valid ground ports
        for link in self._pending_links.values():
            if link.is_driver and link.motor_port not in self._ground_ports:
                errors.append(
                    f"Driver link '{link.id}' references unknown ground port: {link.motor_port}"
                )

        # Validate ternary link geometry (triangle inequality)
        for link in self._pending_links.values():
            if link.port_geometry and len(link.port_geometry) == 3:
                self._validate_triangle(link, errors)

        if errors:
            raise ValueError("Invalid mechanism definition:\n" + "\n".join(f"  - {e}" for e in errors))

    def _validate_triangle(self, link: PendingLink, errors: list[str]) -> None:
        """Validate that ternary link satisfies triangle inequality."""
        ports = list(link.port_geometry.keys())  # type: ignore[union-attr]
        a = link.get_port_distance(ports[0], ports[1]) or 0
        b = link.get_port_distance(ports[1], ports[2]) or 0
        c = link.get_port_distance(ports[0], ports[2]) or 0

        if not (a + b > c and b + c > a and a + c > b):
            errors.append(
                f"Ternary link '{link.id}' violates triangle inequality: "
                f"sides {a:.2f}, {b:.2f}, {c:.2f}"
            )

    def _get_all_port_ids(self) -> set[str]:
        """Get all valid port identifiers."""
        ports: set[str] = set()

        # Ground ports
        if self._ground_link_id:
            for port_name in self._ground_ports:
                ports.add(f"{self._ground_link_id}.{port_name}")

        # Link ports
        for link in self._pending_links.values():
            for port_name in link.ports:
                ports.add(f"{link.id}.{port_name}")

        return ports

    def _assemble(self) -> dict[str, tuple[float, float]]:
        """Compute all joint positions from constraints.

        Returns:
            Dictionary mapping port IDs to (x, y) positions.
        """
        solved: dict[str, tuple[float, float]] = {}

        # 1. Ground ports are known
        if self._ground_link_id:
            for port_name, pos in self._ground_ports.items():
                solved[f"{self._ground_link_id}.{port_name}"] = pos

        # 2. Compute driver output positions from initial angles
        for link in self._pending_links.values():
            if link.is_driver and link.motor_port and link.length:
                motor_key = f"{self._ground_link_id}.{link.motor_port}"
                if motor_key in solved:
                    motor_pos = solved[motor_key]
                    angle = link.initial_angle
                    tip_x = motor_pos[0] + link.length * math.cos(angle)
                    tip_y = motor_pos[1] + link.length * math.sin(angle)
                    solved[f"{link.id}.tip"] = (tip_x, tip_y)
                    # Motor port on the driver link is at the same position as ground port
                    solved[f"{link.id}.motor"] = motor_pos

        # 3. Propagate all initial connections
        # This ensures that connected ports share the same position
        changed = True
        while changed:
            changed = False
            for conn in self._connections:
                if conn.port1 in solved and conn.port2 not in solved:
                    solved[conn.port2] = solved[conn.port1]
                    changed = True
                elif conn.port2 in solved and conn.port1 not in solved:
                    solved[conn.port1] = solved[conn.port2]
                    changed = True

        # 4. Iteratively solve remaining joints
        pending = self._get_unsolved_ports(solved)
        max_iterations = len(pending) * 2 + 10

        for _ in range(max_iterations):
            if not pending:
                break

            progress = False
            for port_id in list(pending):
                # Skip if already solved via propagation
                if port_id not in pending:
                    continue

                maybe_pos = self._try_solve_port(port_id, solved)
                if maybe_pos is not None:
                    solved[port_id] = maybe_pos
                    pending.discard(port_id)
                    progress = True

                    # If this port is connected to others, they share the same position
                    self._propagate_connections(port_id, maybe_pos, solved, pending)

            if not progress:
                raise UnderconstrainedError(
                    f"Cannot solve ports: {pending}. "
                    "Check that all links are properly connected."
                )

        return solved

    def _get_unsolved_ports(self, solved: dict[str, tuple[float, float]]) -> set[str]:
        """Get all port IDs that haven't been solved yet."""
        all_ports = self._get_all_port_ids()
        return all_ports - set(solved.keys())

    def _propagate_connections(
        self,
        port_id: str,
        pos: tuple[float, float],
        solved: dict[str, tuple[float, float]],
        pending: set[str],
    ) -> None:
        """When a port is solved, connected ports get the same position."""
        for conn in self._connections:
            if conn.port1 == port_id and conn.port2 not in solved:
                solved[conn.port2] = pos
                pending.discard(conn.port2)
            elif conn.port2 == port_id and conn.port1 not in solved:
                solved[conn.port1] = pos
                pending.discard(conn.port1)

    def _try_solve_port(
        self,
        port_id: str,
        solved: dict[str, tuple[float, float]],
    ) -> tuple[float, float] | None:
        """Try to solve a port position from already-solved neighbors.

        Returns the position if solvable, None otherwise.
        """
        # Get constraints: (center, radius) pairs for circle constraints
        constraints = self._get_constraints_for(port_id, solved)

        # Check for prismatic constraint
        prismatic_axis = self._get_prismatic_axis_for(port_id)

        if prismatic_axis and len(constraints) >= 1:
            # Circle-line intersection (RRP)
            return self._solve_circle_line(constraints[0], prismatic_axis, port_id)
        elif len(constraints) >= 2:
            # Circle-circle intersection (RRR)
            return self._solve_circle_circle(constraints[0], constraints[1], port_id)

        return None

    def _get_constraints_for(
        self,
        port_id: str,
        solved: dict[str, tuple[float, float]],
    ) -> list[tuple[tuple[float, float], float]]:
        """Get distance constraints for a port from solved neighbors.

        This gathers constraints from:
        1. Other ports on the same link (direct distance constraint)
        2. Other ports on links connected at this joint (via connected ports)

        Returns list of (center, radius) tuples.
        """
        constraints: list[tuple[tuple[float, float], float]] = []

        # Get all ports that share this joint (are connected to this port)
        joint_ports = self._get_connected_ports(port_id)
        joint_ports.add(port_id)

        # For each port in the joint, check its link's constraints
        for jp in joint_ports:
            link_id, port_name = jp.rsplit(".", 1)

            # Skip ground link ports (they're fixed, not constraints)
            if link_id == self._ground_link_id:
                continue

            link = self._pending_links.get(link_id)
            if not link:
                continue

            # Check other ports on this link
            for other_port_name in link.ports:
                if other_port_name == port_name:
                    continue

                other_port_id = f"{link_id}.{other_port_name}"

                # Check if this other port (or any port connected to it) is solved
                other_position = self._get_solved_position(other_port_id, solved)
                if other_position is not None:
                    distance = link.get_port_distance(port_name, other_port_name)
                    if distance is not None:
                        # Avoid duplicate constraints
                        constraint = (other_position, distance)
                        if constraint not in constraints:
                            constraints.append(constraint)

        return constraints

    def _get_connected_ports(self, port_id: str) -> set[str]:
        """Get all ports directly connected to this port."""
        connected: set[str] = set()
        for conn in self._connections:
            if conn.port1 == port_id:
                connected.add(conn.port2)
            elif conn.port2 == port_id:
                connected.add(conn.port1)
        return connected

    def _get_solved_position(
        self,
        port_id: str,
        solved: dict[str, tuple[float, float]],
    ) -> tuple[float, float] | None:
        """Get position for a port, checking connected ports if not directly solved."""
        if port_id in solved:
            return solved[port_id]

        # Check if any connected port is solved
        for conn in self._connections:
            if conn.port1 == port_id and conn.port2 in solved:
                return solved[conn.port2]
            elif conn.port2 == port_id and conn.port1 in solved:
                return solved[conn.port1]

        return None

    def _get_prismatic_axis_for(self, port_id: str) -> SlideAxis | None:
        """Get the slide axis if this port has a prismatic constraint."""
        for pconn in self._prismatic_connections:
            if pconn.port == port_id:
                return self._slide_axes.get(pconn.axis_id)
        return None

    def _solve_circle_circle(
        self,
        c1: tuple[tuple[float, float], float],
        c2: tuple[tuple[float, float], float],
        port_id: str,
    ) -> tuple[float, float]:
        """Solve circle-circle intersection."""
        (x1, y1), r1 = c1
        (x2, y2), r2 = c2

        n, px1, py1, px2, py2 = circle_intersect(x1, y1, r1, x2, y2, r2)

        if n == INTERSECTION_NONE:
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            raise UnbuildableError(
                f"Cannot assemble joint '{port_id}': circles don't intersect. "
                f"Distance between centers: {dist:.3f}, sum of radii: {r1 + r2:.3f}"
            )

        branch = self._configuration.get(port_id, 0)
        if n == 1 or branch == 0:
            return (px1, py1)
        else:
            return (px2, py2)

    def _solve_circle_line(
        self,
        c: tuple[tuple[float, float], float],
        axis: SlideAxis,
        port_id: str,
    ) -> tuple[float, float]:
        """Solve circle-line intersection."""
        (cx, cy), r = c
        (p1x, p1y), (p2x, p2y) = axis.get_line_points()

        n, px1, py1, px2, py2 = circle_line_from_points_intersection(
            cx, cy, r, p1x, p1y, p2x, p2y
        )

        if n == INTERSECTION_NONE:
            raise UnbuildableError(
                f"Cannot assemble joint '{port_id}': circle doesn't intersect line."
            )

        branch = self._configuration.get(port_id, 0)
        if n == 1 or branch == 0:
            return (px1, py1)
        else:
            return (px2, py2)

    def _create_mechanism(
        self,
        positions: dict[str, tuple[float, float]],
    ) -> Mechanism:
        """Build Mechanism from computed positions."""
        from .joint import Joint
        from .link import Link as LinkBase

        joints: list[Joint] = []
        links: list[LinkBase] = []
        joint_map: dict[str, Joint] = {}

        # Track which ports share a joint (connected ports)
        joint_groups = self._build_joint_groups()

        # Create a joint for each group of connected ports
        for group in joint_groups:
            # Use the first port's position (all should be the same)
            representative = next(iter(group))
            pos = positions.get(representative, (0.0, 0.0))

            # Determine joint type
            is_ground = any(
                p.startswith(f"{self._ground_link_id}.")
                for p in group
                if self._ground_link_id
            )
            is_prismatic = any(
                pconn.port in group for pconn in self._prismatic_connections
            )

            # Create joint
            joint_id = "_".join(sorted(group))
            joint: Joint
            if is_ground:
                joint = GroundJoint(joint_id, position=pos)
            elif is_prismatic:
                # Find the axis for this prismatic joint
                axis_dir: tuple[float, float] = (1.0, 0.0)
                for pconn in self._prismatic_connections:
                    if pconn.port in group:
                        axis = self._slide_axes.get(pconn.axis_id)
                        if axis:
                            axis_dir = axis.get_normalized_direction()
                        break
                joint = PrismaticJoint(joint_id, position=pos, axis=axis_dir)
            else:
                joint = RevoluteJoint(joint_id, position=pos)

            joints.append(joint)
            for port_id in group:
                joint_map[port_id] = joint

        # Create ground link
        if self._ground_link_id:
            ground_joints: list[Joint] = [
                joint_map[f"{self._ground_link_id}.{p}"]
                for p in self._ground_ports
                if f"{self._ground_link_id}.{p}" in joint_map
            ]
            ground_link = GroundLink(self._ground_link_id, joints=ground_joints)
            links.append(ground_link)

        # Create other links
        for link_def in self._pending_links.values():
            link_joints: list[Joint] = []
            for port_name in link_def.ports:
                port_id = f"{link_def.id}.{port_name}"
                if port_id in joint_map:
                    link_joints.append(joint_map[port_id])

            link: LinkBase
            if link_def.is_driver:
                # Find motor joint
                motor_port_id = f"{self._ground_link_id}.{link_def.motor_port}"
                motor_joint = joint_map.get(motor_port_id)
                if motor_joint and isinstance(motor_joint, GroundJoint):
                    link = DriverLink(
                        link_def.id,
                        joints=link_joints,
                        motor_joint=motor_joint,
                        angular_velocity=link_def.angular_velocity,
                        initial_angle=link_def.initial_angle,
                    )
                else:
                    link = Link(link_def.id, joints=link_joints)
            else:
                link = Link(link_def.id, joints=link_joints)

            links.append(link)

        return Mechanism(self.name, joints=joints, links=links)

    def _build_joint_groups(self) -> list[set[str]]:
        """Build groups of ports that share a joint (are connected).

        Uses union-find to group connected ports.
        """
        all_ports = self._get_all_port_ids()
        parent: dict[str, str] = {p: p for p in all_ports}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union connected ports
        for conn in self._connections:
            if conn.port1 in all_ports and conn.port2 in all_ports:
                union(conn.port1, conn.port2)

        # Also union driver motor ports with ground ports
        for link in self._pending_links.values():
            if link.is_driver and link.motor_port and self._ground_link_id:
                driver_motor = f"{link.id}.motor"
                ground_port = f"{self._ground_link_id}.{link.motor_port}"
                if driver_motor in all_ports and ground_port in all_ports:
                    union(driver_motor, ground_port)

        # Build groups
        groups: dict[str, set[str]] = {}
        for port in all_ports:
            root = find(port)
            if root not in groups:
                groups[root] = set()
            groups[root].add(port)

        return list(groups.values())
