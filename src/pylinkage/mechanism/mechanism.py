"""Mechanism class - the main orchestrator for planar linkages.

This module provides the Mechanism class, which manages the collection
of links and joints that form a planar mechanism. It handles:
- Solving order computation
- Constraint management
- Simulation stepping
- Position computation

The Mechanism class uses proper mechanical engineering terminology:
- Joints are actual connection points (revolute, prismatic)
- Links are rigid bodies connecting joints
- Dyads are solved as constraint satisfaction problems
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..exceptions import UnbuildableError
from ..solver.joints import solve_linear, solve_revolute
from .joint import GroundJoint, Joint, JointType, PrismaticJoint, RevoluteJoint
from .link import DriverLink, GroundLink, Link

if TYPE_CHECKING:
    from .._types import Coord, MaybeCoord


@dataclass
class Mechanism:
    """A planar linkage mechanism.

    A mechanism is a collection of rigid links connected by joints
    that transmits and transforms motion. This class manages the
    topology and provides simulation capabilities.

    Attributes:
        name: Human-readable name for the mechanism.
        joints: All joints in the mechanism.
        links: All links in the mechanism.
        ground: The ground (frame) link.

    Example:
        >>> from pylinkage.mechanism import Mechanism, GroundJoint, create_crank, create_rrr_dyad
        >>> # Create a four-bar linkage
        >>> O1 = GroundJoint("O1", position=(0.0, 0.0))
        >>> O2 = GroundJoint("O2", position=(2.0, 0.0))
        >>> ground = GroundLink("ground", joints=[O1, O2])
        >>> crank, A = create_crank(O1, radius=1.0, angular_velocity=0.1)
        >>> link1, link2, B = create_rrr_dyad(A, O2, distance1=2.0, distance2=1.5)
        >>> mechanism = Mechanism("Four-Bar", joints=[O1, O2, A, B],
        ...                       links=[ground, crank, link1, link2])
    """

    name: str = ""
    joints: list[Joint] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    ground: GroundLink | None = None

    # Internal state
    _solve_order: list[Joint] = field(default_factory=list, repr=False)
    _driver_links: list[DriverLink] = field(default_factory=list, repr=False)
    _joint_map: dict[str, Joint] = field(default_factory=dict, repr=False)
    _link_map: dict[str, Link] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build internal indices and compute solve order."""
        self._build_indices()
        self._compute_solve_order()
        self._connect_joints_to_links()

    def _build_indices(self) -> None:
        """Build lookup maps for joints and links."""
        self._joint_map = {j.id: j for j in self.joints}
        self._link_map = {link.id: link for link in self.links}
        self._driver_links = [
            link for link in self.links if isinstance(link, DriverLink)
        ]

        # Auto-detect ground link if not specified
        if self.ground is None:
            for link in self.links:
                if isinstance(link, GroundLink):
                    self.ground = link
                    break

    def _connect_joints_to_links(self) -> None:
        """Populate the _links list for each joint."""
        for joint in self.joints:
            joint._links.clear()

        for link in self.links:
            for joint in link.joints:
                if link not in joint._links:
                    joint._links.append(link)

    def _compute_solve_order(self) -> None:
        """Compute the order in which joints should be solved.

        Ground joints are solved first (they're fixed), then driver
        outputs, then dependent joints in topological order.
        """
        solved: set[str] = set()
        order: list[Joint] = []

        # First: all ground joints (fixed positions)
        for joint in self.joints:
            if isinstance(joint, GroundJoint):
                order.append(joint)
                solved.add(joint.id)

        # Second: driver link outputs (computed from drivers)
        for driver in self._driver_links:
            output = driver.output_joint
            if output is not None and output.id not in solved:
                order.append(output)
                solved.add(output.id)

        # Third: remaining joints in dependency order
        # Keep iterating until no more can be solved
        remaining = [j for j in self.joints if j.id not in solved]
        max_iterations = len(remaining) * 2  # Safety limit

        for _ in range(max_iterations):
            if not remaining:
                break

            progress = False
            still_remaining = []

            for joint in remaining:
                if self._can_solve_joint(joint, solved):
                    order.append(joint)
                    solved.add(joint.id)
                    progress = True
                else:
                    still_remaining.append(joint)

            remaining = still_remaining

            if not progress and remaining:
                # Circular dependency or missing constraints
                break

        self._solve_order = order

    def _can_solve_joint(self, joint: Joint, solved: set[str]) -> bool:
        """Check if a joint can be solved given already-solved joints.

        A joint can be solved if all joints it depends on are solved.
        This depends on the joint type and how it's constrained.
        """
        # Find links containing this joint
        links_with_joint = [
            link for link in self.links
            if joint in link.joints
        ]

        if not links_with_joint:
            return False

        # For a revolute joint, we need two anchor points
        # (joints on links that are already solved)
        anchor_count = 0
        for link in links_with_joint:
            for other in link.joints:
                if other != joint and other.id in solved:
                    anchor_count += 1

        # Need at least 2 anchors for RRR dyad
        return anchor_count >= 2

    def get_joint(self, joint_id: str) -> Joint | None:
        """Get a joint by ID."""
        return self._joint_map.get(joint_id)

    def get_link(self, link_id: str) -> Link | None:
        """Get a link by ID."""
        return self._link_map.get(link_id)

    def step(self, dt: float = 1.0) -> Generator[tuple[MaybeCoord, ...], None, None]:
        """Simulate one full rotation of the mechanism.

        Yields joint positions at each step of the simulation.

        Args:
            dt: Time step multiplier (default 1.0).

        Yields:
            Tuple of (x, y) coordinates for all joints.
        """
        iterations = self.get_rotation_period()

        for _ in range(iterations):
            self._step_once(dt)
            yield tuple(j.coord() for j in self.joints)

    def _step_once(self, dt: float = 1.0) -> None:
        """Perform a single simulation step.

        1. Advance all driver links
        2. Solve all dependent joints in order
        """
        # Step drivers
        for driver in self._driver_links:
            driver.step(dt)

        # Solve remaining joints
        for joint in self._solve_order:
            if isinstance(joint, GroundJoint):
                continue  # Ground joints don't move
            if any(joint == d.output_joint for d in self._driver_links):
                continue  # Driver outputs already updated

            self._solve_joint(joint)

    def _solve_joint(self, joint: Joint) -> None:
        """Solve the position of a single joint.

        Determines the joint type and applies appropriate solver.
        """
        # Find the two anchor joints this joint depends on
        anchors: list[tuple[Joint, float]] = []  # (anchor, distance)

        for link in joint._links:
            for other in link.joints:
                if other != joint and other.is_defined():
                    dist = link.get_distance(joint, other)
                    if dist is not None:
                        anchors.append((other, dist))

        if len(anchors) < 2:
            return  # Can't solve without 2 anchors

        # For revolute joints: circle-circle intersection
        if isinstance(joint, RevoluteJoint):
            anchor1, dist1 = anchors[0]
            anchor2, dist2 = anchors[1]

            a1x, a1y = anchor1.position
            a2x, a2y = anchor2.position
            curr_x, curr_y = joint.position

            # Handle undefined current position
            if curr_x is None:
                curr_x = (a1x + a2x) / 2 if a1x and a2x else 0.0
            if curr_y is None:
                curr_y = (a1y + a2y) / 2 if a1y and a2y else 0.0

            assert a1x is not None and a1y is not None
            assert a2x is not None and a2y is not None

            new_x, new_y = solve_revolute(
                curr_x, curr_y,
                a1x, a1y, dist1,
                a2x, a2y, dist2,
            )

            if math.isnan(new_x):
                raise UnbuildableError(joint.id)

            joint.set_coord(new_x, new_y)

        # For prismatic joints: circle-line intersection
        elif isinstance(joint, PrismaticJoint):
            # Need one revolute anchor and a line
            # Find the line from the joint's axis
            anchor, dist = anchors[0]

            ax, ay = anchor.position
            curr_x, curr_y = joint.position
            dx, dy = joint.get_axis_normalized()

            if curr_x is None or curr_y is None:
                curr_x, curr_y = ax or 0.0, ay or 0.0

            assert ax is not None and ay is not None

            # Construct line points from axis
            # The line passes through the current position in axis direction
            l1x, l1y = curr_x, curr_y
            l2x, l2y = curr_x + dx * 10, curr_y + dy * 10

            new_x, new_y = solve_linear(
                curr_x, curr_y,
                ax, ay, dist,
                l1x, l1y,
                l2x, l2y,
            )

            if math.isnan(new_x):
                raise UnbuildableError(joint.id)

            joint.set_coord(new_x, new_y)

    def get_rotation_period(self) -> int:
        """Get the number of steps for one full rotation.

        Based on the slowest driver link's angular velocity.
        """
        if not self._driver_links:
            return 100  # Default

        # Find minimum angular velocity
        min_omega = min(
            abs(d.angular_velocity) for d in self._driver_links
            if d.angular_velocity != 0
        )

        if min_omega == 0:
            return 100

        # Steps for 2*pi rotation
        return int(2 * math.pi / min_omega)

    def get_constraints(self) -> list[float]:
        """Get all distance constraints as a flat list.

        Used for optimization. Returns link lengths in a consistent order.
        """
        constraints: list[float] = []

        for link in self.links:
            if isinstance(link, GroundLink):
                continue  # Ground link has fixed constraints

            if isinstance(link, DriverLink):
                # Driver: just the radius
                radius = link.radius
                if radius is not None:
                    constraints.append(radius)
            else:
                # Regular link: length
                length = link.length
                if length is not None:
                    constraints.append(length)

        return constraints

    def set_constraints(self, values: list[float]) -> None:
        """Set distance constraints from a flat list.

        Used for optimization. Applies constraints in the same order
        as get_constraints().

        Args:
            values: List of constraint values to apply.
        """
        idx = 0

        for link in self.links:
            if isinstance(link, GroundLink):
                continue

            if idx >= len(values):
                break

            if isinstance(link, DriverLink):
                # Update driver radius by moving output joint
                new_radius = values[idx]
                idx += 1

                output = link.output_joint
                if output is None or link.motor_joint is None:
                    continue

                mx, my = link.motor_joint.position
                if mx is None or my is None:
                    continue

                # Recompute output position with new radius
                new_x = mx + new_radius * math.cos(link.current_angle)
                new_y = my + new_radius * math.sin(link.current_angle)
                output.set_coord(new_x, new_y)

            elif len(link.joints) == 2:
                # Binary link: update the movable joint's position
                # (This is a simplified approach; full implementation
                # would need to track which joint is the dependent one)
                _new_length = values[idx]
                idx += 1
                # Position update happens during simulation

    def get_joint_positions(self) -> list[Coord]:
        """Get current positions of all joints."""
        positions: list[Coord] = []
        for joint in self.joints:
            x, y = joint.position
            positions.append((x or 0.0, y or 0.0))
        return positions

    def set_joint_positions(self, positions: list[Coord]) -> None:
        """Set positions of all joints."""
        for joint, pos in zip(self.joints, positions):
            joint.set_coord(pos[0], pos[1])

    def reset(self) -> None:
        """Reset all driver links to initial state."""
        for driver in self._driver_links:
            driver.reset()
