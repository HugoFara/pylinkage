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

import logging
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..exceptions import UnbuildableError
from ..solver.joints import solve_linear, solve_revolute
from .joint import GroundJoint, Joint, PrismaticJoint, RevoluteJoint, TrackerJoint
from .link import ArcDriverLink, DriverLink, GroundLink, Link

if TYPE_CHECKING:
    from .._types import Coord, MaybeCoord
    from ..assur.decomposition import DecompositionResult
    from ..assur.graph import LinkageGraph
    from ..dimensions import Dimensions

logger = logging.getLogger(__name__)


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
    _driver_links: list[DriverLink | ArcDriverLink] = field(default_factory=list, repr=False)
    _joint_map: dict[str, Joint] = field(default_factory=dict, repr=False)
    _link_map: dict[str, Link] = field(default_factory=dict, repr=False)

    # Assur group decomposition (built lazily for group-based solving)
    _decomposition: DecompositionResult | None = field(default=None, repr=False)
    _assur_graph: LinkageGraph | None = field(default=None, repr=False)
    _assur_dimensions: Dimensions | None = field(default=None, repr=False)
    _use_group_solver: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Build internal indices and compute solve order."""
        self._build_indices()
        self._compute_solve_order()
        self._connect_joints_to_links()
        self._cache_link_distances()
        self._build_decomposition()

    def _build_indices(self) -> None:
        """Build lookup maps for joints and links."""
        self._joint_map = {j.id: j for j in self.joints}
        self._link_map = {link.id: link for link in self.links}
        self._driver_links = [
            link for link in self.links if isinstance(link, (DriverLink, ArcDriverLink))
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

    def _cache_link_distances(self) -> None:
        """Cache distances between joints on each link.

        This stores the initial link lengths as fixed constraints
        for use during simulation. Must be called after joints have
        their initial positions set.
        """
        for link in self.links:
            link.cache_distances()

    def _build_decomposition(self) -> None:
        """Build Assur group decomposition for group-based solving.

        Converts the mechanism to a LinkageGraph, decomposes it, and
        caches the result. Falls back to joint-by-joint solving if
        decomposition fails (e.g., for trivial mechanisms).
        """
        try:
            from ..assur.mechanism_conversion import mechanism_to_graph
            from ..solver.solve import solve_group as _solve_group  # noqa: F401

            graph, dimensions = mechanism_to_graph(self)

            from ..assur.decomposition import decompose_assur_groups

            decomposition = decompose_assur_groups(graph)

            self._assur_graph = graph
            self._assur_dimensions = dimensions
            self._decomposition = decomposition
            self._use_group_solver = True
        except Exception:
            logger.debug("Assur decomposition failed, using joint-by-joint solver", exc_info=True)
            self._use_group_solver = False

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
        links_with_joint = [link for link in self.links if joint in link.joints]

        if not links_with_joint:
            return False

        # Count anchor points (joints on links that are already solved)
        anchor_count = 0
        for link in links_with_joint:
            for other in link.joints:
                if other != joint and other.id in solved:
                    anchor_count += 1

        # Prismatic joints need only 1 anchor (circle-line intersection)
        # Revolute joints need 2 anchors (circle-circle intersection)
        if isinstance(joint, PrismaticJoint):
            return anchor_count >= 1
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
        2. Solve all dependent joints (group-based or joint-by-joint)
        3. Update tracker joints
        """
        # Step drivers
        for driver in self._driver_links:
            driver.step(dt)

        if self._use_group_solver:
            self._step_groups()
        else:
            self._step_joints()

        # Update tracker joints (they depend on solved joints)
        self._update_trackers()

    @staticmethod
    def _joint_coord(joint: Joint) -> Coord:
        """Extract defined (x, y) from a joint. Caller must check is_defined()."""
        x, y = joint.position
        assert x is not None and y is not None
        return (x, y)

    def _step_groups(self) -> None:
        """Solve joints using Assur group decomposition."""
        from ..solver.solve import solve_group

        assert self._decomposition is not None
        assert self._assur_dimensions is not None

        # Build current positions from ground + driver joints
        positions: dict[str, Coord] = {}

        for node_id in self._decomposition.ground:
            joint = self._joint_map.get(node_id)
            if joint and joint.is_defined():
                positions[node_id] = self._joint_coord(joint)

        for node_id in self._decomposition.drivers:
            joint = self._joint_map.get(node_id)
            if joint and joint.is_defined():
                positions[node_id] = self._joint_coord(joint)

        # Solve each group in order
        for group in self._decomposition.groups:
            # Use current joint positions as hints for disambiguation
            hint_positions: dict[str, Coord] = {}
            for nid in group.internal_nodes:
                joint = self._joint_map.get(nid)
                if joint and joint.is_defined():
                    hint_positions[nid] = self._joint_coord(joint)

            new_positions = solve_group(group, positions, self._assur_dimensions, hint_positions)

            # Write solved positions back to joints
            for nid, (nx, ny) in new_positions.items():
                joint = self._joint_map.get(nid)
                if joint:
                    joint.set_coord(nx, ny)

            # Add to known positions for subsequent groups
            positions.update(new_positions)

    def _step_joints(self) -> None:
        """Solve joints one by one (legacy fallback for simple mechanisms)."""
        for joint in self._solve_order:
            if isinstance(joint, GroundJoint):
                continue
            if any(joint == d.output_joint for d in self._driver_links):
                continue
            self._solve_joint(joint)

    def _update_trackers(self) -> None:
        """Update tracker joints from their reference joints."""
        for joint in self.joints:
            if isinstance(joint, TrackerJoint):
                ref1 = self._joint_map.get(joint.ref_joint1_id)
                ref2 = self._joint_map.get(joint.ref_joint2_id)
                if ref1 is not None and ref2 is not None:
                    pos1 = ref1.position
                    pos2 = ref2.position
                    if (
                        pos1[0] is not None
                        and pos1[1] is not None
                        and pos2[0] is not None
                        and pos2[1] is not None
                    ):
                        joint.update_position((pos1[0], pos1[1]), (pos2[0], pos2[1]))

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

        # Prismatic joints need 1 anchor, revolute joints need 2
        if isinstance(joint, PrismaticJoint):
            if len(anchors) < 1:
                return  # Can't solve prismatic without 1 anchor
        elif len(anchors) < 2:
            return  # Can't solve revolute without 2 anchors

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
                curr_x,
                curr_y,
                a1x,
                a1y,
                dist1,
                a2x,
                a2y,
                dist2,
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

            # Construct line points from the fixed slide axis
            # The line passes through line_point in axis direction
            lpx, lpy = joint.line_point
            l1x, l1y = lpx, lpy
            l2x, l2y = lpx + dx * 10, lpy + dy * 10

            new_x, new_y = solve_linear(
                curr_x,
                curr_y,
                ax,
                ay,
                dist,
                l1x,
                l1y,
                l2x,
                l2y,
            )

            if math.isnan(new_x):
                raise UnbuildableError(joint.id)

            joint.set_coord(new_x, new_y)

    def get_rotation_period(self) -> int:
        """Get the number of steps for one full cycle.

        For continuous rotation drivers: steps for 2*pi rotation.
        For arc drivers: steps for a full back-and-forth oscillation.
        Based on the slowest driver link's angular velocity.
        """
        if not self._driver_links:
            return 360  # Default: one step per degree

        # Find minimum angular velocity
        min_omega = min(
            abs(d.angular_velocity) for d in self._driver_links if d.angular_velocity != 0
        )

        if min_omega == 0:
            return 360

        # Check if we have arc drivers - they need different period calculation
        for driver in self._driver_links:
            if isinstance(driver, ArcDriverLink):
                # For arc driver: one full oscillation (there and back)
                arc_sweep = abs(driver.arc_end - driver.arc_start)
                # Full oscillation = 2 * arc_sweep (forward + backward)
                return int(2 * arc_sweep / min_omega)

        # Steps for 2*pi rotation (standard crank)
        return int(2 * math.pi / min_omega)

    def get_constraints(self) -> list[float]:
        """Get all distance constraints as a flat list.

        Used for optimization. Returns link lengths in a consistent order.
        """
        constraints: list[float] = []

        for link in self.links:
            if isinstance(link, GroundLink):
                continue  # Ground link has fixed constraints

            if isinstance(link, (DriverLink, ArcDriverLink)):
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

            if isinstance(link, (DriverLink, ArcDriverLink)):
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
        for joint, pos in zip(self.joints, positions, strict=False):
            joint.set_coord(pos[0], pos[1])

    # ------------------------------------------------------------------
    # Velocity / acceleration kinematics
    # ------------------------------------------------------------------

    def set_input_velocity(
        self,
        driver: DriverLink | ArcDriverLink,
        omega: float,
        alpha: float = 0.0,
    ) -> None:
        """Set the angular velocity (and optional acceleration) of a driver.

        These values are used by :meth:`step_with_derivatives` to compute
        joint linear velocities and accelerations. They are independent of
        ``DriverLink.angular_velocity`` (which is in radians per simulation
        step) — ``omega`` is interpreted in physical units, typically rad/s.

        Args:
            driver: Driver link to set the input on.
            omega: Angular velocity (rad/s).
            alpha: Angular acceleration (rad/s²). Default 0.

        Raises:
            ValueError: If ``driver`` is not part of this mechanism.
        """
        if driver not in self._driver_links:
            raise ValueError(f"{driver.id!r} is not a driver link in this mechanism")
        # Stored as runtime attributes (not declared on the dataclass) so
        # they don't widen the public Driver API. ``getattr`` is used at
        # read sites to gracefully fall back to 0.
        object.__setattr__(driver, "_omega", omega)
        object.__setattr__(driver, "_alpha", alpha)

    def get_velocities(self) -> list[Coord | None]:
        """Return per-joint linear velocities, in joint order.

        Each entry is ``(vx, vy)`` or ``None`` if the joint's velocity
        has not been computed (i.e. before :meth:`step_with_derivatives`
        has been run).
        """
        return [j.velocity for j in self.joints]

    def get_accelerations(self) -> list[Coord | None]:
        """Return per-joint linear accelerations, in joint order."""
        return [j.acceleration for j in self.joints]

    def step_with_derivatives(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
    ) -> Generator[
        tuple[
            tuple[MaybeCoord, ...],
            tuple[Coord | None, ...],
            tuple[Coord | None, ...],
        ],
        None,
        None,
    ]:
        """Simulate the mechanism while computing velocities and accelerations.

        On each step yields ``(positions, velocities, accelerations)``.
        The ``omega`` (and optionally ``alpha``) of every driver link
        used as input must have been set via :meth:`set_input_velocity`;
        otherwise the driver is treated as having zero input velocity.

        Args:
            iterations: Number of steps. Defaults to :meth:`get_rotation_period`.
            dt: Time step multiplier (default 1.0).

        Yields:
            Three tuples of length ``len(self.joints)`` containing the
            joint positions, velocities, and accelerations for the step.
        """
        from ..solver.acceleration import (
            solve_crank_acceleration,
            solve_prismatic_acceleration,
            solve_revolute_acceleration,
        )
        from ..solver.velocity import (
            solve_crank_velocity,
            solve_prismatic_velocity,
            solve_revolute_velocity,
        )

        if iterations is None:
            iterations = self.get_rotation_period()

        # Map output joint id → driving DriverLink for fast lookup
        driver_for: dict[str, DriverLink | ArcDriverLink] = {}
        for d in self._driver_links:
            out = d.output_joint
            if out is not None:
                driver_for[out.id] = d

        def _anchors_with_distance(joint: Joint) -> list[tuple[Joint, float]]:
            anchors: list[tuple[Joint, float]] = []
            seen: set[str] = set()
            for link in joint._links:
                for other in link.joints:
                    if other is joint or other.id in seen:
                        continue
                    if not other.is_defined():
                        continue
                    dist = link.get_distance(joint, other)
                    if dist is None:
                        continue
                    anchors.append((other, dist))
                    seen.add(other.id)
            return anchors

        for _ in range(iterations):
            # 1. Positions
            self._step_once(dt)

            # 2. Velocities (in solve order so anchors are populated first)
            for joint in self._solve_order:
                if isinstance(joint, GroundJoint):
                    joint.velocity = (0.0, 0.0)
                    continue

                jx, jy = joint.position
                if jx is None or jy is None:
                    joint.velocity = None
                    continue

                # Driver output joint — use crank velocity formula
                driver = driver_for.get(joint.id)
                if driver is not None and driver.motor_joint is not None:
                    mj = driver.motor_joint
                    if mj.x is None or mj.y is None or driver.radius is None:
                        joint.velocity = None
                        continue
                    omega = float(getattr(driver, "_omega", 0.0))
                    mv = mj.velocity or (0.0, 0.0)
                    vx, vy = solve_crank_velocity(
                        jx,
                        jy,
                        mj.x,
                        mj.y,
                        mv[0],
                        mv[1],
                        driver.radius,
                        omega,
                    )
                    joint.velocity = None if math.isnan(vx) or math.isnan(vy) else (vx, vy)
                    continue

                anchors = _anchors_with_distance(joint)

                if isinstance(joint, PrismaticJoint):
                    # Prismatic: 1 revolute anchor + sliding line. Until the
                    # mechanism model exposes a line constraint API for
                    # prismatic joints we fall back to None.
                    if len(anchors) >= 1 and anchors[0][0].velocity is not None:
                        a, dist = anchors[0]
                        ax_, ay_ = a.position
                        av = a.velocity or (0.0, 0.0)
                        # No second line anchor available → leave undefined.
                        if ax_ is None or ay_ is None:
                            joint.velocity = None
                            continue
                        vx, vy = solve_prismatic_velocity(
                            jx,
                            jy,
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                            dist,
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                        )
                        joint.velocity = None if math.isnan(vx) or math.isnan(vy) else (vx, vy)
                    else:
                        joint.velocity = None
                    continue

                # Generic revolute joint: needs two solved anchors
                if len(anchors) < 2:
                    joint.velocity = None
                    continue
                a1, _d1 = anchors[0]
                a2, _d2 = anchors[1]
                if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                    joint.velocity = None
                    continue
                v1 = a1.velocity or (0.0, 0.0)
                v2 = a2.velocity or (0.0, 0.0)
                vx, vy = solve_revolute_velocity(
                    jx,
                    jy,
                    a1.x,
                    a1.y,
                    v1[0],
                    v1[1],
                    a2.x,
                    a2.y,
                    v2[0],
                    v2[1],
                )
                joint.velocity = None if math.isnan(vx) or math.isnan(vy) else (vx, vy)

            # 3. Accelerations
            for joint in self._solve_order:
                if isinstance(joint, GroundJoint):
                    joint.acceleration = (0.0, 0.0)
                    continue

                jx, jy = joint.position
                if jx is None or jy is None or joint.velocity is None:
                    joint.acceleration = None
                    continue
                jvx, jvy = joint.velocity

                driver = driver_for.get(joint.id)
                if driver is not None and driver.motor_joint is not None:
                    mj = driver.motor_joint
                    if mj.x is None or mj.y is None or driver.radius is None:
                        joint.acceleration = None
                        continue
                    omega = float(getattr(driver, "_omega", 0.0))
                    alpha = float(getattr(driver, "_alpha", 0.0))
                    mv = mj.velocity or (0.0, 0.0)
                    ma = mj.acceleration or (0.0, 0.0)
                    ax, ay = solve_crank_acceleration(
                        jx,
                        jy,
                        jvx,
                        jvy,
                        mj.x,
                        mj.y,
                        mv[0],
                        mv[1],
                        ma[0],
                        ma[1],
                        driver.radius,
                        omega,
                        alpha,
                    )
                    joint.acceleration = None if math.isnan(ax) or math.isnan(ay) else (ax, ay)
                    continue

                anchors = _anchors_with_distance(joint)
                if isinstance(joint, PrismaticJoint):
                    if len(anchors) >= 1 and anchors[0][0].velocity is not None:
                        a, dist = anchors[0]
                        ax_, ay_ = a.position
                        av = a.velocity or (0.0, 0.0)
                        aa = a.acceleration or (0.0, 0.0)
                        if ax_ is None or ay_ is None:
                            joint.acceleration = None
                            continue
                        ax, ay = solve_prismatic_acceleration(
                            jx,
                            jy,
                            jvx,
                            jvy,
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                            aa[0],
                            aa[1],
                            dist,
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                            aa[0],
                            aa[1],
                            ax_,
                            ay_,
                            av[0],
                            av[1],
                            aa[0],
                            aa[1],
                        )
                        joint.acceleration = None if math.isnan(ax) or math.isnan(ay) else (ax, ay)
                    else:
                        joint.acceleration = None
                    continue

                if len(anchors) < 2:
                    joint.acceleration = None
                    continue
                a1, _ = anchors[0]
                a2, _ = anchors[1]
                if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                    joint.acceleration = None
                    continue
                v1 = a1.velocity or (0.0, 0.0)
                v2 = a2.velocity or (0.0, 0.0)
                acc1 = a1.acceleration or (0.0, 0.0)
                acc2 = a2.acceleration or (0.0, 0.0)
                ax, ay = solve_revolute_acceleration(
                    jx,
                    jy,
                    jvx,
                    jvy,
                    a1.x,
                    a1.y,
                    v1[0],
                    v1[1],
                    acc1[0],
                    acc1[1],
                    a2.x,
                    a2.y,
                    v2[0],
                    v2[1],
                    acc2[0],
                    acc2[1],
                )
                joint.acceleration = None if math.isnan(ax) or math.isnan(ay) else (ax, ay)

            yield (
                tuple(j.coord() for j in self.joints),
                tuple(j.velocity for j in self.joints),
                tuple(j.acceleration for j in self.joints),
            )

    def reset(self) -> None:
        """Reset all driver links to initial state."""
        for driver in self._driver_links:
            driver.reset()
