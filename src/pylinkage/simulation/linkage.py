"""Linkage container for simulating planar mechanisms.

The Linkage class orchestrates a collection of components (Ground, actuators,
dyads) to simulate a planar mechanism.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from math import gcd, tau
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..components import Component, ConnectedComponent, _AnchorProxy
from ..exceptions import UnderconstrainedError

if TYPE_CHECKING:
    from .._simulation_context import Simulation as _SimulationContext
    from ..actuators import ArcCrank, Crank, LinearActuator
    from ..linkage.sensitivity import SensitivityAnalysis, ToleranceAnalysis
    from ..linkage.transmission import StrokeAnalysis, TransmissionAngleAnalysis
    from ..solver import SolverData


class Linkage:
    """A planar linkage mechanism built from components.

    The Linkage class orchestrates a collection of components (Ground points,
    actuators, and dyads) to simulate a planar mechanism. It handles solve
    order computation, stepping, and constraint management.

    Example:
        >>> from pylinkage.components import Ground
        >>> from pylinkage.actuators import Crank
        >>> from pylinkage.dyads import RRRDyad
        >>> from pylinkage.simulation import Linkage
        >>>
        >>> O1 = Ground(0.0, 0.0, name="O1")
        >>> O2 = Ground(2.0, 0.0, name="O2")
        >>> crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1)
        >>> rocker = RRRDyad(crank.output, O2, distance1=2.0, distance2=1.5)
        >>> linkage = Linkage([O1, O2, crank, rocker], name="Four-Bar")
        >>> for positions in linkage.step():
        ...     print(positions)
    """

    __slots__ = (
        "name",
        "components",
        "_cranks",
        "_arc_cranks",
        "_linear_actuators",
        "_solve_order",
        "_solver_data",
    )

    name: str
    components: tuple[Component, ...]
    _cranks: tuple[Crank, ...]
    _arc_cranks: tuple[ArcCrank, ...]
    _linear_actuators: tuple[LinearActuator, ...]
    _solve_order: tuple[Component, ...]
    _solver_data: SolverData | None

    def __init__(
        self,
        components: Iterable[Component],
        order: Iterable[Component] | None = None,
        name: str | None = None,
    ) -> None:
        """Create a linkage from components.

        Args:
            components: Collection of components forming the linkage.
            order: Manual solve order. If None, computed automatically.
            name: Human-readable identifier.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator

        self.name = name if name is not None else str(id(self))
        self.components = tuple(components)
        self._cranks = tuple(d for d in self.components if isinstance(d, Crank))
        self._arc_cranks = tuple(d for d in self.components if isinstance(d, ArcCrank))
        self._linear_actuators = tuple(d for d in self.components if isinstance(d, LinearActuator))

        if order is not None:
            self._solve_order = tuple(order)
        self._solver_data = None

    @property
    def dyads(self) -> tuple[Component, ...]:
        """Return components (backwards compatibility alias)."""
        return self.components

    def _find_solve_order(self) -> tuple[Component, ...]:
        """Compute automatic solve order.

        Ground points are solved first (trivial), then actuators
        (cranks and linear actuators), then dependent dyads whose
        anchors are already solved.

        Returns:
            Tuple of components in solvable order.

        Raises:
            UnderconstrainedError: If order cannot be determined.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator
        from ..components import Ground

        # Start with ground points (always solvable)
        solved: list[Component] = [d for d in self.components if isinstance(d, Ground)]
        solved_set: set[Component] = set(solved)

        # Track progress
        progress = True
        while len(solved) < len(self.components) and progress:
            progress = False
            for d in self.components:
                if d in solved_set:
                    continue

                # Check if this component can be solved
                can_solve = False

                if isinstance(d, (Crank, ArcCrank, LinearActuator)):
                    # Actuators can solve if their anchor is solved
                    can_solve = d.anchor in solved_set
                elif isinstance(d, ConnectedComponent):
                    # Connected components can solve if all anchors are solved
                    anchors = d.anchors
                    can_solve = all(
                        a in solved_set or self._is_anchor_solved(a, solved_set) for a in anchors
                    )

                if can_solve:
                    solved.append(d)
                    solved_set.add(d)
                    progress = True

        if len(solved) < len(self.components):
            unsolved = [d for d in self.components if d not in solved_set]
            raise UnderconstrainedError(
                f"Cannot determine solve order. Unsolved components: "
                f"{', '.join(d.name for d in unsolved)}"
            )

        self._solve_order = tuple(solved)
        return self._solve_order

    def _is_anchor_solved(self, anchor: Component, solved_set: set[Component]) -> bool:
        """Check if an anchor (possibly a proxy) is solved.

        Args:
            anchor: The anchor to check.
            solved_set: Set of solved components.

        Returns:
            True if the anchor's parent component is solved.
        """
        if isinstance(anchor, _AnchorProxy):
            return anchor._parent in solved_set
        return anchor in solved_set

    def rebuild(self, positions: list[tuple[float, float]] | None = None) -> None:
        """Rebuild the linkage, optionally setting initial positions.

        Args:
            positions: Initial positions for each component. If None, uses
                current positions.
        """
        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if positions is not None:
            for component, pos in zip(self.components, positions, strict=True):
                component.set_coord(pos[0], pos[1])

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> Generator[tuple[tuple[float | None, float | None], ...], None, None]:
        """Simulate the linkage.

        Yields positions for all components at each step.

        Args:
            iterations: Number of steps. If None, uses get_rotation_period().
            dt: Time step multiplier for actuators (cranks and linear actuators).

        Yields:
            Tuple of (x, y) positions for each component.
        """
        from ..actuators import ArcCrank, Crank, LinearActuator

        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if iterations is None:
            iterations = self.get_rotation_period()

        for _ in range(iterations):
            for component in self._solve_order:
                if isinstance(component, (Crank, ArcCrank, LinearActuator)):
                    component.reload(dt)
                else:
                    component.reload()
            yield tuple(d.position for d in self.components)

    def get_rotation_period(self) -> int:
        """Return number of steps for one full cycle.

        Computes the LCM of all actuator periods (cranks, arc cranks, and
        linear actuators).
        For cranks, period is 2*pi / angular_velocity.
        For arc cranks, period is 2 * (arc_end - arc_start) / angular_velocity.
        For linear actuators, period is 2 * stroke / velocity.

        Returns:
            Number of iterations with dt=1.
        """
        periods = 1

        # Consider crank periods
        for crank in self._cranks:
            if crank.angular_velocity != 0:
                freq = round(tau / abs(crank.angular_velocity))
                periods = periods * freq // gcd(periods, freq)

        # Consider arc crank periods
        for arc_crank in self._arc_cranks:
            if arc_crank.angular_velocity != 0:
                # Full cycle is 2 * arc_range / angular_velocity (one round trip)
                arc_range = arc_crank.arc_end - arc_crank.arc_start
                freq = round(2 * arc_range / abs(arc_crank.angular_velocity))
                periods = periods * freq // gcd(periods, freq)

        # Consider linear actuator periods
        for actuator in self._linear_actuators:
            if actuator.speed != 0:
                # Full cycle is 2 * stroke / speed (one round trip)
                freq = round(2 * actuator.stroke / abs(actuator.speed))
                periods = periods * freq // gcd(periods, freq)

        return periods

    def get_coords(self) -> list[tuple[float | None, float | None]]:
        """Return positions of all components.

        Returns:
            List of (x, y) positions.
        """
        return [d.position for d in self.components]

    def set_coords(self, coords: list[tuple[float, float]]) -> None:
        """Set positions for all components.

        Args:
            coords: List of (x, y) positions.
        """
        for component, coord in zip(self.components, coords, strict=True):
            component.set_coord(coord[0], coord[1])

    def get_num_constraints(self) -> list[float]:
        """Return all constraints as a flat list.

        Used for optimization.

        Returns:
            Flat list of all constraint values.
        """
        constraints: list[float] = []
        for component in self.components:
            for c in component.get_constraints():
                if c is not None:
                    constraints.append(c)
        return constraints

    def set_num_constraints(self, values: list[float]) -> None:
        """Set constraints from a flat list.

        Used to apply optimization results. Invalidates any cached
        SolverData so the next :meth:`step_fast` recompiles.

        Args:
            values: Flat list of constraint values.
        """
        self._solver_data = None
        idx = 0
        for component in self.components:
            n_constraints = len([c for c in component.get_constraints() if c is not None])
            if n_constraints > 0:
                component.set_constraints(*values[idx : idx + n_constraints])
                idx += n_constraints

    def set_completely(
        self,
        constraints: list[float],
        positions: list[tuple[float, float]],
    ) -> None:
        """Apply both constraints and initial positions in one call.

        Args:
            constraints: Flat list (as accepted by :meth:`set_num_constraints`).
            positions: Per-component ``(x, y)`` positions
                (as accepted by :meth:`set_coords`).
        """
        self.set_num_constraints(constraints)
        self.set_coords(positions)

    def simulation(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
    ) -> _SimulationContext:
        """Return a context manager that simulates this linkage.

        The context restores the initial joint positions on exit, so
        repeated invocations return to the same starting state.
        """
        from .._simulation_context import Simulation as _SimulationContext

        return _SimulationContext(self, iterations=iterations, dt=dt)

    def indeterminacy(self) -> int:
        """Mobility (DOF) of the linkage — planar Gruebler-Kutzbach.

        ``DOF = 3·(n − 1) − 2·R − P`` where each non-ground component
        contributes its share of bodies and kinematic pairs:

        - ``Ground`` anchors are points on the frame (no new body, no
          new pair on their own);
        - ``Crank`` / ``LinearActuator`` add 1 body + 1 R/P pair;
        - binary dyads (``RRRDyad``, ``FixedDyad``) add 2 bodies + 3
          R-pairs;
        - ``RRPDyad`` adds 2 bodies + 2 R-pairs + 1 P-pair.

        A standard Grashof four-bar (Crank + RRRDyad) returns ``1``.
        """
        from ..actuators import ArcCrank, LinearActuator
        from ..actuators import Crank as _Crank
        from ..dyads import FixedDyad, RRPDyad, RRRDyad

        bodies = 1  # ground frame
        revolute_pairs = 0
        prismatic_pairs = 0
        for c in self.components:
            if isinstance(c, (_Crank, ArcCrank)):
                bodies += 1
                revolute_pairs += 1
            elif isinstance(c, LinearActuator):
                bodies += 1
                prismatic_pairs += 1
            elif isinstance(c, (RRRDyad, FixedDyad)):
                bodies += 2
                revolute_pairs += 3
            elif isinstance(c, RRPDyad):
                bodies += 2
                revolute_pairs += 2
                prismatic_pairs += 1
        return 3 * (bodies - 1) - 2 * revolute_pairs - prismatic_pairs

    # ------------------------------------------------------------------
    # Analysis bound methods — thin shims over pylinkage.linkage.*
    # ------------------------------------------------------------------

    def analyze_transmission(
        self,
        iterations: int | None = None,
        acceptable_range: tuple[float, float] = (40.0, 140.0),
    ) -> TransmissionAngleAnalysis:
        """Analyze transmission angle over a full motion cycle.

        See :func:`pylinkage.linkage.analyze_transmission` for details.
        """
        from ..linkage.transmission import analyze_transmission

        return analyze_transmission(
            self,
            iterations=iterations,
            acceptable_range=acceptable_range,
        )

    def analyze_stroke(
        self,
        prismatic_joint: object | None = None,
        iterations: int | None = None,
    ) -> StrokeAnalysis:
        """Analyze stroke/slide position over a full motion cycle.

        See :func:`pylinkage.linkage.analyze_stroke`.
        """
        from ..linkage.transmission import analyze_stroke

        return analyze_stroke(
            self,
            prismatic_joint=prismatic_joint,
            iterations=iterations,
        )

    def analyze_sensitivity(
        self,
        output_joint: object | int | None = None,
        delta: float = 0.01,
        include_transmission: bool = True,
        iterations: int | None = None,
    ) -> SensitivityAnalysis:
        """Compute sensitivity of an output path to constraint perturbations.

        See :func:`pylinkage.linkage.analyze_sensitivity`.
        """
        from ..linkage.sensitivity import analyze_sensitivity

        return analyze_sensitivity(
            self,
            output_joint=output_joint,
            delta=delta,
            include_transmission=include_transmission,
            iterations=iterations,
        )

    def analyze_tolerance(
        self,
        tolerances: dict[str, float],
        output_joint: object | int | None = None,
        iterations: int | None = None,
        n_samples: int = 1000,
    ) -> ToleranceAnalysis:
        """Monte-Carlo tolerance analysis over the output path.

        See :func:`pylinkage.linkage.analyze_tolerance`.
        """
        from ..linkage.sensitivity import analyze_tolerance

        return analyze_tolerance(
            self,
            tolerances=tolerances,
            output_joint=output_joint,
            iterations=iterations,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Numba fast path
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Pre-compile the numba solver state for :meth:`step_fast`.

        Cached on ``self._solver_data`` and reused until invalidated by
        a call to ``compile()`` again.
        """
        from ..bridge.solver_conversion import linkage_to_solver_data

        if not hasattr(self, "_solve_order"):
            self._find_solve_order()
        self._solver_data = linkage_to_solver_data(self)

    def step_fast_with_kinematics(
        self,
        iterations: int | None = None,
        dt: float = 1.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Run the numba-compiled simulation, returning velocities and accelerations.

        Per-crank ``omega``/``alpha`` inputs must be set via
        :meth:`set_input_velocity` (cranks without an explicit input default
        to zero).

        Args:
            iterations: Number of steps. Defaults to :meth:`get_rotation_period`.
            dt: Time step multiplier (default 1.0).

        Returns:
            ``(positions, velocities, accelerations)`` — each a numpy array
            of shape ``(iterations, n_components, 2)``.
        """
        from ..bridge.solver_conversion import (
            solver_data_to_linkage,
            update_solver_positions,
        )
        from ..solver.simulation import simulate_with_kinematics

        if self._solver_data is None:
            self.compile()
        assert self._solver_data is not None

        if iterations is None:
            iterations = self.get_rotation_period()

        update_solver_positions(self._solver_data, self)

        n_components = len(self.components)
        n_cranks = len(self._cranks)

        if self._solver_data.velocities is None:
            self._solver_data.velocities = np.zeros((n_components, 2), dtype=np.float64)
        if self._solver_data.accelerations is None:
            self._solver_data.accelerations = np.zeros(
                (n_components, 2),
                dtype=np.float64,
            )
        if self._solver_data.omega_values is None:
            self._solver_data.omega_values = np.zeros(n_cranks, dtype=np.float64)
        if self._solver_data.alpha_values is None:
            self._solver_data.alpha_values = np.zeros(n_cranks, dtype=np.float64)
        if self._solver_data.crank_indices is None:
            crank_indices = [i for i, c in enumerate(self.components) if c in self._cranks]
            self._solver_data.crank_indices = np.array(crank_indices, dtype=np.int32)

        # Sync each crank's stored _omega / _alpha into the solver arrays.
        for i, crank in enumerate(self._cranks):
            self._solver_data.omega_values[i] = float(getattr(crank, "_omega", 0.0))
            self._solver_data.alpha_values[i] = float(getattr(crank, "_alpha", 0.0))

        pos, vel, acc = simulate_with_kinematics(
            self._solver_data.positions,
            self._solver_data.velocities,
            self._solver_data.accelerations,
            self._solver_data.constraints,
            self._solver_data.joint_types,
            self._solver_data.parent_indices,
            self._solver_data.constraint_offsets,
            self._solver_data.solve_order,
            self._solver_data.omega_values,
            self._solver_data.alpha_values,
            self._solver_data.crank_indices,
            iterations,
            dt,
        )

        solver_data_to_linkage(self._solver_data, self)
        return pos, vel, acc

    def step_fast(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> NDArray[np.float64]:
        """Run the simulation through the numba-compiled solver.

        Significantly faster than :meth:`step` for large iteration counts
        because it avoids per-step Python dispatch.

        Args:
            iterations: Number of steps. Defaults to :meth:`get_rotation_period`.
            dt: Time step multiplier (default 1.0).

        Returns:
            Trajectory array of shape ``(iterations, n_components, 2)``.
            Unbuildable configurations appear as NaN — check with
            ``np.isnan(trajectory).any()``.
        """
        from ..bridge.solver_conversion import (
            solver_data_to_linkage,
            update_solver_positions,
        )
        from ..solver.simulation import simulate

        if self._solver_data is None:
            self.compile()
        assert self._solver_data is not None

        if iterations is None:
            iterations = self.get_rotation_period()

        update_solver_positions(self._solver_data, self)

        trajectory: NDArray[np.float64] = simulate(
            self._solver_data.positions,
            self._solver_data.constraints,
            self._solver_data.joint_types,
            self._solver_data.parent_indices,
            self._solver_data.constraint_offsets,
            self._solver_data.solve_order,
            iterations,
            dt,
        )

        solver_data_to_linkage(self._solver_data, self)
        return trajectory

    def set_input_velocity(
        self,
        actuator: Crank,
        omega: float,
        alpha: float = 0.0,
    ) -> None:
        """Set angular velocity and acceleration for a crank actuator.

        This is used for kinematics computation (velocity/acceleration analysis).
        The omega value will be used to compute linear velocities at each joint.

        Args:
            actuator: The crank actuator to set velocity for.
            omega: Angular velocity in rad/s (physical units for analysis).
            alpha: Angular acceleration in rad/s² (default 0).

        Raises:
            ValueError: If the actuator is not part of this linkage.

        Example:
            >>> linkage.set_input_velocity(crank, omega=10.0)  # 10 rad/s
            >>> for pos, vel, acc in linkage.step_with_derivatives():
            ...     print(f"Position: {pos}, Velocity: {vel}")
        """

        if actuator not in self._cranks:
            raise ValueError(f"{actuator} is not a crank in this linkage")
        # Store omega/alpha as attributes for kinematics computation
        # These are separate from angular_velocity which is radians/step
        actuator._omega = omega  # type: ignore[attr-defined]
        actuator._alpha = alpha  # type: ignore[attr-defined]

    def get_velocities(self) -> list[tuple[float, float] | None]:
        """Return velocities for all components.

        Returns:
            List of (vx, vy) tuples, one per component. Returns None for
            components whose velocity has not been computed.
        """
        return [c.velocity for c in self.components]

    def get_accelerations(self) -> list[tuple[float, float] | None]:
        """Return accelerations for all components.

        Returns:
            List of (ax, ay) tuples, one per component. Returns None for
            components whose acceleration has not been computed.
        """
        return [c.acceleration for c in self.components]

    def step_with_derivatives(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> Generator[
        tuple[
            tuple[tuple[float | None, float | None], ...],
            tuple[tuple[float, float] | None, ...],
            tuple[tuple[float, float] | None, ...],
        ],
        None,
        None,
    ]:
        """Simulate the linkage with velocity and acceleration computation.

        Yields positions, velocities, and accelerations for all components
        at each step. Requires that omega (and optionally alpha) is set on
        crank actuators via set_input_velocity().

        Args:
            iterations: Number of steps. If None, uses get_rotation_period().
            dt: Time step multiplier for actuators (cranks and linear actuators).

        Yields:
            Tuple of (positions, velocities, accelerations) where:
            - positions: Tuple of (x, y) for each component
            - velocities: Tuple of (vx, vy) or None for each component
            - accelerations: Tuple of (ax, ay) or None for each component

        Example:
            >>> linkage.set_input_velocity(crank, omega=10.0)
            >>> for pos, vel, acc in linkage.step_with_derivatives():
            ...     print(f"Crank velocity: {vel[2]}")
        """
        import math

        from ..actuators import ArcCrank, Crank, LinearActuator
        from ..components import Ground
        from ..dyads import FixedDyad, RRPDyad, RRRDyad
        from ..solver.acceleration import (
            solve_crank_acceleration,
            solve_fixed_acceleration,
            solve_prismatic_acceleration,
            solve_revolute_acceleration,
        )
        from ..solver.velocity import (
            solve_crank_velocity,
            solve_fixed_velocity,
            solve_prismatic_velocity,
            solve_revolute_velocity,
        )

        if not hasattr(self, "_solve_order"):
            self._find_solve_order()

        if iterations is None:
            iterations = self.get_rotation_period()

        for _ in range(iterations):
            # Step 1: Compute positions
            for component in self._solve_order:
                if isinstance(component, (Crank, ArcCrank, LinearActuator)):
                    component.reload(dt)
                else:
                    component.reload()

            # Step 2: Compute velocities
            for component in self._solve_order:
                if isinstance(component, Ground):
                    # Ground points have zero velocity
                    component.velocity = (0.0, 0.0)

                elif isinstance(component, Crank):
                    # Get omega from set_input_velocity or default to 0
                    omega = getattr(component, "_omega", 0.0)
                    if component.x is None or component.y is None:
                        component.velocity = None
                        continue
                    if component.anchor.x is None or component.anchor.y is None:
                        component.velocity = None
                        continue
                    anchor_vel = component.anchor.velocity or (0.0, 0.0)
                    vx, vy = solve_crank_velocity(
                        component.x,
                        component.y,
                        component.anchor.x,
                        component.anchor.y,
                        anchor_vel[0],
                        anchor_vel[1],
                        component.radius,
                        omega,
                    )
                    if math.isnan(vx) or math.isnan(vy):
                        component.velocity = None
                    else:
                        component.velocity = (vx, vy)

                elif isinstance(component, RRRDyad):
                    if component.x is None or component.y is None:
                        component.velocity = None
                        continue
                    a1 = component.anchor1
                    a2 = component.anchor2
                    if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                        component.velocity = None
                        continue
                    v1 = a1.velocity or (0.0, 0.0)
                    v2 = a2.velocity or (0.0, 0.0)
                    vx, vy = solve_revolute_velocity(
                        component.x,
                        component.y,
                        a1.x,
                        a1.y,
                        v1[0],
                        v1[1],
                        a2.x,
                        a2.y,
                        v2[0],
                        v2[1],
                    )
                    if math.isnan(vx) or math.isnan(vy):
                        component.velocity = None
                    else:
                        component.velocity = (vx, vy)

                elif isinstance(component, RRPDyad):
                    if component.x is None or component.y is None:
                        component.velocity = None
                        continue
                    ra = component.revolute_anchor
                    la1 = component.line_anchor1
                    la2 = component.line_anchor2
                    if (
                        ra.x is None
                        or ra.y is None
                        or la1.x is None
                        or la1.y is None
                        or la2.x is None
                        or la2.y is None
                    ):
                        component.velocity = None
                        continue
                    vr = ra.velocity or (0.0, 0.0)
                    vl1 = la1.velocity or (0.0, 0.0)
                    vl2 = la2.velocity or (0.0, 0.0)
                    vx, vy = solve_prismatic_velocity(
                        component.x,
                        component.y,
                        ra.x,
                        ra.y,
                        vr[0],
                        vr[1],
                        component.distance,
                        la1.x,
                        la1.y,
                        vl1[0],
                        vl1[1],
                        la2.x,
                        la2.y,
                        vl2[0],
                        vl2[1],
                    )
                    if math.isnan(vx) or math.isnan(vy):
                        component.velocity = None
                    else:
                        component.velocity = (vx, vy)

                elif isinstance(component, FixedDyad):
                    if component.x is None or component.y is None:
                        component.velocity = None
                        continue
                    a1 = component.anchor1
                    a2 = component.anchor2
                    if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                        component.velocity = None
                        continue
                    v1 = a1.velocity or (0.0, 0.0)
                    v2 = a2.velocity or (0.0, 0.0)
                    vx, vy = solve_fixed_velocity(
                        component.x,
                        component.y,
                        a1.x,
                        a1.y,
                        v1[0],
                        v1[1],
                        a2.x,
                        a2.y,
                        v2[0],
                        v2[1],
                        component.distance,
                        component.angle,
                    )
                    if math.isnan(vx) or math.isnan(vy):
                        component.velocity = None
                    else:
                        component.velocity = (vx, vy)

                else:
                    # Unknown component type
                    component.velocity = None

            # Step 3: Compute accelerations
            for component in self._solve_order:
                if isinstance(component, Ground):
                    # Ground points have zero acceleration
                    component.acceleration = (0.0, 0.0)

                elif isinstance(component, Crank):
                    omega = getattr(component, "_omega", 0.0)
                    alpha = getattr(component, "_alpha", 0.0)
                    if component.x is None or component.y is None or component.velocity is None:
                        component.acceleration = None
                        continue
                    if component.anchor.x is None or component.anchor.y is None:
                        component.acceleration = None
                        continue
                    anchor_vel = component.anchor.velocity or (0.0, 0.0)
                    anchor_acc = component.anchor.acceleration or (0.0, 0.0)
                    ax, ay = solve_crank_acceleration(
                        component.x,
                        component.y,
                        component.velocity[0],
                        component.velocity[1],
                        component.anchor.x,
                        component.anchor.y,
                        anchor_vel[0],
                        anchor_vel[1],
                        anchor_acc[0],
                        anchor_acc[1],
                        component.radius,
                        omega,
                        alpha,
                    )
                    if math.isnan(ax) or math.isnan(ay):
                        component.acceleration = None
                    else:
                        component.acceleration = (ax, ay)

                elif isinstance(component, RRRDyad):
                    if component.x is None or component.y is None or component.velocity is None:
                        component.acceleration = None
                        continue
                    a1 = component.anchor1
                    a2 = component.anchor2
                    if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                        component.acceleration = None
                        continue
                    v1 = a1.velocity or (0.0, 0.0)
                    v2 = a2.velocity or (0.0, 0.0)
                    acc1 = a1.acceleration or (0.0, 0.0)
                    acc2 = a2.acceleration or (0.0, 0.0)
                    ax, ay = solve_revolute_acceleration(
                        component.x,
                        component.y,
                        component.velocity[0],
                        component.velocity[1],
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
                    if math.isnan(ax) or math.isnan(ay):
                        component.acceleration = None
                    else:
                        component.acceleration = (ax, ay)

                elif isinstance(component, RRPDyad):
                    if component.x is None or component.y is None or component.velocity is None:
                        component.acceleration = None
                        continue
                    ra = component.revolute_anchor
                    la1 = component.line_anchor1
                    la2 = component.line_anchor2
                    if (
                        ra.x is None
                        or ra.y is None
                        or la1.x is None
                        or la1.y is None
                        or la2.x is None
                        or la2.y is None
                    ):
                        component.acceleration = None
                        continue
                    vr = ra.velocity or (0.0, 0.0)
                    vl1 = la1.velocity or (0.0, 0.0)
                    vl2 = la2.velocity or (0.0, 0.0)
                    accr = ra.acceleration or (0.0, 0.0)
                    accl1 = la1.acceleration or (0.0, 0.0)
                    accl2 = la2.acceleration or (0.0, 0.0)
                    ax, ay = solve_prismatic_acceleration(
                        component.x,
                        component.y,
                        component.velocity[0],
                        component.velocity[1],
                        ra.x,
                        ra.y,
                        vr[0],
                        vr[1],
                        accr[0],
                        accr[1],
                        component.distance,
                        la1.x,
                        la1.y,
                        vl1[0],
                        vl1[1],
                        accl1[0],
                        accl1[1],
                        la2.x,
                        la2.y,
                        vl2[0],
                        vl2[1],
                        accl2[0],
                        accl2[1],
                    )
                    if math.isnan(ax) or math.isnan(ay):
                        component.acceleration = None
                    else:
                        component.acceleration = (ax, ay)

                elif isinstance(component, FixedDyad):
                    if component.x is None or component.y is None or component.velocity is None:
                        component.acceleration = None
                        continue
                    a1 = component.anchor1
                    a2 = component.anchor2
                    if a1.x is None or a1.y is None or a2.x is None or a2.y is None:
                        component.acceleration = None
                        continue
                    v1 = a1.velocity or (0.0, 0.0)
                    v2 = a2.velocity or (0.0, 0.0)
                    acc1 = a1.acceleration or (0.0, 0.0)
                    acc2 = a2.acceleration or (0.0, 0.0)
                    ax, ay = solve_fixed_acceleration(
                        component.x,
                        component.y,
                        component.velocity[0],
                        component.velocity[1],
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
                        component.distance,
                        component.angle,
                    )
                    if math.isnan(ax) or math.isnan(ay):
                        component.acceleration = None
                    else:
                        component.acceleration = (ax, ay)

                else:
                    # Unknown component type
                    component.acceleration = None

            # Yield results
            positions = tuple(d.position for d in self.components)
            velocities = tuple(d.velocity for d in self.components)
            accelerations = tuple(d.acceleration for d in self.components)
            yield positions, velocities, accelerations
