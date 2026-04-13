#!/usr/bin/env python3
"""
The linkage module defines useful classes for linkage definition.

Created on Fri Apr 16, 16:39:21 2021.

@author: HugoFara
"""

import warnings
from collections.abc import Generator, Iterable
from math import gcd, tau
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .._types import JointPositions
from ..components._base import Component
from ..exceptions import UnderconstrainedError

if TYPE_CHECKING:
    from ..solver import SolverData
    from .sensitivity import SensitivityAnalysis, ToleranceAnalysis
    from .transmission import StrokeAnalysis, TransmissionAngleAnalysis


def _is_ground(j: Component) -> bool:
    """Check if a component is a ground/anchor (no constraints, not a driver)."""
    return len(j.get_constraints()) == 0


def _is_driver(j: Component) -> bool:
    """Check if a component is a motor-driven driver (crank or linear actuator)."""
    # New actuators expose angular_velocity
    if hasattr(j, "angular_velocity"):
        return True
    # Legacy Crank: has 'angle' attr, single constraint, single anchor
    # (Fixed also has 'angle' but has 2 constraints and 2 anchors)
    return (
        hasattr(j, "angle")
        and len(j.get_constraints()) == 1
        and len(getattr(j, "anchors", ())) <= 1
    )


def _get_anchors(j: Component) -> tuple[Component, ...]:
    """Get parent components of a joint/component."""
    anchors = getattr(j, "anchors", None)
    if anchors is not None:
        return anchors  # type: ignore[no-any-return]
    return ()


class Linkage:
    """A linkage is a set of Joint objects.

    It is defined as a kinematic linkage.
    Coordinates are given relative to its own base.
    """

    __slots__ = "name", "joints", "_cranks", "_solve_order", "_solver_data"

    name: str
    joints: tuple[Component, ...]
    _cranks: tuple[Component, ...]
    _solve_order: tuple[Component, ...]
    _solver_data: "SolverData | None"

    def __init__(
        self,
        joints: Iterable[Component],
        order: Iterable[Component] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Define a linkage, a set of joints.

        :param joints: All Component/Joint to be part of the linkage.
        :param order: Sequence to manually define resolution order for each step.
            It should be a subset of joints.
            Automatic computed order is experimental!
            (Default value = None).
        :param name: Human-readable name for the Linkage.
            If None, take the value str(id(self)).
            (Default value = None).
        """
        warnings.warn(
            "pylinkage.linkage.Linkage is deprecated and will be removed in "
            "version 1.0.0. Use pylinkage.simulation.Linkage with the "
            "component/actuator/dyad API instead. "
            "See CLAUDE.md 'API Migration' section for the mapping.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.name = name if name is not None else str(id(self))
        self.joints = tuple(joints)
        self._cranks = tuple(j for j in self.joints if _is_driver(j))
        self._solver_data = None
        if order:
            self._solve_order = tuple(order)

    def __set_solve_order__(self, order: Iterable[Component]) -> None:
        """Set constraints resolution order."""
        self._solve_order = tuple(order)

    def __find_solving_order__(self) -> tuple[Component, ...]:
        """Find solving order automatically (experimental).

        This method attempts to determine the order in which joints should be solved
        by finding joints whose parent anchors are already in the solvable list.

        Ground components (zero constraints) are automatically detected from parent
        references, so tuple shortcuts like ``joint0=(0, 0)`` work correctly.

        Returns:
            Tuple of components in solvable order.

        Raises:
            UnderconstrainedError: If the linkage cannot be automatically solved.
        """
        warnings.warn(
            "Automatic solving order is still in experimental stage!",
            stacklevel=2,
        )
        # Collect all ground components (zero constraints = no solving needed)
        solvable: list[Component] = [j for j in self.joints if _is_ground(j)]
        # Also discover implicit grounds referenced as anchors but not in self.joints
        for j in self.joints:
            for anchor in _get_anchors(j):
                if _is_ground(anchor) and anchor not in solvable:
                    solvable.append(anchor)
        # Track which joints from self.joints have been solved
        # (solvable may contain implicit grounds not in self.joints)
        joints_solved = sum(1 for j in self.joints if j in solvable)
        # True if new joints were added in the current pass
        solved_in_pass = True
        while joints_solved < len(self.joints) and solved_in_pass:
            solved_in_pass = False
            for j in self.joints:
                if _is_ground(j) or j in solvable:
                    continue
                anchors = _get_anchors(j)
                if anchors and all(anchor in solvable for anchor in anchors):
                    solvable.append(j)
                    joints_solved += 1
                    solved_in_pass = True
        if joints_solved < len(self.joints):
            raise UnderconstrainedError(
                "Unable to determine automatic order! "
                "Those joints are left unsolved: "
                + ", ".join(str(j) for j in self.joints if j not in solvable)
            )
        self._solve_order = tuple(solvable)
        return self._solve_order

    def rebuild(
        self,
        initial_positions: JointPositions | None = None,
        *,
        pos: JointPositions | None = None,
    ) -> None:
        """Redefine linkage joints and given initial positions to joints.

        :param initial_positions: Initial positions for each joint in self.joints.
            Coordinates do not need to be precise, they will allow us the best
            fitting position between all possible positions satisfying
            constraints. (Default value = None).
        :param pos: Deprecated alias for ``initial_positions``.
        """
        if pos is not None:
            initial_positions = pos

        if not hasattr(self, "_solve_order"):
            self.__find_solving_order__()

        # Links parenting in descending order solely.
        # Parents joint do not have children.
        if initial_positions is not None:
            # Definition of initial coordinates
            self.set_coords(initial_positions)

    def get_coords(self) -> list[tuple[float | None, float | None]]:
        """Return the positions of each element in the system."""
        return [j.coord() for j in self.joints]

    def set_coords(self, coords: JointPositions) -> None:
        """Set coordinates for all joints of the linkage.

        :param coords: Joint coordinates.
        """
        for joint, coord in zip(self.joints, coords, strict=False):
            joint.set_coord(*coord)

    def indeterminacy(self) -> int:
        """Return the static indeterminacy degree of the linkage in 2D.

        Uses a variant of the Gruebler-Kutzbach criterion for 2D planar mechanisms:
            DOF = 3 * (n - 1) - kinematic_pairs + mobilities

        Where:
            - n = number of bodies (including the ground/frame)
            - kinematic_pairs = sum of constraint DOFs removed by joints
            - mobilities = input degrees of freedom (e.g., from motors/cranks)

        A positive return value indicates the mechanism is under-constrained,
        zero means it's exactly constrained, and negative means it's
        over-constrained.

        Returns:
            The indeterminacy degree (negative DOF when over-constrained).

        Note:
            This implementation is experimental and results should be verified
            for complex mechanisms. The algorithm counts bodies and constraints
            based on joint types (Static, Crank, Revolute, Fixed).
        """
        warnings.warn(
            "The indeterminacy method is in experimental stage! Results should be double-checked!",
            stacklevel=2,
        )
        # We have at least the frame
        solids = 1
        mobilities = 1
        kinematic_undetermined = 0
        for j in self.joints:
            n_constraints = len(j.get_constraints())
            if n_constraints == 0:
                # Ground/Static - no contribution
                pass
            elif _is_driver(j):
                # Crank/actuator - one body, one revolute pair
                solids += 1
                kinematic_undetermined += 2
            elif n_constraints == 2:
                n_anchors = len(_get_anchors(j))
                solids += 1
                # RRR dyad / Revolute: 2 links + 3 revolute joints
                kinematic_undetermined += 4
                if n_anchors < 2:
                    mobilities += 1
                else:
                    solids += 1
                    kinematic_undetermined += 2

        return 3 * (solids - 1) - kinematic_undetermined + mobilities

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1,
        skip_unbuildable: bool = False,
    ) -> Generator[tuple[tuple[float | None, float | None], ...], None, None]:
        """Make a step of the linkage.

        :param iterations: Number of iterations to run across.
            If None, the default is self.get_rotation_period().
            (Default value = None).
        :param dt: Amount of rotation to turn the cranks by.
            All cranks rotate by their self.angle * dt. The default is 1.
            (Default value = 1).
        :param skip_unbuildable: If True, yield None-coordinate tuples for
            positions where the linkage cannot be assembled, instead of
            raising UnbuildableError. The crank continues advancing so
            the simulation resumes when the linkage becomes buildable
            again. (Default value = False).

        :returns: Iterable of the joints' coordinates.
        """
        if not hasattr(self, "_solve_order"):
            self.__find_solving_order__()
        if iterations is None:
            iterations = self.get_rotation_period()
        if skip_unbuildable:
            from ..exceptions import UnbuildableError

            none_positions = tuple(
                (None, None) for _ in self.joints
            )
            for _ in range(iterations):
                try:
                    for j in self._solve_order:
                        j.reload(dt)
                except UnbuildableError:
                    yield none_positions
                else:
                    yield tuple(j.coord() for j in self.joints)
        else:
            for _ in range(iterations):
                for j in self._solve_order:
                    j.reload(dt)
                yield tuple(j.coord() for j in self.joints)

    def compile(self) -> None:
        """Prepare numba-optimized solver for fast simulation.

        This converts the linkage structure into numeric arrays for
        use by the numba-compiled simulation loop. The compilation
        is done automatically on first call to step_fast(), but can
        be called explicitly to control when the overhead occurs.

        The compiled solver is cached and reused until invalidated
        by changes to constraints (set_num_constraints) or structure.
        """
        from ..solver import linkage_to_solver_data

        if not hasattr(self, "_solve_order"):
            self.__find_solving_order__()
        self._solver_data = linkage_to_solver_data(self)  # type: ignore[operator]

    def step_fast(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> NDArray[np.float64]:
        """Run simulation using numba-optimized solver.

        This is significantly faster than step() for large iteration counts,
        as it avoids Python method call overhead in the hot loop.

        The first call may be slower due to numba compilation (JIT warm-up).
        Subsequent calls will be fast.

        Args:
            iterations: Number of iterations to run. If None, uses
                get_rotation_period(). (Default: None)
            dt: Amount of rotation per step. Cranks rotate by their
                angle * dt. (Default: 1)

        Returns:
            Array of shape (iterations, n_joints, 2) containing all joint
            positions at each step. Access individual step positions via
            trajectory[step_idx, joint_idx] which gives (x, y).

        Note:
            If any configuration becomes unbuildable during simulation,
            the corresponding positions will be NaN. Check for this with
            np.isnan(trajectory).any() or use solver.has_nan_positions().

        Example:
            >>> trajectory = linkage.step_fast(iterations=1000)
            >>> print(trajectory.shape)  # (1000, n_joints, 2)
            >>> # Get position of joint 2 at step 100:
            >>> pos = trajectory[100, 2]  # (x, y)
        """
        from ..solver import (
            simulate,
            solver_data_to_linkage,
            update_solver_positions,
        )

        # Auto-compile if needed
        if self._solver_data is None:
            self.compile()

        assert self._solver_data is not None  # for type checker

        if iterations is None:
            iterations = self.get_rotation_period()

        # Sync positions from joints to solver data
        update_solver_positions(self._solver_data, self)  # type: ignore[operator]

        # Run numba simulation
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

        # Sync final positions back to joints
        solver_data_to_linkage(self._solver_data, self)  # type: ignore[operator]

        return trajectory

    def step_fast_with_kinematics(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Run simulation with velocity and acceleration using numba-optimized solver.

        This extends step_fast() to also compute velocities and accelerations
        at each step. Requires that omega (and optionally alpha) is set on crank
        joints to specify angular velocities and accelerations.

        Args:
            iterations: Number of iterations to run. If None, uses
                get_rotation_period(). (Default: None)
            dt: Amount of rotation per step. Cranks rotate by their
                angle * dt. (Default: 1)

        Returns:
            Tuple of (positions, velocities, accelerations) arrays, each with
            shape (iterations, n_joints, 2).

        Example:
            >>> linkage.set_input_velocity(crank, omega=10.0, alpha=0.0)
            >>> pos, vel, acc = linkage.step_fast_with_kinematics(1000)
            >>> print(vel[100, 2])  # velocity of joint 2 at step 100
            >>> print(acc[100, 2])  # acceleration of joint 2 at step 100
        """
        from ..solver import (
            simulate_with_kinematics,
            solver_data_to_linkage,
            update_solver_positions,
        )

        # Auto-compile if needed
        if self._solver_data is None:
            self.compile()

        assert self._solver_data is not None  # for type checker

        if iterations is None:
            iterations = self.get_rotation_period()

        # Sync positions from joints to solver data
        update_solver_positions(self._solver_data, self)  # type: ignore[operator]

        # Initialize velocity, acceleration, and kinematics arrays if not present
        n_joints = len(self.joints)
        n_cranks = len(self._cranks)
        if self._solver_data.velocities is None:
            self._solver_data.velocities = np.zeros((n_joints, 2), dtype=np.float64)
        if self._solver_data.accelerations is None:
            self._solver_data.accelerations = np.zeros((n_joints, 2), dtype=np.float64)
        if self._solver_data.omega_values is None:
            self._solver_data.omega_values = np.zeros(n_cranks, dtype=np.float64)
        if self._solver_data.alpha_values is None:
            self._solver_data.alpha_values = np.zeros(n_cranks, dtype=np.float64)
        if self._solver_data.crank_indices is None:
            crank_indices = []
            for i, j in enumerate(self.joints):
                if _is_driver(j):
                    crank_indices.append(i)
            self._solver_data.crank_indices = np.array(crank_indices, dtype=np.int32)

        # Sync omega and alpha values from cranks
        for i, crank in enumerate(self._cranks):
            omega = getattr(crank, "omega", None) or 0.0
            alpha = getattr(crank, "alpha", None) or 0.0
            self._solver_data.omega_values[i] = omega
            self._solver_data.alpha_values[i] = alpha

        # Run numba simulation with kinematics
        pos_trajectory, vel_trajectory, acc_trajectory = simulate_with_kinematics(
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

        # Sync final positions, velocities, and accelerations back to joints
        solver_data_to_linkage(self._solver_data, self)  # type: ignore[operator]

        return pos_trajectory, vel_trajectory, acc_trajectory

    def set_input_velocity(
        self,
        crank: Component,
        omega: float,
        alpha: float = 0.0,
    ) -> None:
        """Set angular velocity and acceleration for a crank joint.

        This is used for kinematics computation (velocity/acceleration analysis).

        Args:
            crank: The crank joint to set velocity for.
            omega: Angular velocity in rad/s.
            alpha: Angular acceleration in rad/s² (default 0).

        Raises:
            ValueError: If the crank is not part of this linkage.

        Example:
            >>> linkage.set_input_velocity(crank, omega=10.0)  # 10 rad/s
            >>> positions, velocities = linkage.step_fast_with_kinematics(100)
        """
        if crank not in self._cranks:
            raise ValueError(f"{crank} is not a crank in this linkage")
        crank.omega = omega  # type: ignore[attr-defined]
        crank.alpha = alpha  # type: ignore[attr-defined]

    def get_velocities(self) -> list[tuple[float, float] | None]:
        """Return velocities for all joints.

        Returns:
            List of (vx, vy) tuples, one per joint. Returns None for joints
            whose velocity has not been computed.
        """
        return [j.velocity for j in self.joints]

    def get_accelerations(self) -> list[tuple[float, float] | None]:
        """Return accelerations for all joints.

        Returns:
            List of (ax, ay) tuples, one per joint. Returns None for joints
            whose acceleration has not been computed.
        """
        return [j.acceleration for j in self.joints]

    def get_num_constraints(
        self, flat: bool = True
    ) -> list[float | None] | list[tuple[float | None, ...]]:
        """Numeric constraints of this linkage.

        :param flat: Whether to force one-dimensional representation of constraints.
            The default is True.

        :returns: List of geometric constraints.
        """
        constraints: list[float | None] | list[tuple[float | None, ...]] = []
        for joint in self.joints:
            if flat:
                constraints.extend(joint.get_constraints())  # type: ignore[arg-type]
            else:
                constraints.append(joint.get_constraints())  # type: ignore[arg-type]
        return constraints

    def set_num_constraints(
        self,
        constraints: Iterable[float] | Iterable[tuple[float, ...]],
        flat: bool = True,
    ) -> None:
        """Set numeric constraints for this linkage.

        Numeric constraints are distances or angles between joints.

        Note:
            This invalidates any cached solver data. The next call to
            step_fast() will recompile the solver automatically.

        :param constraints: Sequence of constraints to pass to the joints.
        :param flat: If True, constraints should be a one-dimensional sequence of floats.
            If False, constraints should be a sequence of tuples of digits.
            Each element will be passed to the set_constraints method of each
            corresponding Joint.
            (Default value = True).
        """
        # Invalidate cached solver data
        self._solver_data = None

        if flat:
            # Distribute constraints based on each joint's constraint count
            dispatcher = iter(constraints)
            for joint in self.joints:
                n = len(joint.get_constraints())
                if n > 0:
                    args = [next(dispatcher) for _ in range(n)]
                    joint.set_constraints(*args)  # type: ignore[arg-type]
        else:
            for joint, constraint in zip(self.joints, constraints, strict=False):
                joint.set_constraints(*constraint)  # type: ignore[misc]

    def get_rotation_period(self) -> int:
        """The number of iterations to finish in the previous state.

        Formally, it is the common denominator of all crank periods.

        :returns: Number of iterations with dt=1.
        """
        periods = 1
        for j in self._cranks:
            # Support both new (angular_velocity) and legacy (angle) attribute names
            angle = getattr(j, "angular_velocity", None) or getattr(j, "angle", None)
            if angle is not None:
                freq = round(tau / abs(angle))
                periods = periods * freq // gcd(periods, freq)
        return periods

    def set_completely(
        self,
        dimensions: Iterable[float] | Iterable[tuple[float, ...]],
        positions: JointPositions,
        flat: bool = True,
    ) -> None:
        """Set both dimension and initial positions.

        :param dimensions: List of dimensions.
        :param positions: Initial positions.
        :param flat: If the dimensions are in "flat mode".
            The default is True.
        """
        self.set_num_constraints(dimensions, flat=flat)
        self.set_coords(positions)

    def to_dict(self) -> dict[str, Any]:
        """Convert this linkage to a dictionary representation.

        The dictionary can be used for serialization or reconstruction.

        Returns:
            Dictionary containing the linkage's data including joints and solve order.

        Example:
            >>> data = linkage.to_dict()
            >>> new_linkage = Linkage.from_dict(data)
        """
        from .serialization import linkage_to_dict

        return linkage_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Linkage":
        """Create a linkage from a dictionary representation.

        Args:
            data: Dictionary containing linkage data (as produced by to_dict()).

        Returns:
            A new Linkage instance.

        Example:
            >>> data = linkage.to_dict()
            >>> new_linkage = Linkage.from_dict(data)
        """
        from .serialization import linkage_from_dict

        return linkage_from_dict(data)

    def to_json(self, path: str | Path) -> None:
        """Save this linkage to a JSON file.

        Args:
            path: Path to the output JSON file.

        Example:
            >>> linkage.to_json("my_linkage.json")
            >>> loaded = Linkage.from_json("my_linkage.json")
        """
        from .serialization import save_to_json

        save_to_json(self, path)

    @classmethod
    def from_json(cls, path: str | Path) -> "Linkage":
        """Load a linkage from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new Linkage instance.

        Example:
            >>> linkage.to_json("my_linkage.json")
            >>> loaded = Linkage.from_json("my_linkage.json")
        """
        from .serialization import load_from_json

        return load_from_json(path)

    def simulation(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> "Simulation":
        """Create a simulation context manager for the linkage.

        This provides a convenient way to run and iterate over linkage simulations
        with automatic resource management.

        Example:
            >>> with linkage.simulation(iterations=100) as sim:
            ...     for step, coords in sim:
            ...         process(coords)

        :param iterations: Number of iterations to run. If None, uses get_rotation_period().
        :param dt: Time step for crank rotation. Default is 1.
        :returns: A Simulation context manager.
        """
        return Simulation(self, iterations=iterations, dt=dt)

    def transmission_angle(self) -> float:
        """Get the transmission angle at the current linkage position.

        For a four-bar linkage, this is the angle between the coupler
        and rocker links at their common joint.

        Returns:
            Transmission angle in degrees (0-180).

        Raises:
            ValueError: If joints cannot be auto-detected.

        Example:
            >>> angle = linkage.transmission_angle()
            >>> print(f"Current transmission angle: {angle:.1f}°")
        """
        from .transmission import transmission_angle_at_position

        return transmission_angle_at_position(self)

    def analyze_transmission(
        self,
        iterations: int | None = None,
        acceptable_range: tuple[float, float] = (40.0, 140.0),
    ) -> "TransmissionAngleAnalysis":
        """Analyze transmission angle over a full motion cycle.

        For a standard four-bar, joints are auto-detected.

        Args:
            iterations: Number of simulation steps. Defaults to one full rotation.
            acceptable_range: (min, max) acceptable angles in degrees.

        Returns:
            TransmissionAngleAnalysis with min/max/mean/is_acceptable.

        Raises:
            ValueError: If joints cannot be detected or no valid positions found.

        Example:
            >>> analysis = linkage.analyze_transmission()
            >>> print(f"Min: {analysis.min_angle:.1f}°")
            >>> print(f"Max: {analysis.max_angle:.1f}°")
            >>> print(f"Acceptable: {analysis.is_acceptable}")
        """
        from .transmission import analyze_transmission as _analyze

        return _analyze(self, iterations, acceptable_range)

    def stroke_position(self) -> float:
        """Get the slide position of a prismatic joint at current position.

        For a linkage with a Prismatic joint, this returns the slide
        position along the prismatic axis.

        Returns:
            Slide position along the axis.

        Raises:
            ValueError: If no Prismatic joint found.

        Example:
            >>> pos = linkage.stroke_position()
            >>> print(f"Current slide position: {pos:.2f}")
        """
        from .transmission import stroke_at_position

        return stroke_at_position(self)

    def analyze_stroke(
        self,
        iterations: int | None = None,
    ) -> "StrokeAnalysis":
        """Analyze stroke/slide position over a full motion cycle.

        For a linkage with a Prismatic joint, this tracks the slide
        position throughout the motion cycle.

        Args:
            iterations: Number of simulation steps. Defaults to one full rotation.

        Returns:
            StrokeAnalysis with min/max/range/mean statistics.

        Raises:
            ValueError: If no Prismatic joint found or no valid positions.

        Example:
            >>> analysis = linkage.analyze_stroke()
            >>> print(f"Stroke range: {analysis.stroke_range:.2f}")
            >>> print(f"Min: {analysis.min_position:.2f}")
            >>> print(f"Max: {analysis.max_position:.2f}")
        """
        from .transmission import analyze_stroke as _analyze

        return _analyze(self, iterations=iterations)

    def sensitivity_analysis(
        self,
        output_joint: object | int | None = None,
        delta: float = 0.01,
        include_transmission: bool = True,
        iterations: int | None = None,
    ) -> "SensitivityAnalysis":
        """Analyze sensitivity of output path to each constraint dimension.

        For each constraint, this function perturbs the value by delta
        and measures how much the output path changes.

        Args:
            output_joint: Joint to measure, index, or None (auto-detect last).
            delta: Relative perturbation magnitude (e.g., 0.01 for 1%).
            include_transmission: Whether to also measure transmission angle.
            iterations: Number of simulation steps. Defaults to one full rotation.

        Returns:
            SensitivityAnalysis with sensitivity coefficients and rankings.

        Example:
            >>> analysis = linkage.sensitivity_analysis(delta=0.01)
            >>> print(f"Most sensitive: {analysis.most_sensitive}")
            >>> for name, sens in analysis.sensitivity_ranking:
            ...     print(f"{name}: {sens:.4f}")
        """
        from .sensitivity import analyze_sensitivity as _analyze

        return _analyze(self, output_joint, delta, include_transmission, iterations)

    def tolerance_analysis(
        self,
        tolerances: dict[str, float],
        output_joint: object | int | None = None,
        iterations: int | None = None,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> "ToleranceAnalysis":
        """Analyze manufacturing tolerance effects via Monte Carlo simulation.

        Each sample randomly perturbs constraints within tolerance bounds
        and simulates the linkage to measure output variation.

        Args:
            tolerances: Constraint name -> tolerance value (e.g., {"crank_radius": 0.1}).
            output_joint: Joint to measure, index, or None (auto-detect last).
            iterations: Number of simulation steps. Defaults to one full rotation.
            n_samples: Number of Monte Carlo samples.
            seed: Random seed for reproducibility.

        Returns:
            ToleranceAnalysis with statistics and output cloud for visualization.

        Example:
            >>> tolerances = {"Crank_radius": 0.1, "Revolute_dist1": 0.2}
            >>> analysis = linkage.tolerance_analysis(tolerances, n_samples=500)
            >>> print(f"Max deviation: {analysis.max_deviation:.4f}")
            >>> analysis.plot_cloud()
        """
        from .sensitivity import analyze_tolerance as _analyze

        return _analyze(self, tolerances, output_joint, iterations, n_samples, seed)


class Simulation:
    """Context manager for linkage simulation.

    Provides a clean interface for iterating over linkage positions
    with automatic setup and teardown.

    Example:
        >>> with linkage.simulation(iterations=100) as sim:
        ...     for step, coords in sim:
        ...         # coords is a tuple of (x, y) for each joint
        ...         print(f"Step {step}: {coords}")
    """

    __slots__ = "_linkage", "_iterations", "_dt", "_initial_coords"

    def __init__(
        self,
        linkage: Linkage,
        iterations: int | None = None,
        dt: float = 1,
    ) -> None:
        """Initialize the simulation.

        :param linkage: The linkage to simulate.
        :param iterations: Number of iterations. If None, uses linkage.get_rotation_period().
        :param dt: Time step for crank rotation.
        """
        self._linkage = linkage
        self._iterations = iterations
        self._dt = dt
        self._initial_coords: list[tuple[float | None, float | None]] | None = None

    def __enter__(self) -> "Simulation":
        """Enter the simulation context, saving initial state."""
        self._initial_coords = self._linkage.get_coords()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the simulation context, restoring initial state."""
        if self._initial_coords is not None:
            self._linkage.set_coords(self._initial_coords)

    def __iter__(
        self,
    ) -> Generator[tuple[int, tuple[tuple[float | None, float | None], ...]], None, None]:
        """Iterate over simulation steps.

        Yields:
            Tuples of (step_number, joint_coordinates) where step_number is 0-indexed
            and joint_coordinates is a tuple of (x, y) for each joint.
        """
        yield from enumerate(self._linkage.step(iterations=self._iterations, dt=self._dt))

    @property
    def linkage(self) -> Linkage:
        """The linkage being simulated."""
        return self._linkage

    @property
    def iterations(self) -> int:
        """Number of iterations for this simulation."""
        if self._iterations is None:
            return self._linkage.get_rotation_period()
        return self._iterations
