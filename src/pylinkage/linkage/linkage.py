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
from ..exceptions import UnderconstrainedError
from ..joints import Crank, Fixed, Revolute, Static
from ..joints.joint import Joint

if TYPE_CHECKING:
    from ..solver import SolverData


class Linkage:
    """A linkage is a set of Joint objects.

    It is defined as a kinematic linkage.
    Coordinates are given relative to its own base.
    """

    __slots__ = "name", "joints", "_cranks", "_solve_order", "_solver_data"

    name: str
    joints: tuple[Joint, ...]
    _cranks: tuple[Crank, ...]
    _solve_order: tuple[Joint, ...]
    _solver_data: "SolverData | None"

    def __init__(
        self,
        joints: Iterable[Joint],
        order: Iterable[Joint] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Define a linkage, a set of joints.

        :param joints: All Joint to be part of the linkage.
        :param order: Sequence to manually define resolution order for each step.
            It should be a subset of joints.
            Automatic computed order is experimental!
            (Default value = None).
        :param name: Human-readable name for the Linkage.
            If None, take the value str(id(self)).
            (Default value = None).
        """
        self.name = name if name is not None else str(id(self))
        self.joints = tuple(joints)
        self._cranks = tuple(j for j in self.joints if isinstance(j, Crank))
        self._solver_data = None
        if order:
            self._solve_order = tuple(order)

    def __set_solve_order__(self, order: Iterable[Joint]) -> None:
        """Set constraints resolution order."""
        self._solve_order = tuple(order)

    def __find_solving_order__(self) -> tuple[Joint, ...]:
        """Find solving order automatically (experimental).

        This method attempts to determine the order in which joints should be solved
        by finding joints whose parent joints are already in the solvable list.

        Anchor joints (Static instances) are automatically detected from parent
        references, so tuple shortcuts like ``joint0=(0, 0)`` work correctly.

        Returns:
            Tuple of joints in solvable order.

        Raises:
            UnderconstrainedError: If the linkage cannot be automatically solved.
        """
        warnings.warn(
            "Automatic solving order is still in experimental stage!",
            stacklevel=2,
        )
        # Collect all Static joints: both explicit ones in self.joints
        # and implicit ones created from tuple shortcuts in parent references
        solvable: list[Joint] = [j for j in self.joints if isinstance(j, Static)]
        for j in self.joints:
            if isinstance(j.joint0, Static) and j.joint0 not in solvable:
                solvable.append(j.joint0)
            if isinstance(j.joint1, Static) and j.joint1 not in solvable:
                solvable.append(j.joint1)
        # Track which joints from self.joints have been solved
        # (solvable may contain implicit Statics not in self.joints)
        joints_solved = sum(1 for j in self.joints if j in solvable)
        # True if new joints were added in the current pass
        solved_in_pass = True
        while joints_solved < len(self.joints) and solved_in_pass:
            solved_in_pass = False
            for j in self.joints:
                if isinstance(j, Static) or j in solvable:
                    continue
                if j.joint0 in solvable and (isinstance(j, Crank) or j.joint1 in solvable):
                    solvable.append(j)
                    joints_solved += 1
                    solved_in_pass = True
        if joints_solved < len(self.joints):
            raise UnderconstrainedError(
                'Unable to determine automatic order! '
                'Those joints are left unsolved: '
                + ', '.join(str(j) for j in self.joints if j not in solvable)
            )
        self._solve_order = tuple(solvable)
        return self._solve_order

    def rebuild(self, pos: JointPositions | None = None) -> None:
        """Redefine linkage joints and given initial positions to joints.

        :param pos: Initial positions for each joint in self.joints.
            Coordinates do not need to be precise, they will allow us the best
            fitting position between all possible positions satisfying
            constraints. (Default value = None).
        """
        if not hasattr(self, '_solve_order'):
            self.__find_solving_order__()

        # Links parenting in descending order solely.
        # Parents joint do not have children.
        if pos is not None:
            # Definition of initial coordinates
            self.set_coords(pos)

    def get_coords(self) -> list[tuple[float | None, float | None]]:
        """Return the positions of each element in the system."""
        return [j.coord() for j in self.joints]

    def set_coords(self, coords: JointPositions) -> None:
        """Set coordinates for all joints of the linkage.

        :param coords: Joint coordinates.
        """
        for joint, coord in zip(self.joints, coords):
            joint.set_coord(coord)

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
            if isinstance(j, (Static, Fixed)):
                pass
            elif isinstance(j, Crank):
                solids += 1
                kinematic_undetermined += 2
            elif isinstance(j, Revolute):
                solids += 1
                # A Revolute Joint creates at least two revolute joints
                kinematic_undetermined += 4
                if not hasattr(j, 'joint1') or j.joint1 is None:
                    mobilities += 1
                else:
                    solids += 1
                    kinematic_undetermined += 2

        return 3 * (solids - 1) - kinematic_undetermined + mobilities

    def step(
        self,
        iterations: int | None = None,
        dt: float = 1,
    ) -> Generator[tuple[tuple[float | None, float | None], ...], None, None]:
        """Make a step of the linkage.

        :param iterations: Number of iterations to run across.
            If None, the default is self.get_rotation_period().
            (Default value = None).
        :param dt: Amount of rotation to turn the cranks by.
            All cranks rotate by their self.angle * dt. The default is 1.
            (Default value = 1).

        :returns: Iterable of the joints' coordinates.
        """
        if not hasattr(self, '_solve_order'):
            self.__find_solving_order__()
        if iterations is None:
            iterations = self.get_rotation_period()
        for _ in range(iterations):
            for j in self._solve_order:
                if isinstance(j, Crank):
                    j.reload(dt)
                else:
                    j.reload()
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
            # Is in charge of redistributing constraints
            dispatcher = iter(constraints)
            for joint in self.joints:
                if isinstance(joint, Static):
                    pass
                elif isinstance(joint, Crank):
                    joint.set_constraints(next(dispatcher))  # type: ignore[arg-type]
                elif isinstance(joint, (Fixed, Revolute)):
                    joint.set_constraints(next(dispatcher), next(dispatcher))  # type: ignore[arg-type]
        else:
            for joint, constraint in zip(self.joints, constraints):
                joint.set_constraints(*constraint)  # type: ignore[misc]

    def get_rotation_period(self) -> int:
        """The number of iterations to finish in the previous state.

        Formally, it is the common denominator of all crank periods.

        :returns: Number of iterations with dt=1.
        """
        periods = 1
        for j in self.joints:
            if isinstance(j, Crank) and j.angle is not None:
                freq = round(tau / abs(j.angle))
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
        yield from enumerate(
            self._linkage.step(iterations=self._iterations, dt=self._dt)
        )

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
