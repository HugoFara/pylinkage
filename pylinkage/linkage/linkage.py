#!/usr/bin/env python3
"""
The linkage module defines useful classes for linkage definition.

Created on Fri Apr 16, 16:39:21 2021.

@author: HugoFara
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable
from math import gcd, tau
from typing import TYPE_CHECKING

from ..exceptions import HypostaticError
from ..joints import Crank, Fixed, Revolute, Static
from ..joints.joint import Joint

if TYPE_CHECKING:
    from .._types import JointPositions


class Linkage:
    """A linkage is a set of Joint objects.

    It is defined as a kinematic linkage.
    Coordinates are given relative to its own base.
    """

    __slots__ = "name", "joints", "_cranks", "_solve_order"

    name: str
    joints: tuple[Joint, ...]
    _cranks: tuple[Crank, ...]
    _solve_order: tuple[Joint, ...]

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
        if order:
            self._solve_order = tuple(order)

    def __set_solve_order__(self, order: Iterable[Joint]) -> None:
        """Set constraints resolution order."""
        self._solve_order = tuple(order)

    def __find_solving_order__(self) -> tuple[Joint, ...]:
        """Find solving order automatically (experimental)."""
        # TODO : test it
        warnings.warn(
            "Automatic solving order is still in experimental stage!",
            stacklevel=2,
        )
        solvable: list[Joint] = [j for j in self.joints if isinstance(j, Static)]
        # True if new joints were added in the current pass
        solved_in_pass = True
        while len(solvable) < len(self.joints) and solved_in_pass:
            solved_in_pass = False
            for j in self.joints:
                if isinstance(j, Static) or j in solvable:
                    continue
                if j.joint0 in solvable and (isinstance(j, Crank) or j.joint1 in solvable):
                    solvable.append(j)
                    solved_in_pass = True
        if len(solvable) < len(self.joints):
            raise HypostaticError(
                'Unable to determine automatic order!'
                'Those joints are left unsolved:'
                ','.join(str(j) for j in self.joints if j not in solvable)
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

    def hyperstaticity(self) -> int:
        """Return the hyperstaticity (over-constrainment) degree of the linkage in 2D."""
        # TODO : test it
        warnings.warn(
            "The hyperstaticity method is in experimental stage! Results should be double-checked!",
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
        if iterations is None:
            iterations = self.get_rotation_period()
        for _ in range(iterations):
            for j in self._solve_order:
                if isinstance(j, Crank):
                    j.reload(dt)
                else:
                    j.reload()
            yield tuple(j.coord() for j in self.joints)

    def get_num_constraints(self, flat: bool = True) -> list[float | None] | list[tuple[float | None, ...]]:
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

        :param constraints: Sequence of constraints to pass to the joints.
        :param flat: If True, constraints should be a one-dimensional sequence of floats.
            If False, constraints should be a sequence of tuples of digits.
            Each element will be passed to the set_constraints method of each
            corresponding Joint.
            (Default value = True).
        """
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
                joint.set_constraints(*constraint)  # type: ignore[arg-type]

    def get_rotation_period(self) -> int:
        """The number of iterations to finish in the previous state.

        Formally, it is the common denominator of all crank periods.

        :returns: Number of iterations with dt=1.
        """
        periods = 1
        for j in self.joints:
            if isinstance(j, Crank):
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
