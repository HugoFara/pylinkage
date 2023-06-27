#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The linkage module defines useful classes for linkage definition.

Created on Fri Apr 16, 16:39:21 2021.

@author: HugoFara
"""
import warnings
from math import gcd, tau
from ..exceptions import HypostaticError
from .joint import (
    Static,
    Pivot,
    Crank,
    Fixed
)


class Linkage:
    """
    A linkage is a set of Joint objects.

    It is defined as a kinematic linkage.
    Coordinates are given relative to its own base.
    """

    __slots__ = "name", "joints", "_cranks", "_solve_order"

    def __init__(self, joints, order=None, name=None):
        """
        Define a linkage, a set of joints.

        Arguments
        ---------
        joints : list[Joint]
            All Joint to be part of the linkage
        order : list[Joint]
            Sequence to manually define resolution order for each step.
            It should be a subset of joints.
            Automatic computed order is experimental! The default is None.
        name : str, optional
            Human-readable name for the Linkage. If None, take the value
            str(id(self)). The default is None.
        """
        self.name = name
        if name is None:
            self.name = str(id(self))
        self.joints = tuple(joints)
        self._cranks = tuple(j for j in joints if isinstance(j, Crank))
        if order:
            self._solve_order = tuple(order)

    def __set_solve_order__(self, order):
        """Set constraints resolution order."""
        self._solve_order = order

    def __find_solving_order__(self):
        """Find solving order automatically (experimental)."""
        # TODO : test it
        warnings.warn(
            "Automatic solving order is still in experimental stage!"
        )
        solvable = [j for j in self.joints if isinstance(j, Static)]
        # True of new joints where added in the current pass
        solved_in_pass = True
        while len(solvable) < len(self.joints) and solved_in_pass:
            solved_in_pass = False
            for j in self.joints:
                if isinstance(j, Static) or j in solvable:
                    continue
                if j.joint0 in solvable:
                    if isinstance(j, Crank):
                        solvable.append(j)
                        solved_in_pass = True
                    elif j.joint1 in solvable:
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

    def rebuild(self, pos=None):
        """
        Redefine linkage joints and given initial positions to joints.

        Parameters
        ----------
        pos : tuple[tuple[int]]
            Initial positions for each joint in self.joints.
            Coordinates do not need to be precise, they will allow us the best
            fitting position between all possible positions satisfying
            constraints.
        """
        if not hasattr(self, '_solve_order'):
            self.__find_solving_order__()

        # Links parenting in descending order solely.
        # Parents joint do not have children.
        if pos is not None:
            # Definition of initial coordinates
            self.set_coords(pos)

    def get_coords(self):
        """Return the positions of each element in the system."""
        return [j.coord() for j in self.joints]

    def set_coords(self, coords):
        """Set coordinates for all joints of the linkage."""
        for joint, coord in zip(self.joints, coords):
            joint.set_coord(coord)

    def hyperstaticity(self):
        """Return the hyperstaticity (over-constrainment) degree of the linkage in 2D."""
        # TODO : test it
        warnings.warn(
            "The hyperstaticity method is in experimental stage! Results should be double-checked!"
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
            elif isinstance(j, Pivot):
                solids += 1
                # A Pivot Joint creates at least two pivots
                kinematic_undetermined += 4
                if not hasattr(j, 'joint1') or j.joint1 is None:
                    mobilities += 1
                else:
                    solids += 1
                    kinematic_undetermined += 2

        return 3 * (solids - 1) - kinematic_undetermined + mobilities

    def step(self, iterations=None, dt=1):
        """

        Make a step of the linkage.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to run across.
            If None, the default is self.get_rotation_period().
        dt : int, optional
            Amount of rotation to turn the cranks by.
            All cranks rotate by their self.angle * dt. The default is 1.

        Yields
        ------
        generator
            Iterable of the joints' coordinates.

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

    def get_num_constraints(self, flat=True):
        """
        Return numeric constraints of this linkage.

        Parameters
        ----------
        flat : bool
            Whether to force one-dimensional representation of constraints.
            The default is True.

        Returns
        -------
        constraints : list
            List of geometric constraints.
        """
        constraints = []
        for joint in self.joints:
            if flat:
                constraints.extend(joint.get_constraints())
            else:
                constraints.append(joint.get_constraints())
        return constraints

    def set_num_constraints(self, constraints, flat=True):
        """
        Set numeric constraints for this linkage.

        Numeric constraints are distances or angles between joints.

        Parameters
        ----------
        constraints : sequence
            Sequence of constraints to pass to the joints.
        flat : bool
            If True, constraints should be a one-dimensional sequence of floats.
            If False, constraints should be a sequence of tuples of digits.
            Each element will be passed to the set_constraints method of each
            corresponding Joint.
            The default is True.
        """
        if flat:
            # Is in charge of redistributing constraints
            dispatcher = iter(constraints)
            for joint in self.joints:
                if isinstance(joint, Static):
                    pass
                elif isinstance(joint, Crank):
                    joint.set_constraints(next(dispatcher))
                elif isinstance(joint, (Fixed, Pivot)):
                    joint.set_constraints(next(dispatcher), next(dispatcher))
        else:
            for joint, constraint in zip(self.joints, constraints):
                joint.set_constraints(*constraint)

    def get_rotation_period(self):
        """
        Return the number of iterations to finish in the previous state.

        Formally, it is the common denominator of all crank periods.

        Returns
        -------
        Number of iterations with dt=1.

        """
        periods = 1
        for j in self.joints:
            if isinstance(j, Crank):
                freq = round(tau / abs(j.angle))
                periods = periods * freq // gcd(periods, freq)
        return periods

    def set_completely(self, dimensions, positions, flat=True):
        """
        Set both dimension and initial positions.

        Parameters
        ----------
        dimensions : tuple of float or tuple of tuple of float
            List of dimensions.
        positions : tuple of float
        flat : bool, optional
            If the dimensions are in "flat mode".
            The default is True.

        Returns
        -------

        """
        self.set_num_constraints(dimensions, flat=flat)
        self.set_coords(positions)
