#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The linkage module defines useful classes for linkage definition.

Created on Fri Apr 16 16:39:21 2021.

@author: HugoFara
"""
import abc
from math import atan2, gcd, tau
from abc import ABC
from .exceptions import HypostaticError, UnbuildableError
from .geometry import sqr_dist, circle_intersect, cyl_to_cart


class Joint(ABC):
    """
    Geometric constraint expressed by two joints.

    Abstract class, should not be used by itself.
    """

    __slots__ = "x", "y", "joint0", "joint1", "name"

    def __init__(self, x=0, y=0, joint0=None, joint1=None, name=None):
        """
        Create Joint.

        Arguments
        ---------
        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            Linked pivot joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other pivot joint linked. The default is None.
        """
        self.x, self.y = x, y
        if joint0 is None or isinstance(joint0, Joint):
            self.joint0 = joint0
        else:
            self.joint0 = Static(*joint0)
        if joint1 is None or isinstance(joint1, Joint):
            self.joint1 = joint1
        else:
            self.joint1 = Static(*joint1)
        self.name = name
        if name is None:
            self.name = str(id(self))

    def __repr__(self):
        """Represent object with class name, coordinates, name and state."""
        return "{}(x={}, y={}, name={})".format(
            self.__class__.__name__, self.x, self.y, self.name
        )

    def __get_joints__(self):
        """Return constraint joints as a tuple."""
        return self.joint0, self.joint1

    def coord(self):
        """Return cartesian coordinates."""
        return self.x, self.y

    def set_coord(self, *args):
        """Take a sequence or two scalars, and assign them to object x, y."""
        if len(args) == 1:
            self.x, self.y = args[0]
        else:
            self.x, self.y = args[0], args[1]

    @abc.abstractmethod
    def get_constraints(self):
        """Return geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't call a constraint from an abstract class."
        )

    @abc.abstractmethod
    def set_constraints(self, *args):
        """Set geometric constraints applying to this Joint."""
        raise NotImplementedError(
            "You can't set a constraint from an abstract class."
        )


class Static(Joint):
    """
    Special case of Joint that should not move.

    Mostly used for the frame.
    """

    __slots__ = tuple()

    def __init__(self, x=0, y=0, name=None):
        """
        A Static joint is a point in space to use as anchor by other joints.

        It is NOT a kind of joint as viewed in engineering terms!

        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        """
        super().__init__(x, y, name=name)

    def reload(self):
        """Do nothing, for consistency only."""
        pass

    def get_constraints(self):
        """Return an empty tuple."""
        return tuple()

    def set_constraints(self, *args):
        """Do nothing, for consistency only."""
        pass

    def set_anchor0(self, joint):
        """First joint anchor."""
        self.joint0 = joint

    def set_anchor1(self, joint):
        """Second joint anchor."""
        self.joint1 = joint


class Fixed(Joint):
    """Define a joint using parents locations only, with no ambiguity."""

    __slots__ = "r", "angle"

    def __init__(self, x=None, y=None, joint0=None, joint1=None,
                 distance=None, angle=None, name=None):
        """
        Create a point, of position fully defined by its two references.

        Arguments
        ---------
        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            Linked pivot joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other pivot joint linked. The default is None.
        distance : float, optional
            Distance to keep constant between joint0 and self. The default is
            None.
        angle : float, optional
         Angle (joint1, joint0, self). Should be in radian and in trigonometric
         order. The default is None.
        """
        super().__init__(x, y, joint0, joint1, name)
        self.angle = angle
        self.r = distance

    def reload(self):
        """
        Compute point coordinates.

        We know point position relative to its two parents, which gives a local
        space. The know the orientation of local space, so we can solve the
        whole. Local space is defined by link[0] as the origin and
        (link[0], link[1]) as abscisses axis.
        """
        if self.joint0 is None:
            return
        if self.joint0 is None or self.joint1 is None:
            raise HypostaticError(f'Not enough constraints for {self}')
        # Rotation angle of local space relative to global
        rot = atan2(self.joint1.y - self.joint0.y,
                    self.joint1.x - self.joint0.x)
        # Position in global space
        self.x, self.y = cyl_to_cart(self.r, self.angle + rot,
                                     self.joint0.coord())

    def get_constraints(self):
        """Return the constraining distance and angle parameters."""
        return self.r, self.angle

    def set_constraints(self, distance=None, angle=None):
        """Set geometric constraints."""
        self.r, self.angle = distance or self.r, angle or self.angle

    def set_anchor0(self, joint, distance=None, angle=None):
        """First joint anchor and characterisitcs."""
        self.joint0 = joint
        self.set_constraints(distance, angle)

    def set_anchor1(self, joint):
        """Second joint anchor."""
        self.joint1 = joint


class Pivot(Joint):
    """Center of pivot joint."""

    __slots__ = "r0", "r1"

    def __init__(self, x=0, y=0, joint0=None, joint1=None, distance0=None,
                 distance1=None, name=None):
        """
        Set point position, parents, and if it is fixed for this turn.

        Arguments
        ---------
        x : float, optional
            Position on horizontal axis. The default is 0.
        y : float, optional
            Position on vertical axis. The default is O.
        name : str, optional
            Friendly name for human readability. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            Linked pivot joint 1 (geometric constraints). The default is None.
        joint1 : Union[Joint, tuple[float]], optional
            Other pivot joint linked. The default is None.
        distance0 : float, optional
            Distance from joint0 to the current Joint. The default is None.
        distance1 : float, optional
            Distance from joint1 to the current Joint. The default is None.
        """
        super().__init__(x, y, joint0, joint1, name)
        self.r0, self.r1 = distance0, distance1

    def __get_joint_as_circle__(self, index):
        """Return reference as (ref.x, ref.y, ref.distance)."""
        if index == 0:
            return self.joint0.x, self.joint0.y, self.r0
        if index == 1:
            return self.joint1.x, self.joint1.y, self.r1
        raise ValueError(f'index should be 0 or 1, not {index}')

    def circle(self, joint):
        """
        Return first link between self and parent as a circle.

        Circle is a tuple (abscisse, ordinate, radius).
        """
        if self.joint0 is joint:
            return joint.x, joint.y, self.r0
        if self.joint1 is joint:
            return joint.x, joint.y, self.r1
        raise ValueError(f'{joint} is not in joints of {self}')

    def reload(self):
        """Compute position of pivot joint, use the two linked joints."""
        if self.joint0 is None:
            return
        # Fixed joint as reference. In links, we only keep fixed objects
        ref = tuple(x for x in self.__get_joints__() if None not in x.coord())
        if len(ref) == 0:
            # Don't modify coordinates (irrelevant)
            return
        if len(ref) == 1:
            raise Warning("Unable to set coordinates of pivot joint {}:"
                          "Only one constraint is set."
                          "Coordinates unchanged".format(self.name))
        elif len(ref) == 2:
            # Most common case, optimized here
            coco = circle_intersect(self.__get_joint_as_circle__(0),
                                    self.__get_joint_as_circle__(1))
            if coco[0] == 0:
                raise UnbuildableError(self)
            if coco[0] == 1:
                self.x, self.y = coco[1]
            elif coco[0] == 2:
                if sqr_dist(self.coord(), coco[1]
                            ) < sqr_dist(self.coord(), coco[2]):
                    self.x, self.y = coco[1]
                else:
                    self.x, self.y = coco[2]
            elif coco[0] == 3:
                raise Warning(f"Joint {self.name} has an infinite number of"
                              "solutions, position will be arbitrary")
                # We project position on circle of possible positions
                vect = ((j-i)/abs(j-i) for i, j in zip(coco[1],
                                                       self.coord()))
                self.x, self.y = [i + j * coco[1][2] for i, j in zip(coco[1],
                                  vect)]
    def get_constraints(self):
        """Return the two constraining distances of this joint."""
        return self.r0, self.r1

    def set_constraints(self, distance0=None, distance1=None):
        """Set geometric constraints."""
        self.r0, self.r1 = distance0 or self.r0, distance1 or self.r1

    def set_anchor0(self, joint, distance=None):
        """
        Set the first anchor for this Joint.

        Parameters
        ----------
        joint : Union[Joint, tuple[float]]
            The joint to use as achor.
        distance : float, optional
            Distance to keep constant from the anchor. The default is None.

        Returns
        -------
        None.

        """
        self.joint0 = joint
        self.set_constraints(distance0=distance)

    def set_anchor1(self, joint, distance=None):
        """
        Set the second anchor for this Joint.

        Parameters
        ----------
        joint : Union[Joint, tuple[float]]
            The joint to use as achor.
        distance : float, optional
            Distance to keep constant from the anchor. The default is None.

        Returns
        -------
        None.

        """
        self.joint1 = joint
        self.set_constraints(distance1=distance)


class Crank(Joint):
    """Define a crank joint."""

    __slots__ = "r", "angle"

    def __init__(self, x=None, y=None, joint0=None,
                 distance=None, angle=None, name=None):
        """
        Define a crank (circular motor).

        Parameters
        ----------
        x : float, optional
            initial horizontal position, won't be used thereafter.
            The default is None.
        y : float, optional
            initial vertical position. The default is None.
        joint0 : Union[Joint, tuple[float]], optional
            first reference joint. The default is None.
        distance : float, optional
            distance to keep between joint0 and self. The default is None.
        angle : float, optional
            It is the angle (horizontal axis, joint0, self).
            Should be in radian and in trigonometric order.
            The default is None.
        name : str, optional
            user-friendly name. The default is None.

        Returns
        -------
        None.

        """
        super().__init__(x, y, joint0, name=name)
        self.r, self.angle = distance, angle

    def reload(self, dt=1):
        """Make a step of crank."""
        if self.joint0 is None:
            return
        if None in self.joint0.coord():
            raise HypostaticError(f'{self.joint0} has None coordinates. '
                                  f'{self} cannot be calculated')
        # Rotation angle of local space relative to global
        rot = atan2(self.y - self.joint0.y, self.x - self.joint0.x)
        self.x, self.y = cyl_to_cart(self.r, rot + self.angle * dt,
                                     self.joint0.coord())

    def get_constraints(self):
        """Return the distance to the center of rotation."""
        return (self.r,)

    def set_constraints(self, distance=None, *args):
        """Set geometric constraints, only self.r is affected."""
        self.r = distance or self.r

    def set_anchor0(self, joint, distance=None):
        """First joint anchor and fixed distance."""
        self.joint0 = joint
        self.set_constraints(distance=distance)


class Linkage():
    """
    A linkage is a set of Joint objects.

    It is defined kinematicaly. Coordinates are given relative to its own
    base.
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
            Should be a subset of joints.
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
        raise NotImplementedError('Unable to determine automatic order')
        return self._solve_order

    def rebuild(self, pos=None):
        """
        Redifine linkage joints and given intial positions to joints.

        Parameters
        ----------
        pos : tuple[tuple[int]]
            Initial positions for each joint in self.joints.
            Coordinates does not need to be precise, they will allow us the best
            fitting position between all possible positions satifying
            constraints.
        """
        if not hasattr(self, '_solve_order'):
            self.__find_solving_order__()

        # Links parenting in descending order solely.
        # Parents joint do not have children.
        if pos is not None:
            # Defintition of initial coordinates
            self.set_coords(pos)

    def get_coords(self):
        """Return positions of all elements of the system."""
        return [j.coord() for j in self.joints]

    def set_coords(self, coords):
        """Set coordinatess for all joints of the linkage."""
        for joint, coord in zip(self.joints, coords):
            joint.set_coord(coord)

    def hyperstaticity(self):
        """Return the hyperstaticity degree of the linkage in 2D."""
        # TODO : test it
        # We have at least the frame
        solids = 1
        mobilities = 1
        kinematic_indetermined = 0
        for j in self.joints:
            if isinstance(j, (Static, Fixed)):
                pass
            elif isinstance(j, Crank):
                solids += 1
                kinematic_indetermined += 2
            elif isinstance(j, Pivot):
                solids += 1
                # A Pivot Joint create at least two pivots
                kinematic_indetermined += 4
                if not hasattr(j, 'joint1') or j.joint1 is None:
                    mobilities += 1
                else:
                    solids += 1
                    kinematic_indetermined += 2

        return 3 * (solids - 1) - kinematic_indetermined + mobilities

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
            Whether to force one-dimensionnal representation of constraints.
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
            If True, should be a one-dimensionnal sequence of floats.
            If False, should be a sequence of tuples of digits. Each element
            will be passed to the set_constraints method of each correspondig
            Joint.
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

        Formally it is the common denominator of all crank periods.

        Returns
        -------
        Number of iterations, with dt=1.

        """
        periods = 1
        for j in self.joints:
            if isinstance(j, Crank):
                freq = round(tau / j.angle)
                periods = periods * freq // gcd(periods, freq)
        return periods
