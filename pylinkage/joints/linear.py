"""
Definition of a linear joint.
"""
from . import joint as pl_joint
from .. import exceptions as pl_exceptions
from .. import geometry as geom


class Linear(pl_joint.Joint):
    """Define a point constrained by a prismatic joint and a revolute joint."""

    __slots__ = "revolute_radius", "joint2"

    def __init__(
        self,
        x=0,
        y=0,
        joint0=None,
        joint1=None,
        joint2=None,
        revolute_radius=None,
        name=None
    ):
        """
        Set point position, parents, and if it is fixed for this turn.

        :param x: Position on horizontal axis. The default is 0.
        :type x: float | None
        :param y: Position on vertical axis. The default is 0.
        :type y: float | None
        :param joint0: Linked revolute joint 1 (geometric constraints). The default is None.
        :type joint0: Joint | tuple[float, float]
        :param joint1: First joint or point defining the axis. The default is None.
        :type joint1: Joint | tuple[float, float]
        :param joint2: Second joint or point defining the axis. The default is None.
        :type joint2: Joint | tuple[float, float]
        :param revolute_radius: Distance from joint0 to the current Joint. The default is None.
        :type revolute_radius: float | None
        :param name: Friendly name for human readability. The default is None.
        :type name: str | None
        """
        super().__init__(x, y, joint0, joint1, name)
        self.revolute_radius = revolute_radius
        self.joint2 = pl_joint.joint_syntax_parser(joint2)

    def reload(self):
        """Compute position of revolute joint, with the three linked joints."""
        if self.joint0 is None:
            raise pl_exceptions.NotCompletelyDefinedError(self, "joint0 is not defined")
        positions_circle = (*self.joint0.coord(), self.revolute_radius)
        if None in positions_circle:
            raise pl_exceptions.UnbuildableError(
                self,
                message="Joint has missing constraints. "
                        "Current constraints are " + str(positions_circle)
            )
        positions = geom.circle_line_from_points_intersection(
            positions_circle,
            self.joint1.coord(),
            self.joint2.coord()
        )
        if len(positions) == 0:
            raise pl_exceptions.UnbuildableError(self)
        elif len(positions) == 1:
            self.x, self.y = positions[0]
        else:
            # Got to the nearest point
            self.x, self.y = geom.get_nearest_point(self.coord(), positions[0], positions[1])

    def get_constraints(self):
        """Return the only distance constraint for this joint."""
        return tuple([self.revolute_radius])

    def set_constraints(self, distance0=None):
        """Set the only distance constraint for this joint."""
        self.revolute_radius = distance0 or self.revolute_radius
