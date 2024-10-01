"""
Static joint definition file.
"""

from . import joint as pl_joint


class Static(pl_joint.Joint):
    """Special case of Joint that should not move.

    Mostly used for the frame.
    """

    __slots__ = tuple()

    def __init__(self, x=0, y=0, name=None):
        """
        A Static joint is a point in space to use as anchor by other joints.

        :param float x: Position on horizontal axis. The default is 0.
        :param float y: Position on vertical axis. The default is 0.
        :param name: Friendly name for human readability. The default is None.
        :type name: str | None
        """
        super().__init__(x, y, name=name)

    def reload(self):
        """Do nothing, for consistency only."""
        pass

    def get_constraints(self):
        """Return an empty tuple."""
        return tuple()

    def set_constraints(self, *args):
        """Do nothing, for consistency only.

        :param args: Unused
        """
        pass

    def set_anchor0(self, joint):
        """First joint anchor.

        :param Joint joint: Other joint to join with.
        """
        self.joint0 = joint

    def set_anchor1(self, joint):
        """Second joint anchor.

        :param Joint joint: Other joint to join with.
        """
        self.joint1 = joint
