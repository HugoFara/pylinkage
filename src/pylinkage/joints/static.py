"""
Static joint definition file.

.. deprecated:: 0.7.0
    The `Static` class is deprecated. Use `pylinkage.mechanism.GroundJoint`
    instead for clearer terminology. Static will be removed in version 1.0.0.
"""


import warnings
from typing import TYPE_CHECKING

from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Coord


class Static(pl_joint.Joint):
    """Special case of Joint that should not move.

    Mostly used for the frame.

    .. deprecated:: 0.7.0
        Use :class:`pylinkage.mechanism.GroundJoint` instead.
        This class represents a ground joint (fixed point on the frame),
        not a generic "static" object. The new API uses clearer terminology.
    """

    __slots__: tuple[str, ...] = ()

    def __init__(
        self, x: float = 0, y: float = 0, name: str | None = None
    ) -> None:
        """A Static joint is a point in space to use as anchor by other joints.

        Args:
            x: Position on horizontal axis.
            y: Position on vertical axis.
            name: Friendly name for human readability.

        .. deprecated:: 0.7.0
            Use `pylinkage.mechanism.GroundJoint` instead.
        """
        warnings.warn(
            "Static is deprecated and will be removed in version 1.0.0. "
            "Use pylinkage.mechanism.GroundJoint instead for clearer terminology. "
            "Example: GroundJoint('name', position=(x, y))",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(x, y, name=name)

    def reload(self, dt: float = 1) -> None:
        """Do nothing, for consistency only.

        Args:
            dt: Unused, but preserves the object structure.
        """
        pass

    def get_constraints(self) -> tuple[()]:
        """Return an empty tuple."""
        return ()

    def set_constraints(self, *args: float | None) -> None:
        """Do nothing, for consistency only.

        Args:
            args: Unused.
        """
        pass

    def set_anchor0(self, joint: "pl_joint.Joint | Coord") -> None:
        """First joint anchor.

        Args:
            joint: Joint to set as anchor.
        """
        self.joint0 = pl_joint.joint_syntax_parser(joint)

    def set_anchor1(self, joint: "pl_joint.Joint | Coord") -> None:
        """Second joint anchor.

        Args:
            joint: Joint to set as anchor.
        """
        self.joint1 = pl_joint.joint_syntax_parser(joint)
