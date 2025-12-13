"""
Static joint definition file.
"""


from typing import TYPE_CHECKING

from . import joint as pl_joint

if TYPE_CHECKING:
    from .._types import Coord


class Static(pl_joint.Joint):
    """Special case of Joint that should not move.

    Mostly used for the frame.
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
        """
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
