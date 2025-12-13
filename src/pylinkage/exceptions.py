#!/usr/bin/env python3
"""
The exceptions module is a simple quick way to access the built-in exceptions.

Created on Wed Jun 16, 15:20:06 2021.

@author: HugoFara
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .linkage.linkage import Linkage


class UnbuildableError(Exception):
    """Should be raised when the constraints cannot be solved."""

    def __init__(
        self, joint: Any, message: str = 'Unable to solve constraints'
    ) -> None:
        self.joint = joint
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        """Output the problematic joint.

        Returns:
            Name of the joint that can't be solved.
        """
        return f"{self.message} on {self.joint}"


class UnderconstrainedError(Exception):
    """The linkage is under-constrained and multiple solutions may exist."""

    def __init__(
        self, linkage: "Linkage | str", message: str = 'The linkage is under-constrained!'
    ) -> None:
        self.linkage = linkage
        super().__init__(message)


# Backwards compatibility alias (deprecated)
HypostaticError = UnderconstrainedError


class NotCompletelyDefinedError(Exception):
    """The linkage definition is incomplete."""

    def __init__(
        self, joint: Any, message: str = 'The joint is not completely defined!'
    ) -> None:
        self.joint = joint
        super().__init__(message)


class OptimizationError(Exception):
    """Should be raised when the optimization process fails."""

    def __init__(
        self, message: str = 'Optimization failed'
    ) -> None:
        self.message = message
        super().__init__(message)
