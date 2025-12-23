"""Oscillating cam follower - rocker arm driven by cam profile.

An oscillating cam follower pivots about a fixed point based on
the cam's rotation angle. The cam profile determines the follower's
angular displacement as a function of the cam angle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..components import Component, ConnectedComponent, Ground, _AnchorProxy

if TYPE_CHECKING:
    from ..actuators import Crank
    from ..cam.profiles import CamProfile


class OscillatingCamFollower(ConnectedComponent):
    """Oscillating cam follower - rocker arm driven by cam profile.

    The follower pivots about a fixed point (pivot_anchor). The cam profile
    maps cam angle to follower arm angle offset. The output is at a fixed
    distance from the pivot along the arm direction.

    For oscillating followers, the profile's "displacement" is interpreted
    as an angular displacement (in radians) added to the initial_angle.

    Knife-edge vs roller is controlled by the roller_radius parameter.

    Attributes:
        cam_driver: Crank actuator driving the cam rotation.
        profile: Cam profile defining output angle offset vs input angle.
        pivot_anchor: Fixed pivot point for the rocker arm.
        arm_length: Distance from pivot to follower output.
        initial_angle: Starting angle of the arm (radians from +x).
        roller_radius: Radius of roller (0 for knife-edge).

    Example:
        >>> from pylinkage.components import Ground
        >>> from pylinkage.actuators import Crank
        >>> from pylinkage.cam import FunctionProfile, CycloidalMotionLaw
        >>> import math
        >>>
        >>> O_cam = Ground(0.0, 0.0, name="cam_center")
        >>> O_pivot = Ground(2.0, 0.0, name="pivot")
        >>> cam = Crank(anchor=O_cam, radius=0.1, angular_velocity=0.1)
        >>> # Profile: maps cam angle to arm angle offset (radians)
        >>> profile = FunctionProfile(
        ...     motion_law=CycloidalMotionLaw(),
        ...     base_radius=0.0,  # No offset at base
        ...     total_lift=math.pi/4,  # 45 degree swing
        ... )
        >>> follower = OscillatingCamFollower(
        ...     cam_driver=cam,
        ...     profile=profile,
        ...     pivot_anchor=O_pivot,
        ...     arm_length=1.5,
        ...     initial_angle=math.pi/2,  # Arm starts vertical
        ...     roller_radius=0.1,
        ... )
    """

    __slots__ = (
        "cam_driver",
        "profile",
        "pivot_anchor",
        "arm_length",
        "initial_angle",
        "roller_radius",
        "_output",
        "_cam_angle",
        "_arm_angle",
    )

    cam_driver: Crank
    profile: CamProfile
    pivot_anchor: Ground
    arm_length: float
    initial_angle: float
    roller_radius: float
    _output: _AnchorProxy
    _cam_angle: float
    _arm_angle: float

    def __init__(
        self,
        cam_driver: Crank,
        profile: CamProfile,
        pivot_anchor: Ground,
        arm_length: float,
        initial_angle: float = 0.0,
        roller_radius: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Create an oscillating cam follower.

        Args:
            cam_driver: Crank providing cam rotation angle.
            profile: Cam profile (maps cam angle to arm angle offset).
            pivot_anchor: Fixed pivot point for the rocker arm.
            arm_length: Distance from pivot to output point.
            initial_angle: Base angle of arm when cam displacement is 0.
            roller_radius: Follower roller radius (0 for knife-edge).
            name: Human-readable identifier.

        Raises:
            ValueError: If pivot anchor position is undefined.
        """
        if pivot_anchor.x is None or pivot_anchor.y is None:
            raise ValueError("Pivot anchor must have defined position")

        # Get initial cam angle from driver
        if cam_driver.x is not None and cam_driver.y is not None:
            if cam_driver.anchor.x is not None and cam_driver.anchor.y is not None:
                cam_angle = math.atan2(
                    cam_driver.y - cam_driver.anchor.y,
                    cam_driver.x - cam_driver.anchor.x,
                )
            else:
                cam_angle = 0.0
        else:
            cam_angle = 0.0

        # Compute initial arm angle from profile
        # For oscillating follower, profile returns angular offset
        angle_offset = profile.evaluate(cam_angle) - profile.base_radius
        arm_angle = initial_angle + angle_offset

        x = pivot_anchor.x + arm_length * math.cos(arm_angle)
        y = pivot_anchor.y + arm_length * math.sin(arm_angle)

        super().__init__(x, y, name)

        self.cam_driver = cam_driver
        self.profile = profile
        self.pivot_anchor = pivot_anchor
        self.arm_length = arm_length
        self.initial_angle = initial_angle
        self.roller_radius = roller_radius
        self._output = _AnchorProxy(self)
        self._cam_angle = cam_angle
        self._arm_angle = arm_angle

    @property
    def output(self) -> _AnchorProxy:
        """Return the output proxy for connecting other components.

        Returns:
            An anchor proxy representing the follower output.
        """
        return self._output

    @property
    def cam_angle(self) -> float:
        """Return the current cam angle in radians."""
        return self._cam_angle

    @property
    def arm_angle(self) -> float:
        """Return the current arm angle in radians."""
        return self._arm_angle

    @property
    def angular_displacement(self) -> float:
        """Return the current angular displacement from initial angle."""
        return self._arm_angle - self.initial_angle

    @property
    def anchors(self) -> tuple[Component, ...]:
        """Return parent components (cam driver and pivot)."""
        return (self.cam_driver, self.pivot_anchor)

    def get_constraints(self) -> tuple[float, ...]:
        """Return optimizable constraints.

        Returns arm_length, initial_angle, roller_radius, and profile constraints.
        """
        return (
            self.arm_length,
            self.initial_angle,
            self.roller_radius,
        ) + self.profile.get_constraints()

    def set_constraints(
        self,
        arm_length: float | None = None,
        initial_angle: float | None = None,
        roller_radius: float | None = None,
        *profile_constraints: float | None,
    ) -> None:
        """Set constraints from optimization.

        Args:
            arm_length: New arm length.
            initial_angle: New initial angle.
            roller_radius: New roller radius.
            *profile_constraints: Constraints passed to profile.
        """
        if arm_length is not None:
            self.arm_length = arm_length
        if initial_angle is not None:
            self.initial_angle = initial_angle
        if roller_radius is not None:
            self.roller_radius = roller_radius
        if profile_constraints:
            valid_constraints = [c for c in profile_constraints if c is not None]
            if valid_constraints:
                self.profile.set_constraints(*valid_constraints)

    def reload(self, dt: float = 1) -> None:
        """Recompute follower position from cam angle.

        Args:
            dt: Time step (unused, cam angle from driver position).

        Raises:
            ValueError: If cam driver or pivot has undefined position.
        """
        from ..solver.joints import solve_oscillating_cam_follower

        # Get cam angle from driver position
        if self.cam_driver.x is None or self.cam_driver.y is None:
            return
        if self.cam_driver.anchor.x is None or self.cam_driver.anchor.y is None:
            return

        self._cam_angle = math.atan2(
            self.cam_driver.y - self.cam_driver.anchor.y,
            self.cam_driver.x - self.cam_driver.anchor.x,
        )

        if self.pivot_anchor.x is None or self.pivot_anchor.y is None:
            return

        # Get arm angle offset from profile
        # Profile returns radius, interpret as base_radius + angular_offset
        angle_offset = self.profile.evaluate(self._cam_angle) - self.profile.base_radius
        self._arm_angle = self.initial_angle + angle_offset

        self.x, self.y = solve_oscillating_cam_follower(
            self.pivot_anchor.x,
            self.pivot_anchor.y,
            self.arm_length,
            self._arm_angle,
        )
