"""Translating cam follower - linear motion driven by cam profile.

A translating cam follower moves along a fixed axis (guide) based on
the cam's rotation angle. The cam profile determines the follower's
displacement as a function of the cam angle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..components import Component, ConnectedComponent, Ground, _AnchorProxy

if TYPE_CHECKING:
    from ..actuators import Crank
    from ..cam.profiles import CamProfile


class TranslatingCamFollower(ConnectedComponent):
    """Translating cam follower - linear motion driven by cam profile.

    The follower moves along a fixed axis (defined by guide_angle) based on
    the cam's rotation angle. The output position is determined by the cam
    profile evaluation.

    Knife-edge vs roller is controlled by the roller_radius parameter:
        - roller_radius=0: knife-edge follower (point contact)
        - roller_radius>0: roller follower (uses pitch curve)

    Attributes:
        cam_driver: Crank actuator driving the cam rotation.
        profile: Cam profile defining displacement vs angle.
        guide: Ground point defining the guide axis origin.
        guide_angle: Angle of the guide axis (radians from +x).
        roller_radius: Radius of roller (0 for knife-edge).

    Example:
        >>> from pylinkage.components import Ground
        >>> from pylinkage.actuators import Crank
        >>> from pylinkage.cam import FunctionProfile, HarmonicMotionLaw
        >>> import math
        >>>
        >>> O = Ground(0.0, 0.0, name="cam_center")
        >>> cam = Crank(anchor=O, radius=0.1, angular_velocity=0.1)
        >>> profile = FunctionProfile(
        ...     motion_law=HarmonicMotionLaw(),
        ...     base_radius=1.0,
        ...     total_lift=0.5,
        ... )
        >>> guide = Ground(0.0, 0.0, name="guide")
        >>> follower = TranslatingCamFollower(
        ...     cam_driver=cam,
        ...     profile=profile,
        ...     guide=guide,
        ...     guide_angle=math.pi/2,  # Vertical motion
        ...     roller_radius=0.1,
        ... )
    """

    __slots__ = (
        "cam_driver",
        "profile",
        "guide",
        "guide_angle",
        "roller_radius",
        "_output",
        "_cam_angle",
    )

    cam_driver: Crank
    profile: CamProfile
    guide: Ground
    guide_angle: float
    roller_radius: float
    _output: _AnchorProxy
    _cam_angle: float

    def __init__(
        self,
        cam_driver: Crank,
        profile: CamProfile,
        guide: Ground,
        guide_angle: float = 0.0,
        roller_radius: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Create a translating cam follower.

        Args:
            cam_driver: Crank providing cam rotation angle.
            profile: Cam profile (displacement vs angle function).
            guide: Ground point at the guide axis origin.
            guide_angle: Direction of follower motion (radians from +x).
            roller_radius: Follower roller radius (0 for knife-edge).
            name: Human-readable identifier.

        Raises:
            ValueError: If guide position is undefined.
        """
        if guide.x is None or guide.y is None:
            raise ValueError("Guide must have defined position")

        # Get initial cam angle from driver
        if cam_driver.x is not None and cam_driver.y is not None:
            if cam_driver.anchor.x is not None and cam_driver.anchor.y is not None:
                initial_angle = math.atan2(
                    cam_driver.y - cam_driver.anchor.y,
                    cam_driver.x - cam_driver.anchor.x,
                )
            else:
                initial_angle = 0.0
        else:
            initial_angle = 0.0

        # Compute initial position from cam profile
        if roller_radius > 0:
            displacement = profile.pitch_radius(initial_angle, roller_radius)
        else:
            displacement = profile.evaluate(initial_angle)

        x = guide.x + displacement * math.cos(guide_angle)
        y = guide.y + displacement * math.sin(guide_angle)

        super().__init__(x, y, name)

        self.cam_driver = cam_driver
        self.profile = profile
        self.guide = guide
        self.guide_angle = guide_angle
        self.roller_radius = roller_radius
        self._output = _AnchorProxy(self)
        self._cam_angle = initial_angle

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
    def displacement(self) -> float:
        """Return the current follower displacement from base position."""
        return self.profile.evaluate(self._cam_angle) - self.profile.base_radius

    @property
    def anchors(self) -> tuple[Component, ...]:
        """Return parent components (cam driver and guide)."""
        return (self.cam_driver, self.guide)

    def get_constraints(self) -> tuple[float, ...]:
        """Return optimizable constraints.

        Returns roller_radius and profile constraints.
        """
        return (self.roller_radius,) + self.profile.get_constraints()

    def set_constraints(
        self,
        roller_radius: float | None = None,
        *profile_constraints: float | None,
    ) -> None:
        """Set constraints from optimization.

        Args:
            roller_radius: New roller radius.
            *profile_constraints: Constraints passed to profile.
        """
        if roller_radius is not None:
            self.roller_radius = roller_radius
        if profile_constraints:
            # Filter out None values
            valid_constraints = [c for c in profile_constraints if c is not None]
            if valid_constraints:
                self.profile.set_constraints(*valid_constraints)

    def reload(self, dt: float = 1) -> None:
        """Recompute follower position from cam angle.

        Args:
            dt: Time step (used to get cam angle from driver).

        Raises:
            ValueError: If cam driver or guide has undefined position.
        """
        from ..solver.joints import solve_translating_cam_follower

        # Get cam angle from driver position relative to its anchor
        if self.cam_driver.x is None or self.cam_driver.y is None:
            return
        if self.cam_driver.anchor.x is None or self.cam_driver.anchor.y is None:
            return

        self._cam_angle = math.atan2(
            self.cam_driver.y - self.cam_driver.anchor.y,
            self.cam_driver.x - self.cam_driver.anchor.x,
        )

        if self.guide.x is None or self.guide.y is None:
            return

        # Get displacement from profile
        if self.roller_radius > 0:
            displacement = self.profile.pitch_radius(self._cam_angle, self.roller_radius)
        else:
            displacement = self.profile.evaluate(self._cam_angle)

        self.x, self.y = solve_translating_cam_follower(
            self.guide.x,
            self.guide.y,
            self.guide_angle,
            displacement,
        )
