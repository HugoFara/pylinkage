"""Conversion utilities between dyads and mechanism representations.

This module provides functions to convert between the dyads API
and the lower-level mechanism API (Joint/Link/Mechanism classes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mechanism import Mechanism
    from ..simulation import Linkage


def to_mechanism(linkage: Linkage) -> Mechanism:
    """Convert a dyads Linkage to a mechanism Mechanism.

    This creates a low-level Mechanism object from a dyads Linkage,
    allowing access to the full mechanism API (Joint/Link classes,
    step_fast(), etc.).

    Args:
        linkage: A dyads Linkage to convert.

    Returns:
        A mechanism.Mechanism object.

    Note:
        This is a one-way conversion. Changes to the returned Mechanism
        will not be reflected in the original Linkage.
    """
    from ..actuators import Crank
    from ..components import Ground, _AnchorProxy
    from ..mechanism import GroundJoint, Joint, RevoluteJoint
    from ..mechanism import Mechanism as MechMechanism
    from ..mechanism.link import DriverLink, GroundLink, Link
    from .fixed import FixedDyad
    from .rrp import RRPDyad
    from .rrr import RRRDyad

    # Ensure solve order is computed
    if not hasattr(linkage, "_solve_order"):
        linkage._find_solve_order()

    # Map dyads to mechanism joints
    dyad_to_joint: dict[int, Joint] = {}
    joints: list[Joint] = []
    links: list[Link] = []
    ground_joints: list[Joint] = []

    # First pass: create joints for all components
    for dyad in linkage.components:
        joint: Joint
        if isinstance(dyad, Ground):
            joint = GroundJoint(
                id=dyad.name,
                position=(dyad.x or 0.0, dyad.y or 0.0),
                name=dyad.name,
            )
            ground_joints.append(joint)
        else:
            joint = RevoluteJoint(
                id=dyad.name,
                position=(dyad.x, dyad.y),
                name=dyad.name,
            )
        joints.append(joint)
        dyad_to_joint[id(dyad)] = joint

    # Create ground link if we have ground joints
    if ground_joints:
        ground_link = GroundLink(
            id="ground",
            joints=ground_joints,
            name="ground",
        )
        links.append(ground_link)

    # Second pass: create links
    for dyad in linkage.components:
        joint = dyad_to_joint[id(dyad)]

        if isinstance(dyad, Crank):
            anchor_joint = dyad_to_joint[id(dyad.anchor)]
            driver = DriverLink(
                id=f"{dyad.name}_driver",
                joints=[anchor_joint, joint],
                name=f"{dyad.name}_driver",
                motor_joint=anchor_joint,  # type: ignore[arg-type]
                angular_velocity=dyad.angular_velocity,
                initial_angle=dyad.initial_angle,
            )
            links.append(driver)

        elif isinstance(dyad, RRRDyad):
            # Create two links connecting to anchors
            anchor1 = (
                dyad.anchor1._parent if isinstance(dyad.anchor1, _AnchorProxy) else dyad.anchor1
            )
            anchor2 = (
                dyad.anchor2._parent if isinstance(dyad.anchor2, _AnchorProxy) else dyad.anchor2
            )

            anchor1_joint = dyad_to_joint[id(anchor1)]
            anchor2_joint = dyad_to_joint[id(anchor2)]

            link1 = Link(
                id=f"{dyad.name}_link1",
                joints=[anchor1_joint, joint],
                name=f"{dyad.name}_link1",
            )
            link2 = Link(
                id=f"{dyad.name}_link2",
                joints=[anchor2_joint, joint],
                name=f"{dyad.name}_link2",
            )
            links.append(link1)
            links.append(link2)

        elif isinstance(dyad, RRPDyad):
            rev_anchor = (
                dyad.revolute_anchor._parent
                if isinstance(dyad.revolute_anchor, _AnchorProxy)
                else dyad.revolute_anchor
            )
            rev_joint = dyad_to_joint[id(rev_anchor)]
            l1_joint = dyad_to_joint[id(dyad.line_anchor1)]
            l2_joint = dyad_to_joint[id(dyad.line_anchor2)]

            link1 = Link(
                id=f"{dyad.name}_link1",
                joints=[rev_joint, joint],
                name=f"{dyad.name}_link1",
            )
            link2 = Link(
                id=f"{dyad.name}_guide",
                joints=[l1_joint, l2_joint, joint],
                name=f"{dyad.name}_guide",
            )
            links.append(link1)
            links.append(link2)

        elif isinstance(dyad, FixedDyad):
            anchor1 = (
                dyad.anchor1._parent if isinstance(dyad.anchor1, _AnchorProxy) else dyad.anchor1
            )
            anchor2 = (
                dyad.anchor2._parent if isinstance(dyad.anchor2, _AnchorProxy) else dyad.anchor2
            )

            anchor1_joint = dyad_to_joint[id(anchor1)]
            anchor2_joint = dyad_to_joint[id(anchor2)]

            link1 = Link(
                id=f"{dyad.name}_link1",
                joints=[anchor1_joint, joint],
                name=f"{dyad.name}_link1",
            )
            link2 = Link(
                id=f"{dyad.name}_link2",
                joints=[anchor2_joint, joint],
                name=f"{dyad.name}_link2",
            )
            links.append(link1)
            links.append(link2)

    return MechMechanism(
        joints=joints,
        links=links,
        name=linkage.name,
    )
