"""
STEP export for 3D CAD interchange.

This module provides STEP output for import into 3D CAD applications
like FreeCAD, SolidWorks, Fusion 360, and other STEP-compatible software.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .symbols import SymbolType, get_symbol_spec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .._types import Coord
    from ..linkage.linkage import Linkage

# Try to import build123d, set to None if not available
try:
    import build123d as _bd
except ImportError:
    _bd = None


@dataclass
class LinkProfile:
    """Cross-section profile for 3D links.

    Attributes:
        width: Width of the link bar in world units.
        thickness: Thickness (z-direction) in world units.
        fillet_radius: Radius for edge rounding (0 for sharp edges).
    """

    width: float
    thickness: float
    fillet_radius: float = 0.0


@dataclass
class JointProfile:
    """Profile for 3D joint geometry.

    Attributes:
        radius: Pin/hole radius in world units.
        length: Pin length in z-direction.
    """

    radius: float
    length: float


def _check_build123d() -> Any:
    """Check that build123d is available, raise ImportError if not."""
    if _bd is None:
        raise ImportError(
            "STEP export requires build123d. Install with: pip install pylinkage[cad]"
        )
    return _bd


def _auto_scale_profiles(
    bounds_size: float,
) -> tuple[LinkProfile, JointProfile]:
    """Compute auto-scaled profiles based on linkage bounding box.

    Args:
        bounds_size: Maximum dimension of the linkage bounding box.

    Returns:
        Tuple of (LinkProfile, JointProfile) with auto-scaled dimensions.
    """
    # Link: 10% of bounds for width, 20% of width for thickness
    link_width = bounds_size * 0.10
    link_thickness = link_width * 0.20

    # Joint: 30% of link width for pin radius
    joint_radius = link_width * 0.30
    joint_length = link_thickness * 1.5  # Extend slightly beyond link

    return (
        LinkProfile(
            width=link_width,
            thickness=link_thickness,
            fillet_radius=0.0,
        ),
        JointProfile(
            radius=joint_radius,
            length=joint_length,
        ),
    )


def _create_link_bar(
    bd: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    profile: LinkProfile,
    z_offset: float = 0.0,
    hole_radius: float | None = None,
) -> Any:
    """Create a 3D link bar between two points.

    Creates a stadium-shaped bar (rectangle with semicircular ends)
    extruded in the z-direction, with optional holes at the ends.

    Args:
        bd: build123d module.
        x1, y1: Start point coordinates.
        x2, y2: End point coordinates.
        profile: Link cross-section profile.
        z_offset: Z position for the link.
        hole_radius: Radius for holes at link ends. None for no holes.

    Returns:
        build123d Part object representing the link.
    """
    # Calculate link geometry
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1e-9:
        return None

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Create stadium profile (slot shape)
    # Note: build123d requires nested context managers for its builder pattern
    with bd.BuildPart() as part:
        with bd.BuildSketch(bd.Plane.XY.offset(z_offset)):  # noqa: SIM117
            with bd.Locations((mid_x, mid_y)):
                # Use SlotOverall for stadium shape
                bd.SlotOverall(length, profile.width, rotation=angle)

        # Extrude
        bd.extrude(amount=profile.thickness)

        # Add holes at both ends if requested
        if hole_radius is not None and hole_radius > 0:
            with bd.BuildSketch(bd.Plane.XY.offset(z_offset + profile.thickness)):  # noqa: SIM117
                with bd.Locations((x1, y1), (x2, y2)):
                    bd.Circle(hole_radius)
            bd.extrude(amount=-profile.thickness, mode=bd.Mode.SUBTRACT)

    return part.part


def _create_joint_pin(
    bd: Any,
    x: float,
    y: float,
    z_offset: float,
    profile: JointProfile,
) -> Any:
    """Create a cylindrical pin at joint location.

    Args:
        bd: build123d module.
        x, y: Joint position in XY plane.
        z_offset: Base Z position.
        profile: Joint profile with radius and length.

    Returns:
        build123d Part object representing the pin.
    """
    # Note: build123d requires nested context managers for its builder pattern
    with bd.BuildPart() as part:
        with bd.BuildSketch(bd.Plane.XY.offset(z_offset)):  # noqa: SIM117
            with bd.Locations((x, y)):
                bd.Circle(profile.radius)
        bd.extrude(amount=profile.length)

    return part.part


def _create_ground_symbol(
    bd: Any,
    x: float,
    y: float,
    z_offset: float,
    size: float,
    thickness: float,
) -> Any:
    """Create a 3D ground/fixed support symbol.

    Creates a triangular base with hatching extruded in Z.

    Args:
        bd: build123d module.
        x, y: Joint position.
        z_offset: Base Z position.
        size: Symbol size.
        thickness: Extrusion thickness.

    Returns:
        build123d Part object representing the ground symbol.
    """
    # Note: build123d requires nested context managers for its builder pattern
    with bd.BuildPart() as part:
        with bd.BuildSketch(bd.Plane.XY.offset(z_offset)):  # noqa: SIM117
            # Triangle
            with bd.Locations((x, y)):
                bd.RegularPolygon(size, 3, rotation=180)
        bd.extrude(amount=thickness)

    return part.part


def build_linkage_3d(
    linkage: "Linkage",
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    *,
    frame_index: int = 0,
    link_profile: LinkProfile | None = None,
    joint_profile: JointProfile | None = None,
    z_offset: float = 0.0,
    include_pins: bool = True,
) -> Any:
    """Build a 3D CAD model of the linkage.

    Creates a build123d Compound containing all links and joint pins
    as separate solids for the specified frame position.

    Args:
        linkage: The linkage to model.
        loci: Optional precomputed loci. If None, runs simulation.
        frame_index: Which simulation frame to export (0 = first).
        link_profile: Cross-section dimensions for links. Auto-scaled if None.
        joint_profile: Dimensions for joint pins/holes. Auto-scaled if None.
        z_offset: Base Z position for the linkage.
        include_pins: Whether to model joint pins as separate parts.

    Returns:
        A build123d Compound with all parts.

    Example:
        >>> from pylinkage.visualizer import build_linkage_3d, save_linkage_step
        >>> model = build_linkage_3d(linkage)
        >>> save_linkage_step(linkage, "linkage.step")
    """
    bd = _check_build123d()

    # Run simulation if no loci provided
    loci = list(linkage.step()) if loci is None else list(loci)  # type: ignore[arg-type]

    if not loci:
        raise ValueError("No loci data available. Run linkage.step() first.")

    if frame_index >= len(loci):
        raise ValueError(f"frame_index {frame_index} out of range (max {len(loci) - 1})")

    # Calculate bounding box for auto-scaling
    all_coords = [coord for frame in loci for coord in frame]
    xs = [c[0] for c in all_coords if c[0] is not None]
    ys = [c[1] for c in all_coords if c[1] is not None]

    if not xs or not ys:
        raise ValueError("No valid coordinates found in loci data.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bounds_size = max(max_x - min_x, max_y - min_y)

    # Auto-scale profiles if not provided
    if link_profile is None or joint_profile is None:
        auto_link, auto_joint = _auto_scale_profiles(bounds_size)
        if link_profile is None:
            link_profile = auto_link
        if joint_profile is None:
            joint_profile = auto_joint

    # Get positions for the specified frame
    frame_positions = loci[frame_index]
    current_positions: dict[object, tuple[float, float]] = {
        joint: (frame_positions[i][0] or 0.0, frame_positions[i][1] or 0.0)
        for i, joint in enumerate(linkage.joints)
    }

    def get_position(joint: object) -> tuple[float, float]:
        """Get position of a joint, handling implicit Static joints."""
        if joint in current_positions:
            return current_positions[joint]
        coord = joint.coord()  # type: ignore[attr-defined]
        return (coord[0] or 0.0, coord[1] or 0.0)

    # Import joint types for isinstance checks
    from ..joints import Fixed, Prismatic
    from ..joints.revolute import Pivot

    # Collect all parts
    parts: list[Any] = []
    link_index = 0
    drawn_links: set[tuple[int, int]] = set()

    # Build links
    for joint in linkage.joints:
        pos = get_position(joint)

        # Link to joint0
        if joint.joint0 is not None:
            parent_pos = get_position(joint.joint0)

            joint_ids = (id(joint), id(joint.joint0))
            rev_ids = (id(joint.joint0), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                # Calculate z-offset for this link to avoid intersection
                link_z = z_offset + link_index * link_profile.thickness * 0.1

                link_part = _create_link_bar(
                    bd,
                    parent_pos[0],
                    parent_pos[1],
                    pos[0],
                    pos[1],
                    link_profile,
                    z_offset=link_z,
                    hole_radius=joint_profile.radius if include_pins else None,
                )
                if link_part is not None:
                    parts.append(link_part)
                    drawn_links.add(joint_ids)
                    link_index += 1

        # Link to joint1 for applicable joint types
        has_joint1 = hasattr(joint, "joint1") and joint.joint1 is not None
        is_revolute_type = isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute"
        if has_joint1 and is_revolute_type:
            parent_pos = get_position(joint.joint1)

            joint_ids = (id(joint), id(joint.joint1))
            rev_ids = (id(joint.joint1), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                link_z = z_offset + link_index * link_profile.thickness * 0.1

                link_part = _create_link_bar(
                    bd,
                    parent_pos[0],
                    parent_pos[1],
                    pos[0],
                    pos[1],
                    link_profile,
                    z_offset=link_z,
                    hole_radius=joint_profile.radius if include_pins else None,
                )
                if link_part is not None:
                    parts.append(link_part)
                    drawn_links.add(joint_ids)
                    link_index += 1

        # Prismatic constraint link
        if isinstance(joint, Prismatic) and joint.joint1 is not None and joint.joint2 is not None:
            p1_pos = get_position(joint.joint1)
            p2_pos = get_position(joint.joint2)

            joint_ids = (id(joint.joint1), id(joint.joint2))
            rev_ids = (id(joint.joint2), id(joint.joint1))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                link_z = z_offset + link_index * link_profile.thickness * 0.1

                # Slider rail is thinner
                rail_profile = LinkProfile(
                    width=link_profile.width * 0.8,
                    thickness=link_profile.thickness,
                    fillet_radius=link_profile.fillet_radius,
                )
                link_part = _create_link_bar(
                    bd,
                    p1_pos[0],
                    p1_pos[1],
                    p2_pos[0],
                    p2_pos[1],
                    rail_profile,
                    z_offset=link_z,
                    hole_radius=None,
                )
                if link_part is not None:
                    parts.append(link_part)
                    drawn_links.add(joint_ids)
                    link_index += 1

    # Build joint pins
    if include_pins:
        for joint in linkage.joints:
            pos = get_position(joint)
            spec = get_symbol_spec(joint)

            # Create pin at joint location
            pin = _create_joint_pin(
                bd,
                pos[0],
                pos[1],
                z_offset - joint_profile.length * 0.25,  # Center pin on link stack
                joint_profile,
            )
            parts.append(pin)

            # Add ground symbol for ground joints
            if spec.symbol_type == SymbolType.GROUND:
                ground = _create_ground_symbol(
                    bd,
                    pos[0],
                    pos[1] - link_profile.width,  # Below joint
                    z_offset,
                    link_profile.width * 0.8,
                    link_profile.thickness,
                )
                parts.append(ground)

    # Combine all parts into a compound
    if not parts:
        raise ValueError("No geometry created. Check linkage configuration.")

    compound = bd.Compound(children=parts)
    return compound


def save_linkage_step(
    linkage: "Linkage",
    path: str | Path,
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    **kwargs: object,
) -> None:
    """Save linkage to a STEP file.

    Args:
        linkage: The linkage to export.
        path: Output file path (should end in .step or .stp).
        loci: Optional precomputed loci.
        **kwargs: Additional arguments passed to build_linkage_3d.

    Example:
        >>> from pylinkage.visualizer import save_linkage_step
        >>> save_linkage_step(linkage, "output.step")
    """
    from build123d import export_step

    model = build_linkage_3d(linkage, loci, **kwargs)  # type: ignore[arg-type]
    export_step(model, str(path))
