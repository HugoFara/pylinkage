"""
DXF export for CAD/CNC applications.

This module provides DXF output for import into AutoCAD, CNC software,
and other CAD applications that support the DXF format.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .symbols import SymbolType, get_symbol_spec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any as Linkage  # accepts simulation.Linkage and Mechanism

    from .._types import Coord

# Try to import ezdxf, set to None if not available
try:
    import ezdxf as _ezdxf
except ImportError:
    _ezdxf = None  # type: ignore[assignment,unused-ignore]


# DXF layer configuration
DXF_LAYERS = {
    "LINKS": {"color": 7},  # White
    "JOINTS": {"color": 1},  # Red
    "GROUND": {"color": 8},  # Gray
    "CRANKS": {"color": 3},  # Green
}


def _check_ezdxf() -> Any:
    """Check that ezdxf is available, raise ImportError if not."""
    if _ezdxf is None:
        raise ImportError("DXF export requires ezdxf. Install with: pip install pylinkage[cad]")
    return _ezdxf


def _setup_layers(doc: Any) -> None:
    """Create DXF layers for the linkage diagram."""
    for name, props in DXF_LAYERS.items():
        doc.layers.add(name, color=props["color"])


def _draw_dxf_link(
    msp: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    layer: str = "LINKS",
) -> None:
    """Draw a link as a rounded rectangle (stadium shape).

    Uses LWPOLYLINE with bulge values to create semicircular ends.

    Args:
        msp: DXF modelspace to draw in.
        x1, y1: Start point coordinates.
        x2, y2: End point coordinates.
        width: Width of the link bar.
        layer: DXF layer name.
    """
    # Calculate perpendicular offset for width
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1e-9:
        return

    angle = math.atan2(y2 - y1, x2 - x1)
    dx = -math.sin(angle) * width / 2
    dy = math.cos(angle) * width / 2

    # Create stadium shape using LWPOLYLINE with bulge for semicircles
    # Points: top-left -> top-right (with bulge) -> bottom-right -> bottom-left (with bulge)
    # Bulge = 1.0 creates a semicircle
    points = [
        (x1 + dx, y1 + dy, 0, 0, 0),  # Top-left corner of start
        (x2 + dx, y2 + dy, 0, 0, 1.0),  # Top-right corner with bulge for end cap
        (x2 - dx, y2 - dy, 0, 0, 0),  # Bottom-right corner
        (x1 - dx, y1 - dy, 0, 0, 1.0),  # Bottom-left corner with bulge for start cap
    ]
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    # Add centerline as dashed line
    msp.add_line(
        (x1, y1),
        (x2, y2),
        dxfattribs={"layer": layer, "linetype": "DASHED"},
    )


def _draw_dxf_revolute(
    msp: Any,
    x: float,
    y: float,
    radius: float,
    layer: str = "JOINTS",
) -> None:
    """Draw a revolute (pin) joint symbol.

    Args:
        msp: DXF modelspace to draw in.
        x, y: Joint position.
        radius: Joint symbol radius.
        layer: DXF layer name.
    """
    # Outer circle
    msp.add_circle((x, y), radius, dxfattribs={"layer": layer})
    # Center point
    msp.add_point((x, y), dxfattribs={"layer": layer})


def _draw_dxf_ground(
    msp: Any,
    x: float,
    y: float,
    size: float,
    layer: str = "GROUND",
) -> None:
    """Draw a ground/fixed support symbol (triangle with hatching).

    Args:
        msp: DXF modelspace to draw in.
        x, y: Joint position.
        size: Symbol size.
        layer: DXF layer name.
    """
    # Triangle pointing down (in world coordinates, Y-up)
    triangle = [
        (x, y),
        (x - size, y - size),
        (x + size, y - size),
        (x, y),  # Close
    ]
    msp.add_lwpolyline(triangle, dxfattribs={"layer": layer})

    # Hatching lines below triangle
    hatch_y = y - size
    for i in range(4):
        offset = size * 0.25 + i * size * 0.2
        line_width = size - i * size * 0.15
        msp.add_line(
            (x - line_width, hatch_y - offset),
            (x + line_width, hatch_y - offset),
            dxfattribs={"layer": layer},
        )


def _draw_dxf_crank(
    msp: Any,
    x: float,
    y: float,
    radius: float,
    layer: str = "CRANKS",
) -> None:
    """Draw a crank/motor joint symbol (circle with rotation arc).

    Args:
        msp: DXF modelspace to draw in.
        x, y: Joint position.
        radius: Joint symbol radius.
        layer: DXF layer name.
    """
    # Outer circle
    msp.add_circle((x, y), radius, dxfattribs={"layer": layer})
    # Center dot (small circle)
    msp.add_circle((x, y), radius * 0.25, dxfattribs={"layer": layer})

    # Rotation arc
    arc_radius = radius + radius * 0.5
    msp.add_arc(
        (x, y),
        arc_radius,
        start_angle=-20,
        end_angle=70,
        dxfattribs={"layer": layer},
    )

    # Arrowhead at end of arc
    arrow_angle = math.radians(70)
    arrow_x = x + arc_radius * math.cos(arrow_angle)
    arrow_y = y + arc_radius * math.sin(arrow_angle)
    arrow_size = radius * 0.4

    # Arrow pointing in rotation direction (counterclockwise)
    tip_angle = arrow_angle + math.pi / 2
    p1 = (
        arrow_x + arrow_size * math.cos(tip_angle - 2.5),
        arrow_y + arrow_size * math.sin(tip_angle - 2.5),
    )
    p2 = (
        arrow_x + arrow_size * math.cos(tip_angle + 2.5),
        arrow_y + arrow_size * math.sin(tip_angle + 2.5),
    )
    msp.add_lwpolyline(
        [(arrow_x, arrow_y), p1, p2, (arrow_x, arrow_y)],
        dxfattribs={"layer": layer},
    )


def _draw_dxf_slider(
    msp: Any,
    x: float,
    y: float,
    size: float,
    angle: float = 0,
    layer: str = "JOINTS",
) -> None:
    """Draw a slider/prismatic joint symbol (rectangle on rails).

    Args:
        msp: DXF modelspace to draw in.
        x, y: Joint position.
        size: Symbol size.
        angle: Rotation angle in radians.
        layer: DXF layer name.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    w, h = size, size * 0.6

    # Rectangle corners (rotated)
    corners = [
        (x + cos_a * w / 2 - sin_a * h / 2, y + sin_a * w / 2 + cos_a * h / 2),
        (x - cos_a * w / 2 - sin_a * h / 2, y - sin_a * w / 2 + cos_a * h / 2),
        (x - cos_a * w / 2 + sin_a * h / 2, y - sin_a * w / 2 - cos_a * h / 2),
        (x + cos_a * w / 2 + sin_a * h / 2, y + sin_a * w / 2 - cos_a * h / 2),
    ]
    msp.add_lwpolyline(
        corners + [corners[0]],  # Close the rectangle
        dxfattribs={"layer": layer},
    )

    # Rail lines
    rail_offset = h / 2 + size * 0.3
    rail_length = size * 1.2
    for sign in [-1, 1]:
        rx = x + sign * sin_a * rail_offset
        ry = y - sign * cos_a * rail_offset
        msp.add_line(
            (rx - cos_a * rail_length, ry - sin_a * rail_length),
            (rx + cos_a * rail_length, ry + sin_a * rail_length),
            dxfattribs={"layer": layer},
        )


def plot_linkage_dxf(
    linkage: "Linkage",
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    *,
    frame_index: int = 0,
    link_width: float | None = None,
    joint_radius: float | None = None,
) -> Any:
    """Create a DXF drawing of the linkage.

    Args:
        linkage: The linkage to export.
        loci: Optional precomputed loci. If None, runs simulation.
        frame_index: Which simulation frame to export (0 = first).
        link_width: Width of link bars in world units. Auto-scaled if None.
        joint_radius: Radius of joint circles in world units. Auto-scaled if None.

    Returns:
        An ezdxf Drawing object ready to save.

    Example:
        >>> from pylinkage.visualizer import plot_linkage_dxf, save_linkage_dxf
        >>> doc = plot_linkage_dxf(linkage)
        >>> doc.saveas("linkage.dxf")
    """
    ezdxf = _check_ezdxf()

    # Run simulation if no loci provided
    loci = list(linkage.step()) if loci is None else list(loci)

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

    # Auto-scale link width and joint radius if not specified
    if link_width is None:
        link_width = bounds_size * 0.03
    if joint_radius is None:
        joint_radius = bounds_size * 0.025

    # Create DXF document
    doc = ezdxf.new(dxfversion="R2010")
    _setup_layers(doc)
    msp = doc.modelspace()

    # Ensure DASHED linetype exists
    if "DASHED" not in doc.linetypes:
        doc.linetypes.add("DASHED", pattern=[0.5, 0.25, -0.25])

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
        # For implicit Static joints (created from coordinate tuples)
        coord = joint.coord()  # type: ignore[attr-defined]
        return (coord[0] or 0.0, coord[1] or 0.0)

    from .core import is_prismatic_like, is_revolute_like

    # Draw links
    drawn_links: set[tuple[int, int]] = set()

    for joint in linkage.joints:
        pos = get_position(joint)

        # Draw link to joint0 (first parent)
        joint0 = getattr(joint, "joint0", None)
        if joint0 is not None:
            parent_pos = get_position(joint0)

            joint_ids = (id(joint), id(joint0))
            rev_ids = (id(joint0), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                _draw_dxf_link(
                    msp,
                    parent_pos[0],
                    parent_pos[1],
                    pos[0],
                    pos[1],
                    link_width,
                )
                drawn_links.add(joint_ids)

        # Draw link to joint1 (second parent) for joints that have it
        joint1 = getattr(joint, "joint1", None)
        if joint1 is not None and is_revolute_like(joint):
            parent_pos = get_position(joint1)

            joint_ids = (id(joint), id(joint1))
            rev_ids = (id(joint1), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                _draw_dxf_link(
                    msp,
                    parent_pos[0],
                    parent_pos[1],
                    pos[0],
                    pos[1],
                    link_width,
                )
                drawn_links.add(joint_ids)

        # Handle Prismatic joints - draw constraint line
        p_joint1 = getattr(joint, "joint1", None)
        p_joint2 = getattr(joint, "joint2", None)
        if is_prismatic_like(joint) and p_joint1 is not None and p_joint2 is not None:
            p1_pos = get_position(p_joint1)
            p2_pos = get_position(p_joint2)

            joint_ids = (id(p_joint1), id(p_joint2))
            rev_ids = (id(p_joint2), id(p_joint1))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                _draw_dxf_link(
                    msp,
                    p1_pos[0],
                    p1_pos[1],
                    p2_pos[0],
                    p2_pos[1],
                    link_width * 0.8,
                )
                drawn_links.add(joint_ids)

    # Draw joints (on top of links conceptually, though DXF doesn't have z-order)
    for joint in linkage.joints:
        pos = get_position(joint)
        spec = get_symbol_spec(joint)

        if spec.symbol_type == SymbolType.GROUND:
            _draw_dxf_ground(msp, pos[0], pos[1], joint_radius * 2)
        elif spec.symbol_type == SymbolType.CRANK:
            _draw_dxf_crank(msp, pos[0], pos[1], joint_radius * 1.2)
        elif spec.symbol_type == SymbolType.SLIDER:
            _draw_dxf_slider(msp, pos[0], pos[1], joint_radius * 2)
        else:  # REVOLUTE or FIXED
            _draw_dxf_revolute(msp, pos[0], pos[1], joint_radius)

    return doc


def save_linkage_dxf(
    linkage: "Linkage",
    path: str | Path,
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    **kwargs: object,
) -> None:
    """Save linkage to a DXF file.

    Args:
        linkage: The linkage to export.
        path: Output file path (should end in .dxf).
        loci: Optional precomputed loci.
        **kwargs: Additional arguments passed to plot_linkage_dxf.

    Example:
        >>> from pylinkage.visualizer import save_linkage_dxf
        >>> save_linkage_dxf(linkage, "output.dxf")
    """
    doc = plot_linkage_dxf(linkage, loci, **kwargs)  # type: ignore[arg-type]
    doc.saveas(str(path))
