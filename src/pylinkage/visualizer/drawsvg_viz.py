"""
drawsvg-based visualization for publication-quality kinematic diagrams.

This module provides SVG output with proper ISO 3952 kinematic symbols,
suitable for engineering documentation and academic publications.
"""


import math
from typing import TYPE_CHECKING, Literal

import drawsvg as draw

from ..joints import Fixed, Prismatic
from ..joints.revolute import Pivot
from .symbols import (
    LinkStyle,
    SymbolType,
    get_link_color,
    get_symbol_spec,
    is_ground_joint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .._types import Coord
    from ..linkage.linkage import Linkage

# Default scale and padding
DEFAULT_SCALE = 80  # pixels per unit
DEFAULT_PADDING = 100


def _world_to_canvas(
    x: float, y: float, height: float, scale: float, padding: float
) -> tuple[float, float]:
    """Convert world coordinates to canvas coordinates (flip Y axis)."""
    return (x * scale + padding, height - (y * scale + padding))


def _draw_ground_symbol(
    d: draw.Drawing, x: float, y: float, size: float = 20, color: str = "#333333"
) -> None:
    """Draw ISO ground/fixed support symbol (triangle with hatching)."""
    # Triangle pointing down
    d.append(
        draw.Lines(
            x,
            y,
            x - size,
            y + size,
            x + size,
            y + size,
            close=True,
            fill="white",
            stroke=color,
            stroke_width=2,
        )
    )
    # Hatching lines below triangle
    hatch_y = y + size
    for i in range(4):
        offset = 5 + i * 4
        line_width = size - i * 3
        d.append(
            draw.Line(
                x - line_width,
                hatch_y + offset,
                x + line_width,
                hatch_y + offset,
                stroke=color,
                stroke_width=1.5,
            )
        )


def _draw_revolute_joint(
    d: draw.Drawing, x: float, y: float, radius: float = 10, color: str = "#E63946"
) -> None:
    """Draw ISO revolute (pin) joint symbol - circle with center dot."""
    # Outer circle
    d.append(
        draw.Circle(x, y, radius, fill="white", stroke=color, stroke_width=2.5)
    )
    # Center dot
    d.append(draw.Circle(x, y, radius * 0.25, fill=color, stroke="none"))


def _draw_crank_joint(
    d: draw.Drawing,
    x: float,
    y: float,
    radius: float = 12,
    color: str = "#2E86AB",
    angle: float = 0,
) -> None:
    """Draw crank/motor joint symbol - circle with rotation arrow."""
    # Outer circle
    d.append(draw.Circle(x, y, radius, fill="white", stroke=color, stroke_width=3))
    # Center dot
    d.append(draw.Circle(x, y, radius * 0.3, fill=color, stroke="none"))

    # Rotation arrow arc
    arrow_radius = radius + 6
    start_angle = angle - 0.4
    end_angle = angle + 1.2

    start_x = x + arrow_radius * math.cos(start_angle)
    start_y = y + arrow_radius * math.sin(start_angle)
    end_x = x + arrow_radius * math.cos(end_angle)
    end_y = y + arrow_radius * math.sin(end_angle)

    path = draw.Path(stroke=color, stroke_width=2, fill="none")
    path.M(start_x, start_y)
    path.A(arrow_radius, arrow_radius, 0, False, True, end_x, end_y)
    d.append(path)

    # Arrowhead
    arrow_angle = end_angle + math.pi / 2
    arrow_size = 6
    d.append(
        draw.Lines(
            end_x,
            end_y,
            end_x + arrow_size * math.cos(arrow_angle - 2.5),
            end_y + arrow_size * math.sin(arrow_angle - 2.5),
            end_x + arrow_size * math.cos(arrow_angle + 2.5),
            end_y + arrow_size * math.sin(arrow_angle + 2.5),
            close=True,
            fill=color,
            stroke="none",
        )
    )


def _draw_slider_joint(
    d: draw.Drawing,
    x: float,
    y: float,
    angle: float = 0,
    size: float = 18,
    color: str = "#F18F01",
) -> None:
    """Draw slider/prismatic joint symbol - rectangle on rails."""
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

    d.append(
        draw.Lines(
            *[c for corner in corners for c in corner],
            close=True,
            fill="white",
            stroke=color,
            stroke_width=2,
        )
    )

    # Rail lines
    rail_offset = h / 2 + 6
    rail_length = size * 1.2
    for sign in [-1, 1]:
        rx = x + sign * sin_a * rail_offset
        ry = y - sign * cos_a * rail_offset
        d.append(
            draw.Line(
                rx - cos_a * rail_length,
                ry - sin_a * rail_length,
                rx + cos_a * rail_length,
                ry + sin_a * rail_length,
                stroke=color,
                stroke_width=2,
            )
        )


def _draw_link(
    d: draw.Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#555555",
    width: float = 12,
    style: LinkStyle = LinkStyle.BAR,
) -> None:
    """Draw a mechanical link between two points.

    Args:
        d: The drawing to add the link to.
        x1, y1: Start point coordinates.
        x2, y2: End point coordinates.
        color: Fill color for the link.
        width: Width of the link in pixels.
        style: Visual style (BAR, BONE, or LINE).
    """
    angle = math.atan2(y2 - y1, x2 - x1)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    px = -sin_a * width / 2
    py = cos_a * width / 2

    if style == LinkStyle.BAR:
        # Bar with rounded ends
        path = draw.Path(fill=color, stroke="black", stroke_width=1.5, opacity=0.9)

        # Start cap (semicircle)
        path.M(x1 + px, y1 + py)
        path.A(width / 2, width / 2, 0, False, False, x1 - px, y1 - py)

        # Bottom edge
        path.L(x2 - px, y2 - py)

        # End cap (semicircle)
        path.A(width / 2, width / 2, 0, False, False, x2 + px, y2 + py)

        # Top edge
        path.L(x1 + px, y1 + py)
        path.Z()
        d.append(path)

        # Centerline
        d.append(
            draw.Line(
                x1,
                y1,
                x2,
                y2,
                stroke="white",
                stroke_width=1,
                stroke_dasharray="4,3",
                opacity=0.7,
            )
        )

    elif style == LinkStyle.BONE:
        # Dog-bone style
        bar_width = width * 0.4
        bpx = -sin_a * bar_width / 2
        bpy = cos_a * bar_width / 2

        d.append(
            draw.Lines(
                x1 + bpx,
                y1 + bpy,
                x2 + bpx,
                y2 + bpy,
                x2 - bpx,
                y2 - bpy,
                x1 - bpx,
                y1 - bpy,
                close=True,
                fill=color,
                stroke="black",
                stroke_width=1.5,
            )
        )

        # End circles
        for cx, cy in [(x1, y1), (x2, y2)]:
            d.append(
                draw.Circle(cx, cy, width / 2, fill=color, stroke="black", stroke_width=1.5)
            )

    else:  # LINE
        d.append(
            draw.Line(
                x1, y1, x2, y2, stroke=color, stroke_width=width, stroke_linecap="round"
            )
        )


def _draw_dimension(
    d: draw.Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    offset: float = 25,
) -> None:
    """Draw a dimension line with label."""
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1e-6:
        return

    angle = math.atan2(y2 - y1, x2 - x1)

    ox = -math.sin(angle) * offset
    oy = math.cos(angle) * offset

    dx1, dy1 = x1 + ox, y1 + oy
    dx2, dy2 = x2 + ox, y2 + oy

    # Extension lines
    d.append(draw.Line(x1, y1, dx1, dy1, stroke="#666", stroke_width=0.8))
    d.append(draw.Line(x2, y2, dx2, dy2, stroke="#666", stroke_width=0.8))

    # Dimension line
    d.append(draw.Line(dx1, dy1, dx2, dy2, stroke="#666", stroke_width=0.8))

    # Arrowheads
    arrow_size = 5
    for (ax, ay), direction in [((dx1, dy1), 1), ((dx2, dy2), -1)]:
        d.append(
            draw.Lines(
                ax,
                ay,
                ax + direction * arrow_size * math.cos(angle) - arrow_size / 3 * math.sin(angle),
                ay + direction * arrow_size * math.sin(angle) + arrow_size / 3 * math.cos(angle),
                ax + direction * arrow_size * math.cos(angle) + arrow_size / 3 * math.sin(angle),
                ay + direction * arrow_size * math.sin(angle) - arrow_size / 3 * math.cos(angle),
                close=True,
                fill="#666",
            )
        )

    # Label
    mid_x = (dx1 + dx2) / 2
    mid_y = (dy1 + dy2) / 2
    d.append(
        draw.Text(
            label, 10, mid_x, mid_y - 4, center=True, fill="#333", font_family="Arial"
        )
    )


def plot_linkage_svg(
    linkage: "Linkage",
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    *,
    title: str | None = None,
    show_dimensions: bool = False,
    show_loci: bool = True,
    show_labels: bool = True,
    link_style: Literal["bar", "bone", "line"] = "bar",
    scale: float = DEFAULT_SCALE,
    padding: float = DEFAULT_PADDING,
) -> draw.Drawing:
    """Create a publication-quality SVG kinematic diagram.

    Args:
        linkage: The linkage to visualize.
        loci: Optional precomputed loci. If None, runs simulation.
        title: Optional title for the diagram.
        show_dimensions: Whether to show dimension lines.
        show_loci: Whether to show joint movement paths.
        show_labels: Whether to show joint labels.
        link_style: Visual style for links ('bar', 'bone', or 'line').
        scale: Pixels per unit.
        padding: Canvas padding in pixels.

    Returns:
        A drawsvg.Drawing object.
    """
    # Convert string style to enum
    style_map = {"bar": LinkStyle.BAR, "bone": LinkStyle.BONE, "line": LinkStyle.LINE}
    link_style_enum = style_map.get(link_style, LinkStyle.BAR)

    # Run simulation if no loci provided
    if loci is None:
        loci = list(linkage.step())  # type: ignore[arg-type]
    else:
        loci = list(loci)

    if not loci:
        raise ValueError("No loci data available. Run linkage.step() first.")

    # Calculate bounding box
    all_coords = [coord for frame in loci for coord in frame]
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add padding
    margin = 0.3
    width_world = max_x - min_x + 2 * margin
    height_world = max_y - min_y + 2 * margin

    canvas_width = int(width_world * scale + 2 * padding)
    canvas_height = int(height_world * scale + 2 * padding)

    d = draw.Drawing(canvas_width, canvas_height, origin=(0, 0))
    d.append(draw.Rectangle(0, 0, canvas_width, canvas_height, fill="white"))

    def w2c(x: float, y: float) -> tuple[float, float]:
        """Convert world to canvas coordinates."""
        cx = (x - min_x + margin) * scale + padding
        cy = canvas_height - ((y - min_y + margin) * scale + padding)
        return (cx, cy)

    # Title
    if title or linkage.name:
        d.append(
            draw.Text(
                title or linkage.name,
                16,
                canvas_width / 2,
                25,
                center=True,
                fill="#333",
                font_family="Arial",
                font_weight="bold",
            )
        )

    # Draw loci (movement paths) first (behind everything)
    if show_loci and len(loci) > 1:
        for i, joint in enumerate(linkage.joints):
            path_coords = [w2c(frame[i][0], frame[i][1]) for frame in loci]
            if len(path_coords) > 1:
                spec = get_symbol_spec(joint)
                path = draw.Path(
                    stroke=spec.color,
                    stroke_width=1,
                    fill="none",
                    stroke_dasharray="3,2",
                    opacity=0.5,
                )
                path.M(path_coords[0][0], path_coords[0][1])
                for px, py in path_coords[1:]:
                    path.L(px, py)
                d.append(path)

    # Get current joint positions (use first frame for static diagram)
    current_positions: dict[object, tuple[float, float]] = {
        joint: (loci[0][i][0] or 0.0, loci[0][i][1] or 0.0)
        for i, joint in enumerate(linkage.joints)
    }

    def get_position(joint: object) -> tuple[float, float]:
        """Get position of a joint, handling implicit Static joints."""
        if joint in current_positions:
            return current_positions[joint]
        # For implicit Static joints (created from coordinate tuples)
        coord = joint.coord()  # type: ignore[attr-defined]
        return (coord[0] or 0.0, coord[1] or 0.0)

    # Draw ground line if there are ground joints
    ground_joints = [j for j in linkage.joints if is_ground_joint(j)]
    if ground_joints:
        ground_y_canvas = max(w2c(0, get_position(j)[1])[1] for j in ground_joints)
        x_positions = [w2c(get_position(j)[0], 0)[0] for j in ground_joints]
        d.append(
            draw.Line(
                min(x_positions) - 30,
                ground_y_canvas,
                max(x_positions) + 30,
                ground_y_canvas,
                stroke="#333",
                stroke_width=2.5,
            )
        )

    # Draw links
    link_index = 0
    drawn_links: set[tuple[int, int]] = set()

    for joint in linkage.joints:
        pos = get_position(joint)
        cx, cy = w2c(pos[0], pos[1])

        # Draw link to joint0 (first parent)
        if joint.joint0 is not None:
            parent_pos = get_position(joint.joint0)
            px, py = w2c(parent_pos[0], parent_pos[1])

            joint_ids = (id(joint), id(joint.joint0))
            rev_ids = (id(joint.joint0), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                color = get_link_color(link_index)
                _draw_link(d, px, py, cx, cy, color=color, width=12, style=link_style_enum)
                drawn_links.add(joint_ids)
                link_index += 1

                # Dimension line
                if show_dimensions:
                    length = math.sqrt((pos[0] - parent_pos[0]) ** 2 + (pos[1] - parent_pos[1]) ** 2)
                    _draw_dimension(d, px, py, cx, cy, f"{length:.2f}", offset=25)

        # Draw link to joint1 (second parent) for joints that have it
        if hasattr(joint, "joint1") and joint.joint1 is not None:
            if isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute":
                parent_pos = get_position(joint.joint1)
                px, py = w2c(parent_pos[0], parent_pos[1])

                joint_ids = (id(joint), id(joint.joint1))
                rev_ids = (id(joint.joint1), id(joint))
                if joint_ids not in drawn_links and rev_ids not in drawn_links:
                    color = get_link_color(link_index)
                    _draw_link(d, px, py, cx, cy, color=color, width=12, style=link_style_enum)
                    drawn_links.add(joint_ids)
                    link_index += 1

                    if show_dimensions:
                        length = math.sqrt(
                            (pos[0] - parent_pos[0]) ** 2 + (pos[1] - parent_pos[1]) ** 2
                        )
                        _draw_dimension(d, px, py, cx, cy, f"{length:.2f}", offset=25)

        # Handle Prismatic joints differently
        if isinstance(joint, Prismatic) and joint.joint1 is not None and joint.joint2 is not None:
            # Draw the constraint line between joint1 and joint2
            p1_pos = get_position(joint.joint1)
            p2_pos = get_position(joint.joint2)
            p1x, p1y = w2c(p1_pos[0], p1_pos[1])
            p2x, p2y = w2c(p2_pos[0], p2_pos[1])

            joint_ids = (id(joint.joint1), id(joint.joint2))
            rev_ids = (id(joint.joint2), id(joint.joint1))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                color = get_link_color(link_index)
                _draw_link(d, p1x, p1y, p2x, p2y, color=color, width=10, style=link_style_enum)
                drawn_links.add(joint_ids)
                link_index += 1

    # Draw joints (on top of links)
    for joint in linkage.joints:
        pos = get_position(joint)
        cx, cy = w2c(pos[0], pos[1])
        spec = get_symbol_spec(joint)

        if spec.symbol_type == SymbolType.GROUND:
            _draw_ground_symbol(d, cx, cy, size=18 * spec.size, color=spec.color)
        elif spec.symbol_type == SymbolType.CRANK:
            _draw_crank_joint(d, cx, cy, radius=12 * spec.size, color=spec.color)
        elif spec.symbol_type == SymbolType.SLIDER:
            _draw_slider_joint(d, cx, cy, size=16 * spec.size, color=spec.color)
        else:  # REVOLUTE or FIXED
            _draw_revolute_joint(d, cx, cy, radius=9 * spec.size, color=spec.color)

        # Joint label
        if show_labels and joint.name:
            label_x = cx + spec.label_offset[0] * 15
            label_y = cy + spec.label_offset[1] * 15
            d.append(
                draw.Text(
                    joint.name,
                    11,
                    label_x,
                    label_y,
                    fill="#333",
                    font_family="Arial",
                    font_weight="bold",
                )
            )

    return d


def save_linkage_svg(
    linkage: "Linkage",
    path: str,
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    **kwargs: object,
) -> None:
    """Save a linkage diagram to an SVG file.

    Args:
        linkage: The linkage to visualize.
        path: Output file path (should end in .svg).
        loci: Optional precomputed loci.
        **kwargs: Additional arguments passed to plot_linkage_svg.
    """
    drawing = plot_linkage_svg(linkage, loci, **kwargs)  # type: ignore[arg-type]
    drawing.save_svg(path)
