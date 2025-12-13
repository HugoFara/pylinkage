#!/usr/bin/env python3
"""
drawsvg-based kinematic diagram visualization with proper ISO engineering symbols.

This demo shows how to visualize linkages using standard kinematic symbols:
- Ground/fixed joints: Triangle with hatching (ISO 3952)
- Revolute (pin) joints: Circle with center dot
- Crank/motor joints: Circle with arrow indicator
- Links: Proper engineering-style bars

Run with: uv run python docs/examples/drawsvg_kinematic_demo.py
"""
import math

import drawsvg as draw

# Scale factor: 1 unit = 80 pixels
SCALE = 80
# Canvas padding
PADDING = 100


def world_to_canvas(x: float, y: float, height: float) -> tuple[float, float]:
    """Convert world coordinates to canvas coordinates (flip Y axis)."""
    return (x * SCALE + PADDING, height - (y * SCALE + PADDING))


def draw_ground_symbol(d: draw.Drawing, x: float, y: float, size: float = 25):
    """Draw ISO ground/fixed support symbol (triangle with hatching)."""
    # Triangle
    d.append(draw.Lines(
        x, y,
        x - size, y + size,
        x + size, y + size,
        close=True,
        fill='white',
        stroke='black',
        stroke_width=2
    ))
    # Hatching lines below triangle
    hatch_y = y + size
    for i in range(4):
        offset = 6 + i * 5
        line_width = size - i * 4
        d.append(draw.Line(
            x - line_width, hatch_y + offset,
            x + line_width, hatch_y + offset,
            stroke='black',
            stroke_width=1.5
        ))


def draw_revolute_joint(d: draw.Drawing, x: float, y: float, radius: float = 12):
    """Draw ISO revolute (pin) joint symbol - circle with center dot."""
    # Outer circle
    d.append(draw.Circle(
        x, y, radius,
        fill='white',
        stroke='black',
        stroke_width=2.5
    ))
    # Center dot
    d.append(draw.Circle(
        x, y, 3,
        fill='black',
        stroke='none'
    ))


def draw_crank_joint(d: draw.Drawing, x: float, y: float, radius: float = 15, angle: float = 0):
    """Draw crank/motor joint symbol - circle with rotation arrow."""
    # Outer circle
    d.append(draw.Circle(
        x, y, radius,
        fill='white',
        stroke='#2E86AB',
        stroke_width=3
    ))
    # Center dot
    d.append(draw.Circle(
        x, y, 4,
        fill='#2E86AB',
        stroke='none'
    ))
    # Rotation arrow arc
    arrow_radius = radius + 8
    start_angle = angle - 0.4
    end_angle = angle + 1.2

    # Arc path
    start_x = x + arrow_radius * math.cos(start_angle)
    start_y = y + arrow_radius * math.sin(start_angle)
    end_x = x + arrow_radius * math.cos(end_angle)
    end_y = y + arrow_radius * math.sin(end_angle)

    path = draw.Path(
        stroke='#2E86AB',
        stroke_width=2,
        fill='none'
    )
    path.M(start_x, start_y)
    path.A(arrow_radius, arrow_radius, 0, False, True, end_x, end_y)
    d.append(path)

    # Arrowhead
    arrow_angle = end_angle + math.pi/2
    arrow_size = 8
    d.append(draw.Lines(
        end_x, end_y,
        end_x + arrow_size * math.cos(arrow_angle - 2.5),
        end_y + arrow_size * math.sin(arrow_angle - 2.5),
        end_x + arrow_size * math.cos(arrow_angle + 2.5),
        end_y + arrow_size * math.sin(arrow_angle + 2.5),
        close=True,
        fill='#2E86AB',
        stroke='none'
    ))


def draw_slider_joint(d: draw.Drawing, x: float, y: float, angle: float = 0, size: float = 20):
    """Draw slider/prismatic joint symbol - rectangle on rails."""
    # Calculate corners based on angle
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rectangle dimensions
    w, h = size, size * 0.6

    # Rectangle corners (rotated)
    corners = [
        (x + cos_a * w/2 - sin_a * h/2, y + sin_a * w/2 + cos_a * h/2),
        (x - cos_a * w/2 - sin_a * h/2, y - sin_a * w/2 + cos_a * h/2),
        (x - cos_a * w/2 + sin_a * h/2, y - sin_a * w/2 - cos_a * h/2),
        (x + cos_a * w/2 + sin_a * h/2, y + sin_a * w/2 - cos_a * h/2),
    ]

    d.append(draw.Lines(
        *[c for corner in corners for c in corner],
        close=True,
        fill='white',
        stroke='black',
        stroke_width=2
    ))

    # Rail lines
    rail_offset = h/2 + 8
    rail_length = size * 1.5
    for sign in [-1, 1]:
        rx = x + sign * sin_a * rail_offset
        ry = y - sign * cos_a * rail_offset
        d.append(draw.Line(
            rx - cos_a * rail_length, ry - sin_a * rail_length,
            rx + cos_a * rail_length, ry + sin_a * rail_length,
            stroke='black',
            stroke_width=2
        ))


def draw_link(d: draw.Drawing, x1: float, y1: float, x2: float, y2: float,
              color: str = '#555555', width: float = 12, style: str = 'bar'):
    """Draw a mechanical link between two points.

    Styles:
        'bar': Engineering-style bar with rounded ends and centerline
        'line': Simple thick line
        'bone': Dog-bone style link
    """
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    angle = math.atan2(y2-y1, x2-x1)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Perpendicular offset for bar width
    px = -sin_a * width/2
    py = cos_a * width/2

    if style == 'bar':
        # Main bar body with rounded ends
        path = draw.Path(
            fill=color,
            stroke='black',
            stroke_width=1.5,
            opacity=0.9
        )

        # Start cap (semicircle)
        path.M(x1 + px, y1 + py)
        path.A(width/2, width/2, 0, False, False, x1 - px, y1 - py)

        # Bottom edge
        path.L(x2 - px, y2 - py)

        # End cap (semicircle)
        path.A(width/2, width/2, 0, False, False, x2 + px, y2 + py)

        # Top edge back to start
        path.L(x1 + px, y1 + py)
        path.Z()
        d.append(path)

        # Centerline (dashed)
        d.append(draw.Line(
            x1, y1, x2, y2,
            stroke='white',
            stroke_width=1,
            stroke_dasharray='4,3',
            opacity=0.7
        ))

    elif style == 'bone':
        # Dog-bone style with circles at ends
        # Center bar (narrower)
        bar_width = width * 0.4
        bpx = -sin_a * bar_width/2
        bpy = cos_a * bar_width/2

        d.append(draw.Lines(
            x1 + bpx, y1 + bpy,
            x2 + bpx, y2 + bpy,
            x2 - bpx, y2 - bpy,
            x1 - bpx, y1 - bpy,
            close=True,
            fill=color,
            stroke='black',
            stroke_width=1.5
        ))

        # End circles
        for cx, cy in [(x1, y1), (x2, y2)]:
            d.append(draw.Circle(
                cx, cy, width/2,
                fill=color,
                stroke='black',
                stroke_width=1.5
            ))
    else:  # 'line'
        d.append(draw.Line(
            x1, y1, x2, y2,
            stroke=color,
            stroke_width=width,
            stroke_linecap='round'
        ))


def draw_dimension(d: draw.Drawing, x1: float, y1: float, x2: float, y2: float,
                   label: str, offset: float = 30):
    """Draw a dimension line with label (engineering drawing style)."""
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    angle = math.atan2(y2-y1, x2-x1)

    # Offset perpendicular to line
    ox = -math.sin(angle) * offset
    oy = math.cos(angle) * offset

    # Dimension line endpoints
    dx1, dy1 = x1 + ox, y1 + oy
    dx2, dy2 = x2 + ox, y2 + oy

    # Extension lines
    d.append(draw.Line(x1, y1, dx1, dy1, stroke='#666', stroke_width=0.8))
    d.append(draw.Line(x2, y2, dx2, dy2, stroke='#666', stroke_width=0.8))

    # Dimension line
    d.append(draw.Line(dx1, dy1, dx2, dy2, stroke='#666', stroke_width=0.8))

    # Arrowheads
    arrow_size = 6
    for (ax, ay), direction in [((dx1, dy1), 1), ((dx2, dy2), -1)]:
        d.append(draw.Lines(
            ax, ay,
            ax + direction * arrow_size * math.cos(angle) - arrow_size/3 * math.sin(angle),
            ay + direction * arrow_size * math.sin(angle) + arrow_size/3 * math.cos(angle),
            ax + direction * arrow_size * math.cos(angle) + arrow_size/3 * math.sin(angle),
            ay + direction * arrow_size * math.sin(angle) - arrow_size/3 * math.cos(angle),
            close=True,
            fill='#666'
        ))

    # Label
    mid_x = (dx1 + dx2) / 2
    mid_y = (dy1 + dy2) / 2
    d.append(draw.Text(
        label,
        12,
        mid_x, mid_y - 5,
        center=True,
        fill='#333',
        font_family='Arial'
    ))


def create_fourbar_diagram():
    """Create a four-bar linkage kinematic diagram with proper ISO symbols."""
    # Canvas size
    width, height = 700, 500
    d = draw.Drawing(width, height, origin=(0, 0))

    # Background
    d.append(draw.Rectangle(0, 0, width, height, fill='white'))

    # Title
    d.append(draw.Text(
        'Four-Bar Linkage - ISO Kinematic Diagram',
        18,
        width/2, 30,
        center=True,
        fill='#333',
        font_family='Arial',
        font_weight='bold'
    ))

    # Define linkage geometry (world coordinates)
    ground_a = (0, 0)
    ground_d = (4, 0)
    crank_angle = math.pi / 4
    crank_length = 1.5

    joint_b = (
        ground_a[0] + crank_length * math.cos(crank_angle),
        ground_a[1] + crank_length * math.sin(crank_angle)
    )
    joint_c = (3.2, 1.8)

    # Convert to canvas coordinates
    def w2c(x, y):
        return world_to_canvas(x, y, height)

    ga = w2c(*ground_a)
    gd = w2c(*ground_d)
    jb = w2c(*joint_b)
    jc = w2c(*joint_c)

    # Draw ground line
    d.append(draw.Line(
        ga[0] - 40, ga[1],
        gd[0] + 40, gd[1],
        stroke='#333',
        stroke_width=3
    ))

    # Draw links (order matters for layering)
    # Ground link (implied, just the baseline)

    # Crank (link 1)
    draw_link(d, ga[0], ga[1], jb[0], jb[1], color='#2E86AB', width=14, style='bar')

    # Coupler (link 2)
    draw_link(d, jb[0], jb[1], jc[0], jc[1], color='#A23B72', width=14, style='bar')

    # Rocker (link 3)
    draw_link(d, jc[0], jc[1], gd[0], gd[1], color='#F18F01', width=14, style='bar')

    # Draw joints (on top of links)
    # Ground joints with ISO symbol
    draw_ground_symbol(d, ga[0], ga[1], size=20)
    draw_ground_symbol(d, gd[0], gd[1], size=20)

    # Crank motor joint
    draw_crank_joint(d, ga[0], ga[1] - 3, radius=14, angle=-math.pi/2)

    # Revolute joints
    draw_revolute_joint(d, jb[0], jb[1], radius=10)
    draw_revolute_joint(d, jc[0], jc[1], radius=10)
    draw_revolute_joint(d, gd[0], gd[1] - 3, radius=10)

    # Labels
    labels = [
        (ga, 'A (Ground)', -35, 40),
        (gd, 'D (Ground)', 35, 40),
        (jb, 'B', 20, -15),
        (jc, 'C', 20, -15),
    ]
    for (x, y), text, ox, oy in labels:
        d.append(draw.Text(
            text, 12,
            x + ox, y + oy,
            fill='#333',
            font_family='Arial'
        ))

    # Link labels
    link_labels = [
        ((ga[0] + jb[0])/2 - 25, (ga[1] + jb[1])/2, 'Crank', '#2E86AB'),
        ((jb[0] + jc[0])/2, (jb[1] + jc[1])/2 - 20, 'Coupler', '#A23B72'),
        ((jc[0] + gd[0])/2 + 25, (jc[1] + gd[1])/2, 'Rocker', '#F18F01'),
    ]
    for x, y, text, color in link_labels:
        d.append(draw.Text(
            text, 11,
            x, y,
            fill=color,
            font_family='Arial',
            font_weight='bold'
        ))

    # Add dimension example
    draw_dimension(d, ga[0], ga[1], gd[0], gd[1], 'L₀ = 4', offset=70)

    # Legend
    legend_x, legend_y = width - 160, 80
    d.append(draw.Rectangle(
        legend_x - 10, legend_y - 20, 150, 120,
        fill='#f8f8f8',
        stroke='#ccc',
        stroke_width=1,
        rx=5
    ))
    d.append(draw.Text('Legend', 12, legend_x, legend_y, fill='#333',
                       font_family='Arial', font_weight='bold'))

    # Legend items
    ly = legend_y + 25
    draw_ground_symbol(d, legend_x + 15, ly - 8, size=10)
    d.append(draw.Text('Ground', 10, legend_x + 35, ly, fill='#333', font_family='Arial'))

    ly += 30
    draw_revolute_joint(d, legend_x + 15, ly - 4, radius=6)
    d.append(draw.Text('Revolute', 10, legend_x + 35, ly, fill='#333', font_family='Arial'))

    ly += 30
    draw_crank_joint(d, legend_x + 15, ly - 4, radius=7, angle=0)
    d.append(draw.Text('Motor/Crank', 10, legend_x + 35, ly, fill='#333', font_family='Arial'))

    return d


def create_complex_linkage():
    """Create a more complex linkage to demonstrate scalability."""
    width, height = 900, 600
    d = draw.Drawing(width, height, origin=(0, 0))

    d.append(draw.Rectangle(0, 0, width, height, fill='white'))

    d.append(draw.Text(
        'Stephenson Type II Six-Bar Linkage',
        18,
        width/2, 30,
        center=True,
        fill='#333',
        font_family='Arial',
        font_weight='bold'
    ))

    def w2c(x, y):
        return world_to_canvas(x, y, height)

    # Six-bar linkage geometry
    ground_a = (0, 0)
    ground_f = (5, 0)
    joint_b = (1.0, 1.5)
    joint_c = (2.5, 2.2)
    joint_d = (4.0, 1.8)
    joint_e = (3.0, 0.8)

    ga = w2c(*ground_a)
    gf = w2c(*ground_f)
    jb = w2c(*joint_b)
    jc = w2c(*joint_c)
    jd = w2c(*joint_d)
    je = w2c(*joint_e)

    # Ground line
    d.append(draw.Line(
        ga[0] - 40, ga[1],
        gf[0] + 40, gf[1],
        stroke='#333',
        stroke_width=3
    ))

    # Links with different styles for variety
    links = [
        (ga, jb, '#2E86AB', 'bar'),   # Link 1 (crank)
        (jb, jc, '#A23B72', 'bar'),   # Link 2
        (jc, jd, '#F18F01', 'bar'),   # Link 3
        (jd, gf, '#4ECDC4', 'bar'),   # Link 4
        (jc, je, '#95E1D3', 'bone'),  # Link 5 (ternary connection)
        (je, ga, '#DDA0DD', 'bar'),   # Link 6
    ]

    for (start, end, color, style) in links:
        draw_link(d, start[0], start[1], end[0], end[1],
                  color=color, width=12, style=style)

    # Joints
    draw_ground_symbol(d, ga[0], ga[1], size=18)
    draw_ground_symbol(d, gf[0], gf[1], size=18)

    draw_crank_joint(d, ga[0], ga[1] - 3, radius=12, angle=-math.pi/2)

    for jx, jy in [jb, jc, jd, je]:
        draw_revolute_joint(d, jx, jy, radius=9)

    draw_revolute_joint(d, gf[0], gf[1] - 3, radius=9)

    # Joint labels
    joint_labels = [
        (ga, 'A', -25, 35),
        (gf, 'F', 25, 35),
        (jb, 'B', -20, -10),
        (jc, 'C', 0, -20),
        (jd, 'D', 20, -10),
        (je, 'E', 20, 10),
    ]
    for (x, y), label, ox, oy in joint_labels:
        d.append(draw.Text(
            label, 14,
            x + ox, y + oy,
            fill='#333',
            font_family='Arial',
            font_weight='bold'
        ))

    return d


def create_with_slider():
    """Create a linkage with a slider joint."""
    width, height = 600, 450
    d = draw.Drawing(width, height, origin=(0, 0))

    d.append(draw.Rectangle(0, 0, width, height, fill='white'))

    d.append(draw.Text(
        'Slider-Crank Mechanism',
        18,
        width/2, 30,
        center=True,
        fill='#333',
        font_family='Arial',
        font_weight='bold'
    ))

    def w2c(x, y):
        return world_to_canvas(x, y, height)

    # Slider-crank geometry
    ground_a = (0, 1)
    crank_angle = math.pi / 3
    crank_length = 1.2
    connecting_rod = 3.0

    joint_b = (
        ground_a[0] + crank_length * math.cos(crank_angle),
        ground_a[1] + crank_length * math.sin(crank_angle)
    )

    # Slider position (horizontal from A)
    slider_x = joint_b[0] + math.sqrt(connecting_rod**2 - (joint_b[1] - ground_a[1])**2)
    joint_c = (slider_x, ground_a[1])

    ga = w2c(*ground_a)
    jb = w2c(*joint_b)
    jc = w2c(*joint_c)

    # Slider rail
    rail_y = jc[1]
    d.append(draw.Line(
        50, rail_y - 15,
        width - 50, rail_y - 15,
        stroke='#333',
        stroke_width=2
    ))
    d.append(draw.Line(
        50, rail_y + 15,
        width - 50, rail_y + 15,
        stroke='#333',
        stroke_width=2
    ))

    # Links
    draw_link(d, ga[0], ga[1], jb[0], jb[1], color='#2E86AB', width=14, style='bar')
    draw_link(d, jb[0], jb[1], jc[0], jc[1], color='#A23B72', width=14, style='bar')

    # Joints
    draw_ground_symbol(d, ga[0], ga[1], size=18)
    draw_crank_joint(d, ga[0], ga[1] - 3, radius=12, angle=-math.pi/2)
    draw_revolute_joint(d, jb[0], jb[1], radius=10)
    draw_slider_joint(d, jc[0], jc[1], angle=0, size=24)

    # Labels
    labels = [
        (ga, 'A', -30, 0),
        (jb, 'B', 15, -15),
        (jc, 'C (Slider)', 0, -35),
    ]
    for (x, y), text, ox, oy in labels:
        d.append(draw.Text(
            text, 12,
            x + ox, y + oy,
            fill='#333',
            font_family='Arial'
        ))

    return d


if __name__ == "__main__":
    print("Creating ISO kinematic diagrams with drawsvg...\n")

    # Four-bar linkage
    print("1. Four-bar linkage diagram...")
    fourbar = create_fourbar_diagram()
    fourbar.save_svg('kinematic_fourbar_drawsvg.svg')
    print("   -> kinematic_fourbar_drawsvg.svg")

    # Six-bar linkage
    print("2. Six-bar linkage diagram...")
    sixbar = create_complex_linkage()
    sixbar.save_svg('kinematic_sixbar_drawsvg.svg')
    print("   -> kinematic_sixbar_drawsvg.svg")

    # Slider-crank
    print("3. Slider-crank mechanism...")
    slider = create_with_slider()
    slider.save_svg('kinematic_slider_drawsvg.svg')
    print("   -> kinematic_slider_drawsvg.svg")

    print("\nDone! Open the SVG files in a browser or image viewer.")
    print("SVG files can be edited in Inkscape or imported into CAD software.")
    print("\nFor PNG export, install: uv add 'drawsvg[raster]'")
