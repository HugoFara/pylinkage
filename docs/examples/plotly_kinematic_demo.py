#!/usr/bin/env python3
"""
Plotly-based kinematic diagram visualization with proper engineering symbols.

This demo shows how to visualize linkages using standard kinematic symbols:
- Ground/fixed joints: Triangle with hatching
- Revolute (pin) joints: Circle
- Crank joints: Circle with rotation indicator
- Links: Styled lines connecting joints

Run with: uv run python docs/examples/plotly_kinematic_demo.py
"""
import numpy as np
import plotly.graph_objects as go

# Define SVG paths for kinematic symbols (centered at origin, scaled)
# These follow ISO 3952 / engineering drawing conventions

# Ground symbol: triangle pointing down with base hatching
GROUND_SVG = "M -10 -5 L 10 -5 L 0 10 Z M -12 -7 L 12 -7 M -10 -9 L 10 -9 M -8 -11 L 8 -11"

# Revolute (pin) joint: simple circle
REVOLUTE_SVG = "M 0 -8 A 8 8 0 1 1 0 8 A 8 8 0 1 1 0 -8 Z"

# Crank symbol: circle with a small filled center dot
CRANK_SVG = "M 0 -10 A 10 10 0 1 1 0 10 A 10 10 0 1 1 0 -10 Z M 0 -3 A 3 3 0 1 1 0 3 A 3 3 0 1 1 0 -3 Z"

# Slider symbol: rectangle
SLIDER_SVG = "M -8 -5 L 8 -5 L 8 5 L -8 5 Z"


def create_kinematic_diagram():
    """Create a four-bar linkage visualization with proper kinematic symbols."""
    fig = go.Figure()

    # Define the four-bar linkage geometry
    # Ground points (fixed)
    ground_a = (0, 0)
    ground_d = (4, 0)

    # Moving joints (example position)
    crank_angle = np.pi / 4  # 45 degrees
    crank_length = 1.5
    joint_b = (
        ground_a[0] + crank_length * np.cos(crank_angle),
        ground_a[1] + crank_length * np.sin(crank_angle)
    )

    # Coupler and rocker lengths
    coupler_length = 3.5
    rocker_length = 2.0

    # Calculate joint C position (intersection of two circles)
    # Simplified: using a specific valid position
    joint_c = (3.2, 1.8)

    # Define link connections
    links = [
        (ground_a, joint_b, "Crank", "#2E86AB"),
        (joint_b, joint_c, "Coupler", "#A23B72"),
        (joint_c, ground_d, "Rocker", "#F18F01"),
        (ground_a, ground_d, "Ground", "#6B6B6B"),
    ]

    # Draw links as lines with varying thickness for visual hierarchy
    for start, end, name, color in links:
        # Main link line
        fig.add_trace(go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode='lines',
            line=dict(
                color=color,
                width=6 if name != "Ground" else 3,
            ),
            name=name,
            hoverinfo='name',
        ))

        # Add link centerline for engineering look
        if name != "Ground":
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=name,
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="white",
                opacity=0.8
            )

    # Draw joints with custom SVG markers
    joints_data = [
        (*ground_a, "Ground A", GROUND_SVG, "#333333", 20),
        (*ground_d, "Ground D", GROUND_SVG, "#333333", 20),
        (*joint_b, "Pin B (Crank)", CRANK_SVG, "#2E86AB", 18),
        (*joint_c, "Pin C (Revolute)", REVOLUTE_SVG, "#E63946", 16),
    ]

    for x, y, name, svg_path, color, size in joints_data:
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=size,
                color='white',
                line=dict(color=color, width=2)
            ),
            name=name,
            hovertemplate=f"<b>{name}</b><br>x: {x:.2f}<br>y: {y:.2f}<extra></extra>",
        ))

    # Add proper ground hatching as separate traces
    for gx, gy in [ground_a, ground_d]:
        # Draw hatching lines below ground symbol
        for i in range(3):
            offset = -0.15 - i * 0.1
            fig.add_trace(go.Scatter(
                x=[gx - 0.2 + i * 0.05, gx + 0.2 - i * 0.05],
                y=[gy + offset, gy + offset],
                mode='lines',
                line=dict(color='#333333', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Configure layout for engineering diagram look
    fig.update_layout(
        title=dict(
            text="Four-Bar Linkage - Kinematic Diagram",
            font=dict(size=20, family="Arial")
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#CCCCCC',
            title="X (units)"
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#CCCCCC',
            title="Y (units)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=600,
        margin=dict(r=150)
    )

    return fig


def create_animated_linkage():
    """Create an animated four-bar linkage with proper symbols."""
    # Linkage parameters
    ground_a = np.array([0, 0])
    ground_d = np.array([4, 0])
    crank_length = 1.5
    coupler_length = 3.5
    rocker_length = 2.0

    # Generate frames for animation
    n_frames = 60
    angles = np.linspace(0, 2 * np.pi, n_frames)

    frames = []
    all_loci_b = []
    all_loci_c = []

    for i, angle in enumerate(angles):
        # Crank position
        joint_b = ground_a + crank_length * np.array([np.cos(angle), np.sin(angle)])

        # Solve for joint C (circle-circle intersection)
        # Distance from D to B
        db = np.linalg.norm(joint_b - ground_d)

        # Check if linkage is buildable
        if db > coupler_length + rocker_length or db < abs(coupler_length - rocker_length):
            continue

        # Circle intersection math
        a = coupler_length
        b = rocker_length
        c = db

        # Cosine rule to find angle at D
        cos_angle_d = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_angle_d = np.clip(cos_angle_d, -1, 1)
        angle_d = np.arccos(cos_angle_d)

        # Angle of line D->B
        angle_db = np.arctan2(joint_b[1] - ground_d[1], joint_b[0] - ground_d[0])

        # Joint C position (taking one solution)
        joint_c = ground_d + rocker_length * np.array([
            np.cos(angle_db + angle_d),
            np.sin(angle_db + angle_d)
        ])

        all_loci_b.append(joint_b.tolist())
        all_loci_c.append(joint_c.tolist())

        # Create frame data
        frame_data = [
            # Crank link
            go.Scatter(
                x=[ground_a[0], joint_b[0]],
                y=[ground_a[1], joint_b[1]],
                mode='lines',
                line=dict(color='#2E86AB', width=8),
                name='Crank'
            ),
            # Coupler link
            go.Scatter(
                x=[joint_b[0], joint_c[0]],
                y=[joint_b[1], joint_c[1]],
                mode='lines',
                line=dict(color='#A23B72', width=8),
                name='Coupler'
            ),
            # Rocker link
            go.Scatter(
                x=[joint_c[0], ground_d[0]],
                y=[joint_c[1], ground_d[1]],
                mode='lines',
                line=dict(color='#F18F01', width=8),
                name='Rocker'
            ),
            # Moving joints
            go.Scatter(
                x=[joint_b[0], joint_c[0]],
                y=[joint_b[1], joint_c[1]],
                mode='markers',
                marker=dict(size=16, color='white', line=dict(color='#E63946', width=3)),
                name='Pins'
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Initial figure with first valid position
    if all_loci_b:
        joint_b = np.array(all_loci_b[0])
        joint_c = np.array(all_loci_c[0])
    else:
        joint_b = ground_a + crank_length * np.array([1, 0])
        joint_c = np.array([3.2, 1.8])

    fig = go.Figure(
        data=[
            # Locus traces (path of joints)
            go.Scatter(
                x=[p[0] for p in all_loci_b],
                y=[p[1] for p in all_loci_b],
                mode='lines',
                line=dict(color='#2E86AB', width=1, dash='dot'),
                name='Crank locus',
                opacity=0.5
            ),
            go.Scatter(
                x=[p[0] for p in all_loci_c],
                y=[p[1] for p in all_loci_c],
                mode='lines',
                line=dict(color='#A23B72', width=1, dash='dot'),
                name='Coupler locus',
                opacity=0.5
            ),
            # Links
            go.Scatter(
                x=[ground_a[0], joint_b[0]],
                y=[ground_a[1], joint_b[1]],
                mode='lines',
                line=dict(color='#2E86AB', width=8),
                name='Crank'
            ),
            go.Scatter(
                x=[joint_b[0], joint_c[0]],
                y=[joint_b[1], joint_c[1]],
                mode='lines',
                line=dict(color='#A23B72', width=8),
                name='Coupler'
            ),
            go.Scatter(
                x=[joint_c[0], ground_d[0]],
                y=[joint_c[1], ground_d[1]],
                mode='lines',
                line=dict(color='#F18F01', width=8),
                name='Rocker'
            ),
            # Ground joints
            go.Scatter(
                x=[ground_a[0], ground_d[0]],
                y=[ground_a[1], ground_d[1]],
                mode='markers',
                marker=dict(
                    size=22,
                    symbol='triangle-down',
                    color='white',
                    line=dict(color='#333333', width=3)
                ),
                name='Ground'
            ),
            # Moving joints
            go.Scatter(
                x=[joint_b[0], joint_c[0]],
                y=[joint_b[1], joint_c[1]],
                mode='markers',
                marker=dict(size=16, color='white', line=dict(color='#E63946', width=3)),
                name='Pins'
            ),
        ],
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        title="Four-Bar Linkage Animation",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                       "label": str(i), "method": "animate"}
                      for i, f in enumerate(frames)],
            "x": 0.1,
            "len": 0.8,
            "y": -0.05,
            "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
        }],
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 6]),
        yaxis=dict(range=[-2, 4]),
        plot_bgcolor='white',
        width=900,
        height=700,
    )

    return fig


if __name__ == "__main__":
    print("Creating static kinematic diagram...")
    static_fig = create_kinematic_diagram()
    static_fig.write_html("kinematic_static.html")
    print("  -> Saved to kinematic_static.html")

    print("\nCreating animated kinematic diagram...")
    animated_fig = create_animated_linkage()
    animated_fig.write_html("kinematic_animated.html")
    print("  -> Saved to kinematic_animated.html")

    print("\nOpen the HTML files in a browser to view.")
    print("You can also call .show() to open directly:")
    print("  static_fig.show()")
    print("  animated_fig.show()")

    # Optionally show in browser
    static_fig.show()
