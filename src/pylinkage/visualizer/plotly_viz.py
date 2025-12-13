"""
Plotly-based visualization for interactive kinematic diagrams.

This module provides interactive HTML output with zoom, pan, hover tooltips,
and animation controls.
"""


from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ..joints import Fixed, Prismatic
from ..joints.revolute import Pivot
from .symbols import (
    SymbolType,
    get_link_color,
    get_symbol_spec,
    is_ground_joint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .._types import Coord
    from ..linkage.linkage import Linkage


def _get_plotly_marker(symbol_type: SymbolType) -> dict[str, object]:
    """Get Plotly marker properties for a symbol type."""
    if symbol_type == SymbolType.GROUND:
        return {
            "symbol": "triangle-down",
            "size": 18,
            "line": {"width": 3},
        }
    elif symbol_type == SymbolType.CRANK:
        return {
            "symbol": "circle",
            "size": 16,
            "line": {"width": 3},
        }
    elif symbol_type == SymbolType.SLIDER:
        return {
            "symbol": "square",
            "size": 14,
            "line": {"width": 2},
        }
    else:  # REVOLUTE, FIXED
        return {
            "symbol": "circle",
            "size": 14,
            "line": {"width": 2},
        }


def plot_linkage_plotly(
    linkage: "Linkage",
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    *,
    title: str | None = None,
    show_dimensions: bool = False,
    show_loci: bool = True,
    show_labels: bool = True,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Create an interactive Plotly diagram of a linkage.

    Args:
        linkage: The linkage to visualize.
        loci: Optional precomputed loci. If None, runs simulation.
        title: Optional title for the diagram.
        show_dimensions: Whether to show dimension annotations.
        show_loci: Whether to show joint movement paths.
        show_labels: Whether to show joint name labels.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        A plotly Figure object.
    """
    # Run simulation if no loci provided
    if loci is None:
        loci = list(linkage.step())  # type: ignore[arg-type]
    else:
        loci = list(loci)

    if not loci:
        raise ValueError("No loci data available. Run linkage.step() first.")

    fig = go.Figure()

    # Get current joint positions (first frame for static)
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

    # Draw loci (movement paths)
    if show_loci and len(loci) > 1:
        for i, joint in enumerate(linkage.joints):
            spec = get_symbol_spec(joint)
            xs = [frame[i][0] for frame in loci]
            ys = [frame[i][1] for frame in loci]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": spec.color, "width": 1, "dash": "dot"},
                    opacity=0.5,
                    name=f"{joint.name} path" if joint.name else f"Joint {i} path",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Draw links
    link_index = 0
    drawn_links: set[tuple[int, int]] = set()

    for joint in linkage.joints:
        pos = get_position(joint)

        # Link to joint0
        if joint.joint0 is not None:
            parent_pos = get_position(joint.joint0)

            joint_ids = (id(joint), id(joint.joint0))
            rev_ids = (id(joint.joint0), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                color = get_link_color(link_index)
                fig.add_trace(
                    go.Scatter(
                        x=[parent_pos[0], pos[0]],
                        y=[parent_pos[1], pos[1]],
                        mode="lines",
                        line={"color": color, "width": 8},
                        name=f"Link {link_index + 1}",
                        hoverinfo="name",
                    )
                )
                drawn_links.add(joint_ids)

                # Dimension annotation
                if show_dimensions:
                    import math

                    length = math.sqrt(
                        (pos[0] - parent_pos[0]) ** 2 + (pos[1] - parent_pos[1]) ** 2
                    )
                    mid_x = (pos[0] + parent_pos[0]) / 2
                    mid_y = (pos[1] + parent_pos[1]) / 2
                    fig.add_annotation(
                        x=mid_x,
                        y=mid_y,
                        text=f"{length:.2f}",
                        showarrow=False,
                        font={"size": 10, "color": color},
                        bgcolor="white",
                        opacity=0.8,
                    )

                link_index += 1

        # Link to joint1
        if hasattr(joint, "joint1") and joint.joint1 is not None:
            if isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute":
                parent_pos = get_position(joint.joint1)

                joint_ids = (id(joint), id(joint.joint1))
                rev_ids = (id(joint.joint1), id(joint))
                if joint_ids not in drawn_links and rev_ids not in drawn_links:
                    color = get_link_color(link_index)
                    fig.add_trace(
                        go.Scatter(
                            x=[parent_pos[0], pos[0]],
                            y=[parent_pos[1], pos[1]],
                            mode="lines",
                            line={"color": color, "width": 8},
                            name=f"Link {link_index + 1}",
                            hoverinfo="name",
                        )
                    )
                    drawn_links.add(joint_ids)

                    if show_dimensions:
                        import math

                        length = math.sqrt(
                            (pos[0] - parent_pos[0]) ** 2 + (pos[1] - parent_pos[1]) ** 2
                        )
                        mid_x = (pos[0] + parent_pos[0]) / 2
                        mid_y = (pos[1] + parent_pos[1]) / 2
                        fig.add_annotation(
                            x=mid_x,
                            y=mid_y,
                            text=f"{length:.2f}",
                            showarrow=False,
                            font={"size": 10, "color": color},
                            bgcolor="white",
                            opacity=0.8,
                        )

                    link_index += 1

        # Handle Prismatic joints
        if isinstance(joint, Prismatic) and joint.joint1 is not None and joint.joint2 is not None:
            p1_pos = get_position(joint.joint1)
            p2_pos = get_position(joint.joint2)

            joint_ids = (id(joint.joint1), id(joint.joint2))
            rev_ids = (id(joint.joint2), id(joint.joint1))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                color = get_link_color(link_index)
                fig.add_trace(
                    go.Scatter(
                        x=[p1_pos[0], p2_pos[0]],
                        y=[p1_pos[1], p2_pos[1]],
                        mode="lines",
                        line={"color": color, "width": 6},
                        name=f"Link {link_index + 1}",
                        hoverinfo="name",
                    )
                )
                drawn_links.add(joint_ids)
                link_index += 1

    # Draw ground hatching for ground joints
    ground_joints = [j for j in linkage.joints if is_ground_joint(j)]
    for joint in ground_joints:
        pos = get_position(joint)
        # Draw hatching lines below ground symbol
        for i in range(3):
            offset = -0.1 - i * 0.05
            fig.add_trace(
                go.Scatter(
                    x=[pos[0] - 0.15 + i * 0.03, pos[0] + 0.15 - i * 0.03],
                    y=[pos[1] + offset, pos[1] + offset],
                    mode="lines",
                    line={"color": "#333333", "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Draw joints (on top)
    for joint in linkage.joints:
        pos = get_position(joint)
        spec = get_symbol_spec(joint)
        marker_props = _get_plotly_marker(spec.symbol_type)

        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers",
                marker={
                    **marker_props,
                    "color": "white",
                    "line": {**marker_props.get("line", {}), "color": spec.color},  # type: ignore[dict-item]
                },
                name=joint.name or type(joint).__name__,
                hovertemplate=(
                    f"<b>{joint.name or type(joint).__name__}</b><br>"
                    f"x: {pos[0]:.3f}<br>"
                    f"y: {pos[1]:.3f}<extra></extra>"
                ),
            )
        )

        # Label
        if show_labels and joint.name:
            fig.add_annotation(
                x=pos[0] + spec.label_offset[0] * 0.2,
                y=pos[1] + spec.label_offset[1] * 0.2,
                text=joint.name,
                showarrow=False,
                font={"size": 12, "color": "#333"},
            )

    # Layout
    fig.update_layout(
        title={
            "text": title or linkage.name or "Linkage Diagram",
            "font": {"size": 18},
        },
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        xaxis={
            "scaleanchor": "y",
            "scaleratio": 1,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#E5E5E5",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "#CCCCCC",
            "title": "X",
        },
        yaxis={
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "#E5E5E5",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "#CCCCCC",
            "title": "Y",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        margin={"r": 150},
    )

    return fig


def animate_linkage_plotly(
    linkage: "Linkage",
    loci: "Iterable[tuple[Coord, ...]] | None" = None,
    *,
    title: str | None = None,
    show_loci: bool = True,
    width: int = 900,
    height: int = 700,
    frame_duration: int = 50,
) -> go.Figure:
    """Create an animated Plotly diagram with play/pause controls.

    Args:
        linkage: The linkage to visualize.
        loci: Optional precomputed loci. If None, runs simulation.
        title: Optional title for the diagram.
        show_loci: Whether to show joint movement paths.
        width: Figure width in pixels.
        height: Figure height in pixels.
        frame_duration: Milliseconds per frame.

    Returns:
        A plotly Figure object with animation.
    """
    # Run simulation if no loci provided
    if loci is None:
        loci = list(linkage.step())  # type: ignore[arg-type]
    else:
        loci = list(loci)

    if not loci:
        raise ValueError("No loci data available. Run linkage.step() first.")

    # Build link pairs
    link_pairs: list[tuple[object, object]] = []
    drawn_links: set[tuple[int, int]] = set()

    for joint in linkage.joints:
        if joint.joint0 is not None:
            joint_ids = (id(joint), id(joint.joint0))
            rev_ids = (id(joint.joint0), id(joint))
            if joint_ids not in drawn_links and rev_ids not in drawn_links:
                link_pairs.append((joint, joint.joint0))
                drawn_links.add(joint_ids)

        if hasattr(joint, "joint1") and joint.joint1 is not None:
            if isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute":
                joint_ids = (id(joint), id(joint.joint1))
                rev_ids = (id(joint.joint1), id(joint))
                if joint_ids not in drawn_links and rev_ids not in drawn_links:
                    link_pairs.append((joint, joint.joint1))
                    drawn_links.add(joint_ids)

    # Create frames
    frames = []
    joint_list = list(linkage.joints)

    def get_pos_from_map(
        pos_map: dict[object, tuple[float, float]], joint: object
    ) -> tuple[float, float]:
        """Get position, handling implicit Static joints."""
        if joint in pos_map:
            return pos_map[joint]
        coord = joint.coord()  # type: ignore[attr-defined]
        return (coord[0] or 0.0, coord[1] or 0.0)

    for frame_idx, positions in enumerate(loci):
        pos_map: dict[object, tuple[float, float]] = {
            joint: (positions[i][0] or 0.0, positions[i][1] or 0.0)
            for i, joint in enumerate(joint_list)
        }

        frame_data = []

        # Links for this frame
        for link_idx, (j1, j2) in enumerate(link_pairs):
            p1 = get_pos_from_map(pos_map, j1)
            p2 = get_pos_from_map(pos_map, j2)
            color = get_link_color(link_idx)
            frame_data.append(
                go.Scatter(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    mode="lines",
                    line={"color": color, "width": 8},
                )
            )

        # Joints for this frame
        joint_xs = [pos_map[j][0] for j in joint_list]
        joint_ys = [pos_map[j][1] for j in joint_list]
        frame_data.append(
            go.Scatter(
                x=joint_xs,
                y=joint_ys,
                mode="markers",
                marker={"size": 14, "color": "white", "line": {"width": 2, "color": "#E63946"}},
            )
        )

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Initial data
    initial_pos_map: dict[object, tuple[float, float]] = {
        joint: (loci[0][i][0] or 0.0, loci[0][i][1] or 0.0)
        for i, joint in enumerate(joint_list)
    }
    initial_data = []

    def get_initial_pos(joint: object) -> tuple[float, float]:
        """Get initial position, handling implicit Static joints."""
        if joint in initial_pos_map:
            return initial_pos_map[joint]
        coord = joint.coord()  # type: ignore[attr-defined]
        return (coord[0] or 0.0, coord[1] or 0.0)

    # Loci traces (static, behind animation)
    if show_loci:
        for i, joint in enumerate(joint_list):
            spec = get_symbol_spec(joint)
            xs = [frame[i][0] for frame in loci]
            ys = [frame[i][1] for frame in loci]
            initial_data.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": spec.color, "width": 1, "dash": "dot"},
                    opacity=0.4,
                    name=f"{joint.name} path" if joint.name else f"Path {i}",
                    showlegend=False,
                )
            )

    # Initial links
    for link_idx, (j1, j2) in enumerate(link_pairs):
        p1 = get_initial_pos(j1)
        p2 = get_initial_pos(j2)
        color = get_link_color(link_idx)
        initial_data.append(
            go.Scatter(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                mode="lines",
                line={"color": color, "width": 8},
                name=f"Link {link_idx + 1}",
            )
        )

    # Initial joints
    joint_xs = [get_initial_pos(j)[0] for j in joint_list]
    joint_ys = [get_initial_pos(j)[1] for j in joint_list]
    initial_data.append(
        go.Scatter(
            x=joint_xs,
            y=joint_ys,
            mode="markers",
            marker={"size": 14, "color": "white", "line": {"width": 2, "color": "#E63946"}},
            name="Joints",
        )
    )

    fig = go.Figure(data=initial_data, frames=frames)

    # Animation controls
    fig.update_layout(
        title={"text": title or linkage.name or "Linkage Animation", "font": {"size": 18}},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": 0,
                "x": 0.1,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i, f in enumerate(frames)
                ],
                "x": 0.1,
                "len": 0.8,
                "y": -0.05,
                "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
            }
        ],
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        yaxis={},
        plot_bgcolor="white",
        width=width,
        height=height,
    )

    return fig
