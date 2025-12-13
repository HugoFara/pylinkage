"""Visualization panel component for the Pylinkage web editor."""

import plotly.graph_objects as go
import streamlit as st
from linkage_manager import try_simulate
from state import cache_loci, get_cached_loci, get_linkage, get_state

from pylinkage.visualizer.plotly_viz import plot_linkage_plotly


def render_visualization() -> None:
    """Render the main visualization area."""
    linkage = get_linkage()

    if linkage is None or not linkage.joints:
        st.info(
            "No linkage defined. Use the sidebar to add joints or load an example."
        )
        return

    # Try to simulate first (needed for all visualizations)
    state = get_state()
    cached = get_cached_loci()

    if cached is not None:
        loci = cached
        error = None
    else:
        loci, error = try_simulate(linkage)
        if loci is not None:
            cache_loci(loci)

    # Visualization controls
    # Use linkage version in keys to reset state when linkage changes
    version = st.session_state.get("linkage_version", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        # Disable animation if no valid loci
        can_animate = loci is not None and len(loci) > 1
        show_animation = st.checkbox(
            "Animate",
            value=False,
            key=f"viz_animate_{version}",
            disabled=not can_animate,
        )
    with col2:
        show_loci = st.checkbox("Show Paths", value=True, key=f"viz_loci_{version}")
    with col3:
        show_dimensions = st.checkbox(
            "Show Dimensions", value=False, key=f"viz_dims_{version}"
        )

    if error:
        st.error(f"Simulation Error: {error}")
        state.is_buildable = False
        _render_static_fallback(linkage)
    else:
        state.is_buildable = True
        _render_linkage_plot(linkage, loci, show_animation, show_loci, show_dimensions)

    # Status panel
    _render_status_panel(linkage, state)


def _render_linkage_plot(
    linkage,
    loci: list | None,
    show_animation: bool,
    show_loci: bool,
    show_dimensions: bool,
) -> None:
    """Render the linkage visualization."""
    try:
        if show_animation and loci and len(loci) > 1:
            _render_animation_with_controls(linkage, loci, show_loci)
        else:
            fig = plot_linkage_plotly(
                linkage,
                loci=loci if show_loci else None,
                show_loci=show_loci,
                show_dimensions=show_dimensions,
            )
            st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Visualization error: {e}")


def _render_animation_with_controls(linkage, loci: list, show_loci: bool) -> None:
    """Render animation with Plotly's native animation system."""
    version = st.session_state.get("linkage_version", 0)

    # Create animated figure using Plotly's native animation (15 fps default)
    fig = _create_animated_figure(linkage, loci, show_loci, fps=15)

    st.plotly_chart(fig, width="stretch", key=f"anim_chart_{version}")

    st.caption(
        f"{len(loci)} frames | Use the Play button and slider below the chart to control animation"
    )


def _create_animated_figure(
    linkage, loci: list, show_loci: bool, fps: int
) -> go.Figure:
    """Create a Plotly figure with native animation frames."""
    from pylinkage.joints import Fixed
    from pylinkage.joints.revolute import Pivot
    from pylinkage.visualizer.symbols import get_link_color, get_symbol_spec

    fig = go.Figure()

    # Build link structure once (connections don't change)
    links_info = []  # List of (joint, parent_joint, link_idx)
    link_idx = 0
    drawn = set()

    for joint in linkage.joints:
        if joint.joint0 is not None:
            key = (id(joint), id(joint.joint0))
            rev_key = (id(joint.joint0), id(joint))
            if key not in drawn and rev_key not in drawn:
                links_info.append((joint, joint.joint0, link_idx))
                drawn.add(key)
                link_idx += 1

        if (
            hasattr(joint, "joint1")
            and joint.joint1 is not None
            and (isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute")
        ):
            key = (id(joint), id(joint.joint1))
            rev_key = (id(joint.joint1), id(joint))
            if key not in drawn and rev_key not in drawn:
                links_info.append((joint, joint.joint1, link_idx))
                drawn.add(key)
                link_idx += 1

    # Helper to get position from a frame
    def get_pos_at_frame(joint, frame_positions):
        idx = list(linkage.joints).index(joint)
        return (frame_positions[idx][0] or 0.0, frame_positions[idx][1] or 0.0)

    # Initial frame (frame 0)
    initial_positions = loci[0]

    # Draw loci paths (static, full trajectory) - these don't animate
    if show_loci:
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
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add initial link traces (these will be animated)
    for joint, parent, lidx in links_info:
        pos = get_pos_at_frame(joint, initial_positions)
        p0 = get_pos_at_frame(parent, initial_positions)
        color = get_link_color(lidx)
        fig.add_trace(
            go.Scatter(
                x=[p0[0], pos[0]],
                y=[p0[1], pos[1]],
                mode="lines",
                line={"color": color, "width": 8},
                name=f"Link {lidx + 1}",
            )
        )

    # Add initial joint traces (these will be animated)
    for joint in linkage.joints:
        pos = get_pos_at_frame(joint, initial_positions)
        spec = get_symbol_spec(joint)
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers",
                marker={
                    "size": 14,
                    "color": "white",
                    "line": {"width": 2, "color": spec.color},
                },
                name=joint.name or type(joint).__name__,
            )
        )

    # Calculate trace indices
    # Loci traces come first (if show_loci), then links, then joints
    num_loci_traces = len(linkage.joints) if show_loci else 0
    num_link_traces = len(links_info)
    num_joint_traces = len(linkage.joints)

    # Create animation frames
    frames = []
    for frame_num, frame_positions in enumerate(loci):
        frame_data = []

        # Update link traces
        for joint, parent, _ in links_info:
            pos = get_pos_at_frame(joint, frame_positions)
            p0 = get_pos_at_frame(parent, frame_positions)
            frame_data.append(
                go.Scatter(x=[p0[0], pos[0]], y=[p0[1], pos[1]])
            )

        # Update joint traces
        for joint in linkage.joints:
            pos = get_pos_at_frame(joint, frame_positions)
            frame_data.append(go.Scatter(x=[pos[0]], y=[pos[1]]))

        # Build trace indices to update (skip loci traces)
        trace_indices = list(
            range(num_loci_traces, num_loci_traces + num_link_traces + num_joint_traces)
        )

        frames.append(
            go.Frame(data=frame_data, name=str(frame_num), traces=trace_indices)
        )

    fig.frames = frames

    # Calculate axis ranges from all positions
    all_x = [pos[0] for frame in loci for pos in frame if pos[0] is not None]
    all_y = [pos[1] for frame in loci for pos in frame if pos[1] is not None]
    if not all_x or not all_y:
        # Fallback if all positions are None
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
    else:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.1 or 1
    y_margin = (y_max - y_min) * 0.1 or 1

    # Frame duration in milliseconds
    frame_duration = 1000 // fps

    # Add animation controls
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": -0.05,
                "x": 0.1,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "\u25b6 Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "\u23f8 Pause",
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
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.85,
                "x": 0.15,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(i + 1),
                        "method": "animate",
                    }
                    for i in range(len(loci))
                ],
            }
        ],
    )

    # Layout
    fig.update_layout(
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        xaxis={
            "scaleanchor": "y",
            "scaleratio": 1,
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "range": [x_min - x_margin, x_max + x_margin],
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#E5E5E5",
            "range": [y_min - y_margin, y_max + y_margin],
        },
        plot_bgcolor="white",
        margin={"r": 150, "b": 100},
        height=550,
    )

    return fig


def _create_frame_figure(linkage, loci: list, frame_idx: int, show_loci: bool) -> go.Figure:
    """Create a figure showing a specific animation frame."""
    from pylinkage.joints import Fixed
    from pylinkage.joints.revolute import Pivot
    from pylinkage.visualizer.symbols import get_link_color, get_symbol_spec

    fig = go.Figure()

    # Get positions for this frame
    positions = loci[frame_idx]
    pos_map = {
        joint: (positions[i][0] or 0.0, positions[i][1] or 0.0)
        for i, joint in enumerate(linkage.joints)
    }

    def get_pos(joint):
        if joint in pos_map:
            return pos_map[joint]
        coord = joint.coord()
        return (coord[0] or 0.0, coord[1] or 0.0)

    # Draw loci paths (full trajectory)
    if show_loci:
        for i, joint in enumerate(linkage.joints):
            spec = get_symbol_spec(joint)
            xs = [frame[i][0] for frame in loci]
            ys = [frame[i][1] for frame in loci]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line={"color": spec.color, "width": 1, "dash": "dot"},
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            ))

    # Draw links
    link_idx = 0
    drawn = set()

    for joint in linkage.joints:
        pos = get_pos(joint)

        if joint.joint0 is not None:
            p0 = get_pos(joint.joint0)
            key = (id(joint), id(joint.joint0))
            rev_key = (id(joint.joint0), id(joint))
            if key not in drawn and rev_key not in drawn:
                color = get_link_color(link_idx)
                fig.add_trace(go.Scatter(
                    x=[p0[0], pos[0]], y=[p0[1], pos[1]],
                    mode="lines",
                    line={"color": color, "width": 8},
                    name=f"Link {link_idx + 1}",
                ))
                drawn.add(key)
                link_idx += 1

        if (
            hasattr(joint, "joint1")
            and joint.joint1 is not None
            and (isinstance(joint, (Fixed, Pivot)) or type(joint).__name__ == "Revolute")
        ):
            p1 = get_pos(joint.joint1)
            key = (id(joint), id(joint.joint1))
            rev_key = (id(joint.joint1), id(joint))
            if key not in drawn and rev_key not in drawn:
                color = get_link_color(link_idx)
                fig.add_trace(go.Scatter(
                    x=[p1[0], pos[0]], y=[p1[1], pos[1]],
                    mode="lines",
                    line={"color": color, "width": 8},
                    name=f"Link {link_idx + 1}",
                ))
                drawn.add(key)
                link_idx += 1

    # Draw joints
    for joint in linkage.joints:
        pos = get_pos(joint)
        spec = get_symbol_spec(joint)
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode="markers",
            marker={"size": 14, "color": "white", "line": {"width": 2, "color": spec.color}},
            name=joint.name or type(joint).__name__,
            hovertemplate=f"<b>{joint.name}</b><br>x: {pos[0]:.3f}<br>y: {pos[1]:.3f}<extra></extra>",
        ))

    # Layout
    fig.update_layout(
        showlegend=True,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        xaxis={"scaleanchor": "y", "scaleratio": 1, "showgrid": True, "gridcolor": "#E5E5E5"},
        yaxis={"showgrid": True, "gridcolor": "#E5E5E5"},
        plot_bgcolor="white",
        margin={"r": 150, "b": 50},
        height=500,
    )

    return fig


def _render_static_fallback(linkage) -> None:
    """Attempt to render a static view when simulation fails."""
    try:
        # Try to create a single-frame view using current joint positions
        fig = plot_linkage_plotly(
            linkage,
            loci=None,
            show_loci=False,
            show_dimensions=False,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.warning("Cannot render linkage visualization.")


def _render_status_panel(linkage, state) -> None:
    """Show linkage status information."""
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if state.is_buildable:
            st.success("Buildable")
        else:
            st.error("Unbuildable")

    with col2:
        st.metric("Joints", len(linkage.joints))

    with col3:
        if state.is_buildable:
            try:
                period = linkage.get_rotation_period()
                st.metric("Rotation Period", f"{period} steps")
            except Exception:
                st.metric("Rotation Period", "N/A")
        else:
            st.metric("Rotation Period", "N/A")
