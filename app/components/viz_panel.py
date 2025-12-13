"""Visualization panel component for the Pylinkage web editor."""

import streamlit as st
from linkage_manager import try_simulate
from state import cache_loci, get_cached_loci, get_linkage, get_state

from pylinkage.visualizer.plotly_viz import animate_linkage_plotly, plot_linkage_plotly


def render_visualization() -> None:
    """Render the main visualization area."""
    linkage = get_linkage()

    if linkage is None or not linkage.joints:
        st.info(
            "No linkage defined. Use the sidebar to add joints or load an example."
        )
        return

    # Visualization controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_animation = st.checkbox("Animate", value=False, key="viz_animate")
    with col2:
        show_loci = st.checkbox("Show Paths", value=True, key="viz_loci")
    with col3:
        show_dimensions = st.checkbox("Show Dimensions", value=False, key="viz_dims")

    # Try to simulate
    state = get_state()
    cached = get_cached_loci()

    if cached is not None:
        loci = cached
        error = None
    else:
        loci, error = try_simulate(linkage)
        if loci is not None:
            cache_loci(loci)

    if error:
        st.error(f"Simulation Error: {error}")
        state.is_buildable = False
        # Try to show a static view without simulation
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
        if show_animation and loci:
            fig = animate_linkage_plotly(
                linkage,
                loci=loci,
                show_loci=show_loci,
                frame_duration=50,
            )
        else:
            fig = plot_linkage_plotly(
                linkage,
                loci=loci if show_loci else None,
                show_loci=show_loci,
                show_dimensions=show_dimensions,
            )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")


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
        st.plotly_chart(fig, use_container_width=True)
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
