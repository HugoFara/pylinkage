"""
Pylinkage Web Editor - Main Entry Point.

Run with: uv run streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add app directory to path for imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import streamlit as st  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402
from components.viz_panel import render_visualization  # noqa: E402
from state import get_state, init_session_state  # noqa: E402


def main() -> None:
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Pylinkage Editor",
        page_icon="\u2699",  # gear emoji
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("Pylinkage Editor")
    st.caption("Build and simulate planar linkages interactively")

    # Render sidebar controls
    render_sidebar()

    # Main visualization area
    render_visualization()

    # Show any pending errors
    state = get_state()
    if state.error_message:
        st.error(state.error_message)


if __name__ == "__main__":
    main()
