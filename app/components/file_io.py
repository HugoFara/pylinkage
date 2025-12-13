"""File I/O controls for the Pylinkage web editor."""

import json

import streamlit as st
from state import clear_linkage, get_linkage, set_linkage

import pylinkage as pl


def render_file_operations() -> None:
    """Render file save/load controls in sidebar."""
    st.sidebar.subheader("File Operations")

    linkage = get_linkage()

    # Download JSON
    if linkage and linkage.joints:
        try:
            json_str = json.dumps(linkage.to_dict(), indent=2)
            linkage_name = linkage.name or "linkage"
            filename = f"{linkage_name.lower().replace(' ', '_')}.json"

            st.sidebar.download_button(
                label="Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json",
                key="download_json_btn",
            )
        except Exception as e:
            st.sidebar.error(f"Export error: {e}")
    else:
        st.sidebar.button(
            "Download JSON",
            disabled=True,
            key="download_json_btn_disabled",
        )

    # Upload JSON
    uploaded_file = st.sidebar.file_uploader(
        "Load JSON",
        type=["json"],
        key="json_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            loaded_linkage = pl.Linkage.from_dict(data)
            set_linkage(loaded_linkage)
            st.sidebar.success(f"Loaded: {loaded_linkage.name or 'Linkage'}")
            st.rerun()
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file")
        except Exception as e:
            st.sidebar.error(f"Failed to load: {e}")

    # Clear button
    st.sidebar.divider()
    if st.sidebar.button("Clear Linkage", key="clear_linkage_btn"):
        clear_linkage()
        st.rerun()
