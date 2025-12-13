"""Sidebar controls for the Pylinkage web editor."""

import streamlit as st
from examples.prebuilt import get_example_names, load_example
from linkage_manager import (
    add_joint_to_linkage,
    delete_joint_from_linkage,
    get_joint_data,
    update_joint_in_linkage,
    validate_joint_name,
)
from state import (
    clear_linkage,
    get_linkage,
    get_selected_joint,
    set_linkage,
    set_selected_joint,
)

from components.file_io import render_file_operations
from components.joint_editor import (
    build_joint_data,
    get_joint_types,
    render_joint_form,
)


def render_sidebar() -> None:
    """Render the sidebar with all controls."""
    st.sidebar.title("Linkage Editor")

    # Example selector
    _render_example_selector()

    st.sidebar.divider()

    # Joint list
    _render_joint_list()

    st.sidebar.divider()

    # Add joint form
    _render_add_joint_section()

    st.sidebar.divider()

    # File operations
    render_file_operations()


def _render_example_selector() -> None:
    """Dropdown to load prebuilt examples."""
    example_names = get_example_names()

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        selected = st.selectbox(
            "Load Example",
            options=["(select)"] + example_names,
            key="example_selector",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("Load", key="load_example_btn") and selected != "(select)":
            linkage = load_example(selected)
            set_linkage(linkage)
            st.rerun()


def _render_joint_list() -> None:
    """Display list of joints with edit/delete buttons."""
    st.sidebar.subheader("Joints")
    linkage = get_linkage()

    if linkage is None or not linkage.joints:
        st.sidebar.info("No joints defined.")
        return

    selected_joint = get_selected_joint()

    for joint in linkage.joints:
        joint_type = type(joint).__name__
        joint_name = joint.name or "(unnamed)"
        is_selected = selected_joint == joint_name

        # Joint row with name and buttons on separate lines for clarity
        type_abbrev = {
            "Static": "S",
            "Crank": "C",
            "Revolute": "R",
            "Fixed": "F",
            "Prismatic": "P",
        }
        abbrev = type_abbrev.get(joint_type, "?")

        # Joint name
        if is_selected:
            st.sidebar.markdown(f"**[{abbrev}] {joint_name}**")
        else:
            st.sidebar.write(f"[{abbrev}] {joint_name}")

        # Buttons in a row below the name
        btn_col1, btn_col2, btn_col3 = st.sidebar.columns([1, 1, 1])
        with btn_col1:
            edit_label = "Close" if is_selected else "Edit"
            if st.button(edit_label, key=f"edit_{joint_name}", use_container_width=True):
                if is_selected:
                    set_selected_joint(None)
                else:
                    set_selected_joint(joint_name)
                st.rerun()
        with btn_col2:
            if st.button("Del", key=f"del_{joint_name}", use_container_width=True):
                new_linkage, error = delete_joint_from_linkage(linkage, joint_name)
                if error:
                    st.sidebar.error(error)
                elif new_linkage is None:
                    clear_linkage()
                else:
                    set_linkage(new_linkage)
                if selected_joint == joint_name:
                    set_selected_joint(None)
                st.rerun()
        with btn_col3:
            pass  # Empty column for spacing

        # Show edit form if selected
        if is_selected:
            _render_edit_joint_form(joint_name)


def _render_edit_joint_form(joint_name: str) -> None:
    """Render inline edit form for a joint."""
    linkage = get_linkage()
    if linkage is None:
        return

    joint_data = get_joint_data(linkage, joint_name)
    if joint_data is None:
        return

    joint_type = joint_data.get("type", "Static")

    with st.sidebar.container():
        st.markdown("---")
        st.caption(f"Editing: {joint_name} ({joint_type})")

        # Render parameter form
        params = render_joint_form(
            joint_type,
            key_prefix=f"edit_{joint_name}",
            existing_values=joint_data,
            exclude_joint_name=joint_name,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", key=f"save_{joint_name}"):
                updated_data = build_joint_data(joint_type, joint_name, params)
                new_linkage, error = update_joint_in_linkage(
                    linkage, joint_name, updated_data
                )
                if error:
                    st.error(error)
                elif new_linkage:
                    set_linkage(new_linkage)
                    set_selected_joint(None)
                    st.rerun()

        with col2:
            if st.button("Cancel", key=f"cancel_{joint_name}"):
                set_selected_joint(None)
                st.rerun()

        st.markdown("---")


def _render_add_joint_section() -> None:
    """Render the add joint form section."""
    with st.sidebar.expander("Add New Joint", expanded=False):
        # Joint type selector
        joint_type = st.selectbox(
            "Joint Type",
            options=get_joint_types(),
            key="new_joint_type",
        )

        # Joint name
        linkage = get_linkage()
        default_name = _generate_default_name(linkage, joint_type)
        name = st.text_input("Name", value=default_name, key="new_joint_name")

        # Validate name
        name_error = validate_joint_name(linkage, name)
        if name_error:
            st.warning(name_error)

        # Render parameter form
        params = render_joint_form(
            joint_type,
            key_prefix="new_joint",
            exclude_joint_name=None,
        )

        # Add button
        if st.button("Add Joint", key="add_joint_btn", disabled=bool(name_error)):
            joint_data = build_joint_data(joint_type, name, params)
            new_linkage, error = add_joint_to_linkage(linkage, joint_data)
            if error:
                st.error(error)
            elif new_linkage:
                set_linkage(new_linkage)
                st.rerun()


def _generate_default_name(linkage, joint_type: str) -> str:
    """Generate a default unique name for a new joint."""
    existing = set()
    if linkage:
        existing = {j.name for j in linkage.joints if j.name}

    # Use type prefix
    prefix_map = {
        "Static": "S",
        "Crank": "C",
        "Revolute": "R",
        "Fixed": "F",
        "Prismatic": "P",
    }
    prefix = prefix_map.get(joint_type, "J")

    # Find next available number
    i = 1
    while f"{prefix}{i}" in existing:
        i += 1

    return f"{prefix}{i}"
