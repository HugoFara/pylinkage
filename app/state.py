"""Session state management for the Pylinkage web editor."""

from dataclasses import dataclass
from typing import Any

import streamlit as st

import pylinkage as pl


@dataclass
class LinkageState:
    """Manages the current linkage state in session."""

    linkage_dict: dict[str, Any] | None = None
    error_message: str | None = None
    is_buildable: bool = True
    selected_joint_name: str | None = None
    loci: list[tuple[tuple[float, float], ...]] | None = None


def init_session_state() -> None:
    """Initialize session state with defaults."""
    if "linkage_state" not in st.session_state:
        st.session_state.linkage_state = LinkageState()
    if "linkage" not in st.session_state:
        st.session_state.linkage = None


def get_state() -> LinkageState:
    """Get the current linkage state."""
    return st.session_state.linkage_state


def get_linkage() -> pl.Linkage | None:
    """Get the current linkage from session state."""
    return st.session_state.get("linkage")


def set_linkage(linkage: pl.Linkage) -> None:
    """Set the current linkage and update serialized form."""
    st.session_state.linkage = linkage
    state = get_state()
    state.linkage_dict = linkage.to_dict()
    state.loci = None  # Clear cached loci
    state.error_message = None
    # Increment linkage version to reset all visualization widgets
    st.session_state.linkage_version = st.session_state.get("linkage_version", 0) + 1


def clear_linkage() -> None:
    """Clear the current linkage."""
    st.session_state.linkage = None
    state = get_state()
    state.linkage_dict = None
    state.loci = None
    state.error_message = None
    state.is_buildable = True
    state.selected_joint_name = None


def set_error(message: str) -> None:
    """Set an error message."""
    state = get_state()
    state.error_message = message
    state.is_buildable = False


def clear_error() -> None:
    """Clear the error message."""
    state = get_state()
    state.error_message = None
    state.is_buildable = True


def set_selected_joint(name: str | None) -> None:
    """Set the selected joint for editing."""
    get_state().selected_joint_name = name


def get_selected_joint() -> str | None:
    """Get the selected joint name."""
    return get_state().selected_joint_name


def cache_loci(loci: list[tuple[tuple[float, float], ...]]) -> None:
    """Cache simulation results."""
    get_state().loci = loci


def get_cached_loci() -> list[tuple[tuple[float, float], ...]] | None:
    """Get cached simulation results."""
    return get_state().loci
