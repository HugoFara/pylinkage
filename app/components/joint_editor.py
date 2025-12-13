"""Joint parameter editor forms for the Pylinkage web editor."""

import math
from typing import Any

import streamlit as st
from linkage_manager import get_joint_names
from state import get_linkage

# Parameter schemas for each joint type
JOINT_PARAM_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "Static": {
        "x": {"type": "float", "default": 0.0, "label": "X Position"},
        "y": {"type": "float", "default": 0.0, "label": "Y Position"},
    },
    "Crank": {
        "x": {"type": "float", "default": 0.0, "label": "Initial X"},
        "y": {"type": "float", "default": 1.0, "label": "Initial Y"},
        "joint0": {"type": "parent", "label": "Anchor Point", "required": True},
        "distance": {"type": "float", "default": 1.0, "label": "Radius", "min": 0.001},
        "angle": {
            "type": "angle",
            "default": 0.314,
            "label": "Angle per Step (rad)",
        },
    },
    "Revolute": {
        "x": {"type": "float", "default": 0.0, "label": "Initial X"},
        "y": {"type": "float", "default": 0.0, "label": "Initial Y"},
        "joint0": {"type": "parent", "label": "Parent 1", "required": True},
        "joint1": {"type": "parent", "label": "Parent 2", "required": True},
        "distance0": {
            "type": "float",
            "default": 1.0,
            "label": "Distance from Parent 1",
            "min": 0.001,
        },
        "distance1": {
            "type": "float",
            "default": 1.0,
            "label": "Distance from Parent 2",
            "min": 0.001,
        },
    },
    "Fixed": {
        "x": {"type": "float", "default": 0.0, "label": "Initial X"},
        "y": {"type": "float", "default": 0.0, "label": "Initial Y"},
        "joint0": {"type": "parent", "label": "Origin Point", "required": True},
        "joint1": {"type": "parent", "label": "Reference Point", "required": True},
        "distance": {
            "type": "float",
            "default": 1.0,
            "label": "Distance from Origin",
            "min": 0.001,
        },
        "angle": {"type": "angle", "default": 0.0, "label": "Angle Offset (rad)"},
    },
    "Prismatic": {
        "x": {"type": "float", "default": 0.0, "label": "Initial X"},
        "y": {"type": "float", "default": 0.0, "label": "Initial Y"},
        "joint0": {"type": "parent", "label": "Circle Center", "required": True},
        "joint1": {"type": "parent", "label": "Line Point 1", "required": True},
        "joint2": {"type": "parent", "label": "Line Point 2", "required": True},
        "revolute_radius": {
            "type": "float",
            "default": 1.0,
            "label": "Circle Radius",
            "min": 0.001,
        },
    },
}


def get_joint_types() -> list[str]:
    """Get list of available joint types."""
    return list(JOINT_PARAM_SCHEMAS.keys())


def render_joint_form(
    joint_type: str,
    key_prefix: str,
    existing_values: dict[str, Any] | None = None,
    exclude_joint_name: str | None = None,
) -> dict[str, Any]:
    """Render input fields for joint parameters based on type.

    Args:
        joint_type: The type of joint to render fields for.
        key_prefix: Unique prefix for streamlit widget keys.
        existing_values: Optional existing values to populate fields.
        exclude_joint_name: Joint name to exclude from parent dropdowns (self).

    Returns:
        Dictionary of parameter values.
    """
    schema = JOINT_PARAM_SCHEMAS.get(joint_type, {})
    params: dict[str, Any] = {}

    if existing_values is None:
        existing_values = {}

    for param_name, spec in schema.items():
        current_val = existing_values.get(param_name, spec.get("default"))
        widget_key = f"{key_prefix}_{param_name}"

        if spec["type"] == "float":
            min_val = spec.get("min", None)
            params[param_name] = st.number_input(
                spec["label"],
                value=float(current_val) if current_val is not None else 0.0,
                min_value=min_val,
                format="%.4f",
                key=widget_key,
            )

        elif spec["type"] == "angle":
            rad_val = float(current_val) if current_val is not None else 0.0
            deg_val = math.degrees(rad_val)
            params[param_name] = st.number_input(
                f"{spec['label']} ({deg_val:.1f} deg)",
                value=rad_val,
                format="%.4f",
                key=widget_key,
            )

        elif spec["type"] == "parent":
            params[param_name] = _render_parent_selector(
                spec["label"],
                current_val,
                widget_key,
                required=spec.get("required", False),
                exclude_name=exclude_joint_name,
            )

    return params


def _render_parent_selector(
    label: str,
    current_value: Any,
    key: str,
    required: bool = False,
    exclude_name: str | None = None,
) -> dict[str, Any] | None:
    """Render a dropdown to select a parent joint or enter coordinates.

    Args:
        label: The label for the selector.
        current_value: Current value (name ref dict, inline dict, or None).
        key: Unique streamlit widget key.
        required: Whether the parent is required.
        exclude_name: Joint name to exclude from options (self).

    Returns:
        Reference dict for the parent joint, or None.
    """
    linkage = get_linkage()
    joint_names = get_joint_names(linkage)

    # Exclude self from options
    if exclude_name and exclude_name in joint_names:
        joint_names = [n for n in joint_names if n != exclude_name]

    options = ["(coordinates)"] + joint_names
    if not required:
        options = ["(none)"] + options

    # Determine current selection
    default_idx = 0
    coord_x, coord_y = 0.0, 0.0

    if current_value is None:
        default_idx = 0 if not required else options.index("(coordinates)")
    elif isinstance(current_value, dict):
        if current_value.get("inline"):
            # Inline static - select coordinates mode
            default_idx = options.index("(coordinates)")
            coord_x = float(current_value.get("x", 0))
            coord_y = float(current_value.get("y", 0))
        elif current_value.get("ref"):
            # Reference to existing joint
            ref_name = current_value["ref"]
            if ref_name in options:
                default_idx = options.index(ref_name)
            else:
                default_idx = options.index("(coordinates)")

    selection = st.selectbox(label, options, index=default_idx, key=key)

    if selection == "(none)":
        return None
    elif selection == "(coordinates)":
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input(
                f"{label} X",
                value=coord_x,
                format="%.4f",
                key=f"{key}_x",
            )
        with col2:
            y = st.number_input(
                f"{label} Y",
                value=coord_y,
                format="%.4f",
                key=f"{key}_y",
            )
        # Return inline static definition
        return {"inline": True, "type": "Static", "x": x, "y": y, "name": None}
    else:
        # Return reference to existing joint
        return {"ref": selection}


def build_joint_data(
    joint_type: str,
    name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete joint data dictionary from form parameters.

    Args:
        joint_type: The type of joint.
        name: The joint name.
        params: Parameter values from the form.

    Returns:
        Complete joint data dictionary ready for serialization.
    """
    data: dict[str, Any] = {
        "type": joint_type,
        "name": name,
        "x": params.get("x", 0.0),
        "y": params.get("y", 0.0),
    }

    # Add parent references
    for parent_key in ["joint0", "joint1", "joint2"]:
        if parent_key in params and params[parent_key] is not None:
            data[parent_key] = params[parent_key]

    # Add type-specific parameters
    if joint_type == "Crank":
        data["distance"] = params.get("distance", 1.0)
        data["angle"] = params.get("angle", 0.314)
    elif joint_type == "Revolute":
        data["distance0"] = params.get("distance0", 1.0)
        data["distance1"] = params.get("distance1", 1.0)
    elif joint_type == "Fixed":
        data["distance"] = params.get("distance", 1.0)
        data["angle"] = params.get("angle", 0.0)
    elif joint_type == "Prismatic":
        data["revolute_radius"] = params.get("revolute_radius", 1.0)

    return data
