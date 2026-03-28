"""Business logic for mechanism export operations."""

from __future__ import annotations

import io
import math
import tempfile
from typing import Any

from pylinkage.mechanism import Mechanism
from pylinkage.mechanism.conversion import mechanism_to_linkage
from pylinkage.mechanism.joint import GroundJoint, PrismaticJoint
from pylinkage.mechanism.link import ArcDriverLink, DriverLink, GroundLink


def generate_python_code(mechanism: Mechanism) -> str:
    """Generate Python code that recreates the mechanism using the pylinkage API.

    Args:
        mechanism: The built Mechanism object.

    Returns:
        Python source code as a string.
    """
    lines: list[str] = [
        '"""',
        f"Auto-generated code for mechanism: {mechanism.name or 'Unnamed'}",
        '"""',
        "",
        "from pylinkage.components import Ground",
        "from pylinkage.actuators import Crank",
        "from pylinkage.dyads import RRRDyad",
        "from pylinkage.simulation import Linkage",
        "",
        "# Ground points (fixed frame)",
    ]

    # Track variable names for joints
    joint_vars: dict[str, str] = {}
    var_counter = 0

    def _safe_var(prefix: str, name: str | None) -> str:
        nonlocal var_counter
        if name:
            # Sanitize name for Python identifier
            clean = name.replace(" ", "_").replace("-", "_")
            if clean.isidentifier():
                return clean
        var_counter += 1
        return f"{prefix}_{var_counter}"

    # Pass 1: Ground joints
    ground_joints = [j for j in mechanism.joints if isinstance(j, GroundJoint)]
    for joint in ground_joints:
        var = _safe_var("ground", joint.name)
        joint_vars[joint.id] = var
        x, y = joint.position
        lines.append(f'{var} = Ground({x}, {y}, name="{joint.name or joint.id}")')

    # Pass 2: Driver links (cranks)
    driver_links = [
        link for link in mechanism.links if isinstance(link, (DriverLink, ArcDriverLink))
    ]
    if driver_links:
        lines.append("")
        lines.append("# Cranks (motor-driven inputs)")

    for link in driver_links:
        if not isinstance(link, DriverLink):
            continue
        motor = link.motor_joint
        output = link.output_joint
        if motor is None or output is None:
            continue

        anchor_var = joint_vars.get(motor.id, "None")
        crank_var = _safe_var("crank", link.name)
        joint_vars[link.id + "_crank"] = crank_var

        # The output joint of the crank
        output_var = _safe_var("point", output.name)
        joint_vars[output.id] = output_var

        lines.append(
            f"{crank_var} = Crank("
            f"anchor={anchor_var}, "
            f"radius={link.radius:.6g}, "
            f"angular_velocity={link.angular_velocity:.6g})"
        )
        lines.append(f"{output_var} = {crank_var}.output")

    # Pass 3: Regular links forming dyads
    # Find revolute joints connected to two parent joints via links
    regular_links = [
        link
        for link in mechanism.links
        if not isinstance(link, (GroundLink, DriverLink, ArcDriverLink))
    ]

    # Build adjacency: for each non-ground/driver joint, find what other joints it connects to
    joint_connections: dict[str, list[tuple[str, float]]] = {}
    for link in regular_links:
        if len(link.joints) == 2:
            j1, j2 = link.joints
            dist = link.get_distance(j1, j2) or _compute_distance(j1, j2)
            joint_connections.setdefault(j1.id, []).append((j2.id, dist))
            joint_connections.setdefault(j2.id, []).append((j1.id, dist))

    # Find joints that have exactly 2 connections (dyad pattern)
    dyad_joints: list[str] = []
    for joint in mechanism.joints:
        if joint.id in joint_vars:
            continue  # Already handled (ground or crank output)
        conns = joint_connections.get(joint.id, [])
        if len(conns) >= 2:
            dyad_joints.append(joint.id)

    if dyad_joints:
        lines.append("")
        lines.append("# Dyads (constrained connections)")

    for jid in dyad_joints:
        joint = next(j for j in mechanism.joints if j.id == jid)
        conns = joint_connections[jid]

        # Find two anchor connections where the anchors are already defined
        anchors = [(cid, dist) for cid, dist in conns if cid in joint_vars]

        if len(anchors) >= 2:
            a1_id, d1 = anchors[0]
            a2_id, d2 = anchors[1]
            var = _safe_var("dyad", joint.name)
            joint_vars[jid] = var

            if isinstance(joint, PrismaticJoint):
                lines.append(
                    f"{var} = RRPDyad("
                    f"anchor1={joint_vars[a1_id]}, "
                    f"anchor2={joint_vars[a2_id]}, "
                    f"distance1={d1:.6g}, "
                    f'slide_direction=({joint.axis[0]:.6g}, {joint.axis[1]:.6g}))'
                )
            else:
                lines.append(
                    f"{var} = RRRDyad("
                    f"anchor1={joint_vars[a1_id]}, "
                    f"anchor2={joint_vars[a2_id]}, "
                    f"distance1={d1:.6g}, "
                    f"distance2={d2:.6g})"
                )

    # Handle remaining joints (those with only 1 known anchor so far)
    # Iterate until no more can be resolved
    max_iterations = len(mechanism.joints)
    for _ in range(max_iterations):
        resolved_any = False
        for jid in list(joint_connections.keys()):
            if jid in joint_vars:
                continue
            joint = next((j for j in mechanism.joints if j.id == jid), None)
            if joint is None:
                continue
            conns = joint_connections[jid]
            anchors = [(cid, dist) for cid, dist in conns if cid in joint_vars]
            if len(anchors) >= 2:
                a1_id, d1 = anchors[0]
                a2_id, d2 = anchors[1]
                var = _safe_var("dyad", joint.name)
                joint_vars[jid] = var
                lines.append(
                    f"{var} = RRRDyad("
                    f"anchor1={joint_vars[a1_id]}, "
                    f"anchor2={joint_vars[a2_id]}, "
                    f"distance1={d1:.6g}, "
                    f"distance2={d2:.6g})"
                )
                resolved_any = True
        if not resolved_any:
            break

    # Build linkage
    all_vars = list(dict.fromkeys(joint_vars.values()))  # preserve order, deduplicate
    lines.append("")
    lines.append("# Assemble and simulate")
    components_str = ", ".join(all_vars)
    lines.append(
        f'linkage = Linkage([{components_str}], name="{mechanism.name or "Unnamed"}")'
    )
    lines.append("")
    lines.append("# Run simulation")
    lines.append("loci = list(linkage.step())")
    lines.append("")
    lines.append("# Visualize")
    lines.append("from pylinkage.visualizer import show_linkage")
    lines.append("show_linkage(linkage, loci)")
    lines.append("")

    return "\n".join(lines)


def _compute_distance(j1: Any, j2: Any) -> float:
    """Compute distance between two joints from their positions."""
    x1, y1 = j1.position
    x2, y2 = j2.position
    if any(v is None for v in (x1, y1, x2, y2)):
        return 1.0
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def export_svg(mechanism: Mechanism) -> str:
    """Export mechanism as SVG string.

    Args:
        mechanism: The built Mechanism object.

    Returns:
        SVG content as a string.
    """
    from pylinkage.visualizer import plot_linkage_svg

    linkage = mechanism_to_linkage(mechanism)
    loci = list(linkage.step())
    drawing = plot_linkage_svg(linkage, loci)
    return drawing.as_svg()


def export_dxf(mechanism: Mechanism) -> bytes:
    """Export mechanism as DXF file bytes.

    Args:
        mechanism: The built Mechanism object.

    Returns:
        DXF file content as bytes.
    """
    from pylinkage.visualizer import plot_linkage_dxf

    linkage = mechanism_to_linkage(mechanism)
    loci = list(linkage.step())
    doc = plot_linkage_dxf(linkage, loci)

    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def export_step(mechanism: Mechanism) -> bytes:
    """Export mechanism as STEP file bytes.

    Args:
        mechanism: The built Mechanism object.

    Returns:
        STEP file content as bytes.
    """
    from pylinkage.visualizer import build_linkage_3d

    linkage = mechanism_to_linkage(mechanism)
    loci = list(linkage.step())
    model = build_linkage_3d(linkage, loci)

    with tempfile.NamedTemporaryFile(suffix=".step", delete=True) as tmp:
        model.export_step(tmp.name)
        tmp.seek(0)
        return tmp.read()
