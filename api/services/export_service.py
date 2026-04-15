"""Business logic for mechanism export operations."""

from __future__ import annotations

import io
import math
import tempfile
from typing import Any

from pylinkage.mechanism import Mechanism
from pylinkage.mechanism.joint import GroundJoint, PrismaticJoint, TrackerJoint
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

    # Track variable names for joints (used to resolve anchor references).
    joint_vars: dict[str, str] = {}
    # Components actually passed to ``Linkage([...])`` — Ground/Crank/Dyad
    # objects, but NOT the bare crank.output joints they expose.
    components: list[str] = []
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

    # Driver output joints are produced by the crank itself; never treat
    # them as static anchors even if they came in as plain trackers.
    driver_output_ids: set[str] = set()
    for link in mechanism.links:
        if isinstance(link, (DriverLink, ArcDriverLink)):
            output = getattr(link, "output_joint", None)
            if output is not None:
                driver_output_ids.add(output.id)

    # Pass 1: Ground joints. Tracker joints with no ref joints have no
    # motion constraint, so treat them as static anchors as well — this
    # mirrors how the editor renders free trackers as fixed pivots.
    def _is_static_anchor(j: Any) -> bool:
        if j.id in driver_output_ids:
            return False
        if isinstance(j, GroundJoint):
            return True
        if isinstance(j, TrackerJoint):
            return not (
                getattr(j, "ref_joint1_id", "") and getattr(j, "ref_joint2_id", "")
            )
        return False

    ground_joints = [j for j in mechanism.joints if _is_static_anchor(j)]
    for joint in ground_joints:
        var = _safe_var("ground", joint.name)
        joint_vars[joint.id] = var
        components.append(var)
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
        components.append(crank_var)

        # The output joint of the crank — exposed so dyads can reference it,
        # but it is NOT a standalone Linkage component.
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

    joint_by_id = {j.id: j for j in mechanism.joints}
    dyad_header_emitted = False

    def _emit_dyad(jid: str, anchors: list[tuple[str, float]]) -> None:
        nonlocal dyad_header_emitted
        if not dyad_header_emitted:
            lines.append("")
            lines.append("# Dyads (constrained connections)")
            dyad_header_emitted = True
        joint = joint_by_id[jid]
        a1_id, d1 = anchors[0]
        a2_id, d2 = anchors[1]
        var = _safe_var("dyad", joint.name)
        joint_vars[jid] = var
        components.append(var)
        if isinstance(joint, PrismaticJoint):
            lines.append(
                f"{var} = RRPDyad("
                f"anchor1={joint_vars[a1_id]}, "
                f"anchor2={joint_vars[a2_id]}, "
                f"distance1={d1:.6g}, "
                f"slide_direction=({joint.axis[0]:.6g}, {joint.axis[1]:.6g}))"
            )
        else:
            lines.append(
                f"{var} = RRRDyad("
                f"anchor1={joint_vars[a1_id]}, "
                f"anchor2={joint_vars[a2_id]}, "
                f"distance1={d1:.6g}, "
                f"distance2={d2:.6g})"
            )

    def _resolve_dyads_pass() -> bool:
        progress = False
        for jid in list(joint_connections.keys()):
            if jid in joint_vars:
                continue
            anchors = [
                (cid, dist)
                for cid, dist in joint_connections[jid]
                if cid in joint_vars
            ]
            if len(anchors) >= 2:
                _emit_dyad(jid, anchors)
                progress = True
        return progress

    def _pin_one_static_fallback() -> bool:
        """Promote one unresolved joint to a Ground anchor and return True.

        Picks the lowest-degree unresolved joint — a leaf revolute on the
        outer edge of the chain is almost always a fixed pivot, and pinning
        it lets the adjacent coupler resolve on the next pass. Couplers
        (high degree) are intentionally left for the dyad resolver.
        """
        candidate: tuple[int, str] | None = None
        for jid, conns in joint_connections.items():
            if jid in joint_vars or not conns:
                continue
            degree = len(conns)
            if candidate is None or degree < candidate[0]:
                candidate = (degree, jid)
        if candidate is None:
            return False
        jid = candidate[1]
        joint = joint_by_id[jid]
        x, y = joint.position
        if x is None or y is None:
            x, y = 0.0, 0.0
        var = _safe_var("ground", joint.name)
        joint_vars[jid] = var
        components.append(var)
        lines.append(f'{var} = Ground({x}, {y}, name="{joint.name or jid}")')
        return True

    # Alternate dyad resolution and static-fallback pinning until quiescent.
    safety_limit = max(len(mechanism.joints) * 2, 1)
    for _ in range(safety_limit):
        progress = _resolve_dyads_pass()
        if progress:
            continue
        if not _pin_one_static_fallback():
            break

    # Build linkage
    component_vars = list(dict.fromkeys(components))  # preserve order, dedupe
    lines.append("")
    lines.append("# Assemble and simulate")
    components_str = ", ".join(component_vars)
    lines.append(f'linkage = Linkage([{components_str}], name="{mechanism.name or "Unnamed"}")')
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

    loci = list(mechanism.step())
    drawing = plot_linkage_svg(mechanism, loci)
    return drawing.as_svg()


def export_dxf(mechanism: Mechanism) -> bytes:
    """Export mechanism as DXF file bytes.

    Args:
        mechanism: The built Mechanism object.

    Returns:
        DXF file content as bytes.
    """
    from pylinkage.visualizer import plot_linkage_dxf

    loci = list(mechanism.step())
    doc = plot_linkage_dxf(mechanism, loci)

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

    loci = list(mechanism.step())
    model = build_linkage_3d(mechanism, loci)

    with tempfile.NamedTemporaryFile(suffix=".step", delete=True) as tmp:
        model.export_step(tmp.name)
        tmp.seek(0)
        return tmp.read()
