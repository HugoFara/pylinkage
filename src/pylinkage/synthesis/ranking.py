"""Solution quality metrics and ranking utilities.

Provides functions to evaluate and rank synthesis solutions by
simulating the resulting linkage and computing kinematic quality
metrics (path accuracy, transmission angle, link ratios, etc.).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from ._types import PrecisionPoint
from .topology_types import QualityMetrics

if TYPE_CHECKING:
    from ..simulation import Linkage


def compute_metrics(
    linkage: Linkage,
    precision_points: list[PrecisionPoint],
    num_links: int = 4,
) -> QualityMetrics:
    """Compute quality metrics for a synthesized linkage.

    Simulates the linkage and evaluates path accuracy, transmission
    angle quality, link ratios, and compactness.

    Args:
        linkage: Synthesized Linkage to evaluate.
        precision_points: Target points for accuracy measurement.
        num_links: Number of links in the topology (for simplicity score).

    Returns:
        QualityMetrics with all fields populated including overall_score.
    """
    accuracy = compute_path_accuracy(linkage, precision_points)
    link_ratio = compute_link_ratio(linkage)
    compactness = compute_compactness(linkage)
    transmission = compute_transmission_angle(linkage)
    grashof = _check_grashof_driver(linkage)

    metrics = QualityMetrics(
        path_accuracy=accuracy,
        min_transmission_angle=transmission,
        link_ratio=link_ratio,
        compactness=compactness,
        num_links=num_links,
        is_grashof=grashof,
    )
    metrics.overall_score = score_solution(metrics)
    return metrics


def compute_path_accuracy(
    linkage: Linkage,
    precision_points: list[PrecisionPoint],
) -> float:
    """Compute RMS path accuracy of coupler trajectory vs precision points.

    Simulates one full cycle and finds the minimum distance from the
    coupler point trajectory to each precision point. Returns the RMS
    of these minimum distances.

    Args:
        linkage: Linkage to evaluate.
        precision_points: Target points.

    Returns:
        RMS distance (0.0 = perfect).
    """
    from .._compat import get_parts
    from ..exceptions import UnbuildableError

    if not precision_points:
        return 0.0

    parts = get_parts(linkage)

    # Find coupler point joint (named "P" by convention)
    coupler_joint = None
    for joint in parts:
        if getattr(joint, "name", None) == "P":
            coupler_joint = joint
            break
    if coupler_joint is None:
        coupler_joint = parts[-1]

    # Simulate
    try:
        trajectory: list[tuple[float, float]] = []
        coupler_idx = parts.index(coupler_joint)
        for positions in linkage.step():
            px, py = positions[coupler_idx]
            if px is not None and py is not None:
                trajectory.append((px, py))
    except UnbuildableError:
        return float("inf")

    if not trajectory:
        return float("inf")

    # RMS of minimum distances
    sum_sq = 0.0
    for px, py in precision_points:
        min_dist = min(math.sqrt((tx - px) ** 2 + (ty - py) ** 2) for tx, ty in trajectory)
        sum_sq += min_dist * min_dist

    return math.sqrt(sum_sq / len(precision_points))


def compute_transmission_angle(linkage: Linkage) -> float:
    """Compute minimum transmission angle over the motion cycle.

    The transmission angle is the angle between the coupler and the
    output link (rocker). A value near 90 degrees is ideal; values
    near 0 or 180 indicate poor force transmission.

    For non-four-bar linkages, estimates from the first Revolute joint.

    Args:
        linkage: Linkage to evaluate.

    Returns:
        Minimum transmission angle in degrees (0-90 range).
    """
    from .._compat import get_parts
    from ..exceptions import UnbuildableError

    parts = get_parts(linkage)
    # Find the first RRR-like dyad (coupler-rocker connection)
    revolute = None
    for part in parts:
        if type(part).__name__ in ("RRRDyad", "Revolute"):
            revolute = part
            break

    if revolute is None:
        return 0.0

    # We need the coupler-side anchor (B) and the rocker-side anchor (D)
    joint_b = getattr(revolute, "anchor1", None) or getattr(revolute, "joint0", None)
    joint_d = getattr(revolute, "anchor2", None) or getattr(revolute, "joint1", None)

    if joint_b is None or joint_d is None:
        return 0.0

    # Resolve AnchorProxy to its parent component
    joint_b = getattr(joint_b, "_parent", joint_b)
    joint_d = getattr(joint_d, "_parent", joint_d)

    min_angle = 180.0
    try:
        rev_idx = parts.index(revolute)
        b_idx = parts.index(joint_b)
        d_idx = parts.index(joint_d)

        for positions in linkage.step():
            bx, by = positions[b_idx]
            cx, cy = positions[rev_idx]
            dx, dy = positions[d_idx]

            if bx is None or by is None or cx is None or cy is None or dx is None or dy is None:
                continue

            # Coupler vector B→C
            ux, uy = cx - bx, cy - by
            # Rocker vector D→C
            vx, vy = cx - dx, cy - dy

            dot = ux * vx + uy * vy
            mag_u = math.sqrt(ux * ux + uy * uy)
            mag_v = math.sqrt(vx * vx + vy * vy)

            if mag_u < 1e-12 or mag_v < 1e-12:
                continue

            cos_val = max(-1.0, min(1.0, dot / (mag_u * mag_v)))
            angle = math.degrees(math.acos(cos_val))

            # Normalize to 0-90 range (transmission angle is the acute version)
            if angle > 90:
                angle = 180 - angle
            min_angle = min(min_angle, angle)

    except (UnbuildableError, ValueError):
        return 0.0

    return min_angle


def compute_link_ratio(linkage: Linkage) -> float:
    """Compute max/min link length ratio.

    Examines all constrained distances in the linkage (crank radius,
    dyad distances, etc.) and returns the ratio of longest to shortest.

    Args:
        linkage: Linkage to evaluate.

    Returns:
        Ratio >= 1.0 (lower is better). Returns inf if any length is 0.
    """
    from .._compat import get_parts

    lengths: list[float] = []

    for joint in get_parts(linkage):
        # Crank: radius to the ground anchor
        r = getattr(joint, "radius", None) or getattr(joint, "r", None)
        if r is not None and r > 0:
            lengths.append(r)

        # Dyad distances to both anchors
        d1 = getattr(joint, "distance1", None) or getattr(joint, "r0", None)
        if d1 is not None and d1 > 0:
            lengths.append(d1)
        d2 = getattr(joint, "distance2", None) or getattr(joint, "r1", None)
        if d2 is not None and d2 > 0:
            lengths.append(d2)

        # Fixed dyad: explicit distance
        d = getattr(joint, "distance", None)
        if d is not None and isinstance(d, (int, float)) and d > 0:
            lengths.append(d)

    if len(lengths) < 2:
        return 1.0

    min_len = min(lengths)
    max_len = max(lengths)

    if min_len < 1e-12:
        return float("inf")

    return max_len / min_len


def compute_compactness(linkage: Linkage) -> float:
    """Compute bounding box area of the mechanism trajectory.

    Simulates the linkage and computes the bounding box of all
    joint positions over the full cycle.

    Args:
        linkage: Linkage to evaluate.

    Returns:
        Bounding box area (width * height).
    """
    from ..exceptions import UnbuildableError

    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    try:
        for positions in linkage.step():
            for px, py in positions:
                if px is not None and py is not None:
                    x_min = min(x_min, px)
                    x_max = max(x_max, px)
                    y_min = min(y_min, py)
                    y_max = max(y_max, py)
    except UnbuildableError:
        return float("inf")

    if x_min == float("inf"):
        return float("inf")

    return (x_max - x_min) * (y_max - y_min)


def score_solution(
    metrics: QualityMetrics,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted overall score from quality metrics.

    Default weights prioritize path accuracy, then transmission angle,
    then link ratio, then simplicity, then compactness.

    Args:
        metrics: Quality metrics to score.
        weights: Optional custom weight dict. Keys:
            "accuracy", "transmission", "link_ratio",
            "compactness", "simplicity", "grashof".

    Returns:
        Overall score (lower is better).
    """
    w = weights or {
        "accuracy": 0.40,
        "transmission": 0.25,
        "link_ratio": 0.10,
        "compactness": 0.05,
        "simplicity": 0.10,
        "grashof": 0.10,
    }

    # Normalize metrics to [0, 1] scale where 0 = best
    # Path accuracy: already lower=better, use directly (clamp to [0, 10])
    norm_accuracy = min(metrics.path_accuracy, 10.0) / 10.0

    # Transmission angle: higher=better, invert (90 is perfect)
    norm_transmission = 1.0 - min(metrics.min_transmission_angle, 90.0) / 90.0

    # Link ratio: lower=better (1.0 is ideal, 10+ is bad)
    norm_ratio = min(max(metrics.link_ratio - 1.0, 0.0), 9.0) / 9.0

    # Compactness: normalize relative (0-1 scale is hard without reference)
    # Use log scale: 1 = area of 1, 0 = area of ~0
    if metrics.compactness > 0 and metrics.compactness != float("inf"):
        norm_compact = min(math.log1p(metrics.compactness) / 10.0, 1.0)
    else:
        norm_compact = 1.0

    # Simplicity: fewer links = better. 4 is best, 8 is worst in catalog.
    norm_simplicity = (metrics.num_links - 4) / 4.0

    # Grashof bonus: 0 if Grashof, penalty if not
    norm_grashof = 0.0 if metrics.is_grashof else 1.0

    return (
        w.get("accuracy", 0.4) * norm_accuracy
        + w.get("transmission", 0.25) * norm_transmission
        + w.get("link_ratio", 0.1) * norm_ratio
        + w.get("compactness", 0.05) * norm_compact
        + w.get("simplicity", 0.1) * norm_simplicity
        + w.get("grashof", 0.1) * norm_grashof
    )


def _check_grashof_driver(linkage: Linkage) -> bool:
    """Check if the driving four-bar loop satisfies Grashof criterion."""
    from .._compat import get_parts
    from .utils import is_grashof

    parts = get_parts(linkage)

    crank = None
    revolute = None
    grounds: list[Any] = []

    for part in parts:
        name = type(part).__name__
        if name == "Crank" and crank is None:
            crank = part
        elif name in ("RRRDyad", "Revolute") and revolute is None:
            revolute = part
        elif name in ("Ground", "Static", "_StaticBase"):
            grounds.append(part)

    if crank is None or revolute is None or len(grounds) < 2:
        return False

    crank_len = getattr(crank, "radius", None) or getattr(crank, "r", None)
    coupler_len = getattr(revolute, "distance1", None) or getattr(revolute, "r0", None)
    rocker_len = getattr(revolute, "distance2", None) or getattr(revolute, "r1", None)

    if crank_len is None or coupler_len is None or rocker_len is None:
        return False

    ax = getattr(grounds[0], "x", 0.0) or 0.0
    ay = getattr(grounds[0], "y", 0.0) or 0.0
    dx = getattr(grounds[1], "x", 0.0) or 0.0
    dy = getattr(grounds[1], "y", 0.0) or 0.0
    ground_len = math.sqrt((dx - ax) ** 2 + (dy - ay) ** 2)

    if ground_len < 1e-12:
        return False

    return is_grashof(crank_len, coupler_len, rocker_len, ground_len)
