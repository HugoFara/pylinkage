"""Generate the static four-bar path-synthesis figure used in paper.md.

The figure shows the four precision points from the README example and the
best four-bar linkage whose coupler curve passes through them. Run::

    python docs/generate_paper_figure.py

to refresh ``docs/assets/paper_fourbar_synthesis.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pylinkage.synthesis import path_generation

OUTPUT_PATH = Path(__file__).parent / "assets" / "paper_fourbar_synthesis.png"
TARGET_POINTS = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]
FULL_ROTATION_FRAMES = 240

GROUND_ARM_COLOR = "#2c3e50"   # crank + rocker (the arms anchored to ground)
COUPLER_COLOR = "#2980b9"      # floating coupler link carrying point P
CURVE_COLOR = "#e67e22"        # trajectory traced by the coupler point
TARGET_COLOR = "#c0392b"       # precision points the curve must visit
COUPLER_POINT_COLOR = "#e67e22"
GROUND_COLOR = "#1a1a1a"


def simulate(linkage) -> np.ndarray:
    period = linkage.get_rotation_period()
    coords = np.array(list(linkage.step(iterations=period, dt=1.0)), dtype=float)
    if coords.shape[0] > FULL_ROTATION_FRAMES:
        idx = np.linspace(0, coords.shape[0] - 1, FULL_ROTATION_FRAMES, dtype=int)
        coords = coords[idx]
    return coords


def pick_best(result) -> tuple[object, np.ndarray]:
    """Pick a solution that is both accurate *and* visually compact.

    The raw "closest to all four points" winner often has a rocker that
    swings far outside the target window, wasting canvas. We penalise
    solutions whose bounding box is much larger than the target span.
    """
    target_span = max(
        max(p[0] for p in TARGET_POINTS) - min(p[0] for p in TARGET_POINTS),
        max(p[1] for p in TARGET_POINTS) - min(p[1] for p in TARGET_POINTS),
    )

    scored: list[tuple[float, object, np.ndarray]] = []
    for sol in result.solutions:
        try:
            coords = simulate(sol)
        except Exception:
            continue
        if np.isnan(coords).any():
            continue
        P = coords[:, 4]
        path_error = sum(
            np.hypot(P[:, 0] - tx, P[:, 1] - ty).min() for tx, ty in TARGET_POINTS
        )
        bbox_span = max(
            coords[..., 0].max() - coords[..., 0].min(),
            coords[..., 1].max() - coords[..., 1].min(),
        )
        compactness_penalty = max(0.0, bbox_span / target_span - 1.5)
        score = path_error + 0.5 * compactness_penalty
        scored.append((score, sol, coords))

    if not scored:
        raise RuntimeError("no buildable solutions")
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][2]


def pick_pose(coords: np.ndarray) -> int:
    """Pick a pose where the coupler point P is near the middle target.

    Frame 0 often puts the coupler on top of a joint. Choosing the pose
    closest to one of the middle target points gives a cleaner figure
    where the full coupler triangle is visible.
    """
    target = np.array(TARGET_POINTS[1])
    P = coords[:, 4]
    d = np.hypot(P[:, 0] - target[0], P[:, 1] - target[1])
    return int(np.argmin(d))


def draw_linkage(ax, frame) -> None:
    A, D, B, C, P = frame
    # Crank A-B and rocker D-C share the "ground-arm" colour
    ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=GROUND_ARM_COLOR,
            lw=2.8, zorder=3)
    ax.plot([D[0], C[0]], [D[1], C[1]], "-", color=GROUND_ARM_COLOR,
            lw=2.8, zorder=3)
    # Floating coupler link + triangle to the traced point P
    ax.plot([B[0], C[0]], [B[1], C[1]], "-", color=COUPLER_COLOR,
            lw=2.8, zorder=3)
    ax.plot([B[0], P[0]], [B[1], P[1]], "-", color=COUPLER_COLOR,
            lw=1.9, alpha=0.9, zorder=3)
    ax.plot([C[0], P[0]], [C[1], P[1]], "-", color=COUPLER_COLOR,
            lw=1.9, alpha=0.9, zorder=3)
    # Ground pivots
    for gx, gy in (A, D):
        ax.scatter([gx], [gy], marker="^", s=170, color=GROUND_COLOR, zorder=5)
    # Pin joints coloured by which link "owns" them (B, C belong to coupler)
    for jx, jy in (B, C):
        ax.scatter([jx], [jy], s=55, color=COUPLER_COLOR,
                   edgecolors="white", linewidths=1.2, zorder=6)
    # Coupler point P — same hue as the trajectory it traces
    ax.scatter([P[0]], [P[1]], s=75, color=COUPLER_POINT_COLOR,
               edgecolors="white", linewidths=1.3, zorder=7)


def main() -> None:
    print("Synthesizing four-bar linkages through target points...")
    result = path_generation(
        TARGET_POINTS,
        n_orientation_samples=48,
        max_solutions=12,
        require_grashof=True,
    )
    print(f"  {len(result.solutions)} buildable solutions")

    linkage, coords = pick_best(result)
    coupler = coords[:, 4]

    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)
    fig.patch.set_facecolor("white")

    # Coupler curve traced over one revolution
    ax.plot(coupler[:, 0], coupler[:, 1], "-", color=CURVE_COLOR,
            lw=2.2, alpha=0.95, label="Coupler curve")

    # Target points
    tx = [p[0] for p in TARGET_POINTS]
    ty = [p[1] for p in TARGET_POINTS]
    ax.scatter(tx, ty, s=110, color=TARGET_COLOR, edgecolors="white",
               linewidths=1.6, zorder=8, label="Target points")
    for i, (x, y) in enumerate(TARGET_POINTS, 1):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(9, 7), fontsize=11, color=TARGET_COLOR,
                    weight="bold")

    pose_idx = pick_pose(coords)
    draw_linkage(ax, coords[pose_idx])

    # Invisible lines just to seed the legend with the linkage colours
    ax.plot([], [], "-", color=GROUND_ARM_COLOR, lw=2.8, label="Crank / rocker")
    ax.plot([], [], "-", color=COUPLER_COLOR, lw=2.8, label="Coupler link")

    # Frame
    all_x = np.concatenate([coupler[:, 0], coords[:, :, 0].ravel(), np.array(tx)])
    all_y = np.concatenate([coupler[:, 1], coords[:, :, 1].ravel(), np.array(ty)])
    pad = 0.15
    cx, cy = (all_x.min() + all_x.max()) / 2, (all_y.min() + all_y.max()) / 2
    half = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * (1 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
