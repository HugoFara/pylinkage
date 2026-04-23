"""Generate the multi-phase synthesis animation used in social posts.

Three phases:

1. Target points dropped onto the canvas one at a time.
2. Several candidate four-bar linkages flash by, each tracing its
   coupler curve in a faint color.
3. The chosen winner spins through a full rotation while its coupler
   point paints the final path.

Run ``python docs/generate_synthesis_animation.py`` to refresh
``docs/assets/synthesis_path_generation.gif``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from pylinkage.synthesis import path_generation

OUTPUT_PATH = Path(__file__).parent / "assets" / "synthesis_path_generation.gif"

TARGET_POINTS = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]

FPS = 20
PHASE1_FRAMES_PER_POINT = 10  # ~0.5s drop per point
PHASE1_HOLD = 14              # ~0.7s after the last point lands
PHASE2_FRAMES_PER_CAND = 18   # ~0.9s per candidate
PHASE3_HOLD_FRAMES = 12       # ~0.6s on the finished path
N_CANDIDATES = 4              # candidates flashed in phase 2

WINNER_COLOR = "#1f77b4"
ACCENT_COLOR = "#d62728"
GHOST_COLOR = "#9aa0a6"
TARGET_COLOR = "#d62728"
TRACE_COLOR = "#1f77b4"
BG_TRACE = "#c8d4e3"


FULL_ROTATION_FRAMES = 120  # how densely we sample one full revolution


def simulate_linkage(linkage):
    """Run one full revolution and return ``FULL_ROTATION_FRAMES`` samples."""
    period = linkage.get_rotation_period()
    coords = np.array(list(linkage.step(iterations=period, dt=1.0)), dtype=float)
    if coords.shape[0] > FULL_ROTATION_FRAMES:
        idx = np.linspace(0, coords.shape[0] - 1, FULL_ROTATION_FRAMES,
                          dtype=int)
        coords = coords[idx]
    return coords


def _within_window(coords: np.ndarray, points, slack: float = 1.5) -> bool:
    """Reject candidates whose anchors sit far outside the target window."""
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    xrange = xs.max() - xs.min()
    yrange = ys.max() - ys.min()
    span = max(xrange, yrange)
    xlim = (xs.min() - slack * span, xs.max() + slack * span)
    ylim = (ys.min() - slack * span, ys.max() + slack * span)
    cx, cy = coords[..., 0], coords[..., 1]
    return bool(
        (cx.min() >= xlim[0]) and (cx.max() <= xlim[1])
        and (cy.min() >= ylim[0]) and (cy.max() <= ylim[1])
    )


def pick_solutions(result, n_candidates: int, points):
    """Pick a winner plus a handful of well-framed candidates."""
    sols = list(result.solutions)
    if not sols:
        raise RuntimeError("path_generation returned no solutions")

    # Score every solution by "how close does the coupler curve pass to each
    # target point": smaller is better. Use the lowest-error one as winner.
    scored: list[tuple[float, object, np.ndarray]] = []
    for sol in sols:
        try:
            coords = simulate_linkage(sol)
        except Exception:
            continue
        if np.isnan(coords).any():
            continue
        if not _within_window(coords, points):
            continue
        P = coords[:, 4]
        error = 0.0
        for tx, ty in points:
            d = np.hypot(P[:, 0] - tx, P[:, 1] - ty).min()
            error += d
        scored.append((error, sol, coords))

    if not scored:
        raise RuntimeError("no well-framed buildable solutions to animate")

    scored.sort(key=lambda t: t[0])
    winner = (scored[0][1], scored[0][2])
    candidates = [(s, c) for _, s, c in scored[1: 1 + n_candidates]]
    return winner, candidates


def setup_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def compute_window(points, winner_coords, candidates):
    """Square-padded window covering targets, winner, and candidates."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pools_x = [np.array(xs), winner_coords[..., 0].ravel()]
    pools_y = [np.array(ys), winner_coords[..., 1].ravel()]
    for _, c in candidates:
        pools_x.append(c[..., 0].ravel())
        pools_y.append(c[..., 1].ravel())
    all_x = np.concatenate(pools_x)
    all_y = np.concatenate(pools_y)

    pad = 0.12
    x_lo, x_hi = all_x.min(), all_x.max()
    y_lo, y_hi = all_y.min(), all_y.max()
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    half = max(x_hi - x_lo, y_hi - y_lo) / 2 * (1 + pad)
    return ((cx - half, cx + half), (cy - half, cy + half))


def draw_target_points(ax, points, n_visible, pulse_frame=None):
    """Plot the first ``n_visible`` target points; pulse the newest."""
    for i, (x, y) in enumerate(points[:n_visible]):
        ax.scatter([x], [y], s=70, color=TARGET_COLOR, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=10, color=TARGET_COLOR,
                    weight="bold")
    if pulse_frame is not None and n_visible >= 1:
        x, y = points[n_visible - 1]
        radius = 0.05 + 0.20 * pulse_frame
        alpha = max(0.0, 1.0 - pulse_frame)
        ax.add_patch(Circle((x, y), radius, color=TARGET_COLOR,
                            alpha=alpha * 0.5, fill=False, lw=2))


def draw_linkage(ax, frame, color=WINNER_COLOR, alpha=1.0, lw=2.5,
                 show_joints=True):
    """Draw a four-bar from one positions snapshot.

    ``frame`` is the components row from ``simulate_linkage``: A, D, B, C, P.
    """
    A, D, B, C, P = frame
    # Crank A-B
    ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=color, alpha=alpha, lw=lw)
    # Coupler triangle B-C and B-P, C-P
    ax.plot([B[0], C[0]], [B[1], C[1]], "-", color=color, alpha=alpha, lw=lw)
    ax.plot([B[0], P[0]], [B[1], P[1]], "-", color=color, alpha=alpha * 0.8,
            lw=lw * 0.75)
    ax.plot([C[0], P[0]], [C[1], P[1]], "-", color=color, alpha=alpha * 0.8,
            lw=lw * 0.75)
    # Rocker D-C
    ax.plot([D[0], C[0]], [D[1], C[1]], "-", color=color, alpha=alpha, lw=lw)

    if show_joints:
        # Ground triangles
        for gx, gy in (A, D):
            ax.scatter([gx], [gy], marker="^", s=110, color="#444",
                       alpha=alpha, zorder=4)
        # Pin joints
        for jx, jy in (B, C):
            ax.scatter([jx], [jy], s=40, color=color, alpha=alpha,
                       edgecolors="white", linewidths=1.0, zorder=4)
        # Coupler point P
        ax.scatter([P[0]], [P[1]], s=60, color=ACCENT_COLOR, alpha=alpha,
                   edgecolors="white", linewidths=1.2, zorder=6)


def render_phase1(ax, points, xlim, ylim, frame_idx_in_phase):
    """Drop points one by one with a pulse on the newest."""
    fpp = PHASE1_FRAMES_PER_POINT
    n_visible = min(len(points), frame_idx_in_phase // fpp + 1)
    pulse = (frame_idx_in_phase % fpp) / fpp
    setup_axes(ax, xlim, ylim)
    ax.set_title("Step 1 — sketch the path you need",
                 fontsize=13, color="#222")
    if n_visible >= 2:
        xs = [p[0] for p in points[:n_visible]]
        ys = [p[1] for p in points[:n_visible]]
        ax.plot(xs, ys, ":", color=TARGET_COLOR, alpha=0.35, lw=1.5)
    draw_target_points(ax, points, n_visible, pulse_frame=pulse)


def render_phase1_hold(ax, points, xlim, ylim):
    setup_axes(ax, xlim, ylim)
    ax.set_title("Step 1 — sketch the path you need",
                 fontsize=13, color="#222")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, ":", color=TARGET_COLOR, alpha=0.35, lw=1.5)
    draw_target_points(ax, points, len(points))


def render_phase2(ax, points, xlim, ylim, candidates, cand_idx, sub_frame):
    setup_axes(ax, xlim, ylim)
    ax.set_title(f"Step 2 — synthesis tries candidate "
                 f"{cand_idx + 1}/{len(candidates)}",
                 fontsize=13, color="#222")
    draw_target_points(ax, points, len(points))

    sol, coords = candidates[cand_idx]
    n = coords.shape[0]
    # Show the full coupler curve (faint) immediately so the audience can
    # see this candidate's "reach" — the linkage then sweeps along it.
    P_xy_full = coords[:, 4]
    ax.plot(P_xy_full[:, 0], P_xy_full[:, 1], "--",
            color=GHOST_COLOR, alpha=0.55, lw=1.4)
    progress = (sub_frame + 1) / PHASE2_FRAMES_PER_CAND
    snap_idx = int((n - 1) * progress) % n
    draw_linkage(ax, coords[snap_idx], color=GHOST_COLOR, alpha=0.95,
                 lw=2.0, show_joints=True)


def render_phase3(ax, points, xlim, ylim, winner_coords, frame_in_phase,
                  total):
    setup_axes(ax, xlim, ylim)
    ax.set_title("Step 3 — winning linkage traces the path",
                 fontsize=13, color="#222")
    draw_target_points(ax, points, len(points))

    n = winner_coords.shape[0]
    P_full = winner_coords[:, 4]
    # Faint preview of the full coupler curve (where it WILL go)
    ax.plot(P_full[:, 0], P_full[:, 1], ":", color=BG_TRACE, alpha=0.7,
            lw=1.3)

    upto = max(2, int(n * (frame_in_phase + 1) / total))
    P_xy = winner_coords[:upto, 4]
    ax.plot(P_xy[:, 0], P_xy[:, 1], "-", color=TRACE_COLOR, alpha=0.95,
            lw=2.6)

    snap = winner_coords[(upto - 1) % n]
    draw_linkage(ax, snap, color=WINNER_COLOR, alpha=1.0, lw=2.6)


def render_phase3_hold(ax, points, xlim, ylim, winner_coords):
    setup_axes(ax, xlim, ylim)
    ax.set_title("Step 3 — winning linkage traces the path",
                 fontsize=13, color="#222")
    draw_target_points(ax, points, len(points))
    P_xy = winner_coords[:, 4]
    ax.plot(P_xy[:, 0], P_xy[:, 1], "-", color=TRACE_COLOR, alpha=0.95,
            lw=2.6)
    draw_linkage(ax, winner_coords[0], color=WINNER_COLOR, alpha=1.0, lw=2.6)


def main() -> None:
    print("Synthesizing four-bar linkages...")
    result = path_generation(
        TARGET_POINTS,
        n_orientation_samples=48,
        max_solutions=8,
        require_grashof=True,
    )
    print(f"  found {len(result.solutions)} buildable solutions")

    (winner_sol, winner_coords), candidate_pairs = pick_solutions(
        result, N_CANDIDATES, TARGET_POINTS
    )
    print(f"  using winner + {len(candidate_pairs)} candidates")

    xlim, ylim = compute_window(TARGET_POINTS, winner_coords, candidate_pairs)

    n_phase1 = PHASE1_FRAMES_PER_POINT * len(TARGET_POINTS) + PHASE1_HOLD
    n_phase2 = PHASE2_FRAMES_PER_CAND * len(candidate_pairs)
    n_phase3 = winner_coords.shape[0]
    total_frames = n_phase1 + n_phase2 + n_phase3 + PHASE3_HOLD_FRAMES
    print(f"  timeline: {n_phase1} + {n_phase2} + {n_phase3} + "
          f"{PHASE3_HOLD_FRAMES} = {total_frames} frames @ {FPS} fps "
          f"(~{total_frames / FPS:.1f}s)")

    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=110)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.04)

    def update(i: int):
        ax.clear()
        if i < PHASE1_FRAMES_PER_POINT * len(TARGET_POINTS):
            render_phase1(ax, TARGET_POINTS, xlim, ylim, i)
        elif i < n_phase1:
            render_phase1_hold(ax, TARGET_POINTS, xlim, ylim)
        elif i < n_phase1 + n_phase2:
            j = i - n_phase1
            cand_idx = j // PHASE2_FRAMES_PER_CAND
            sub = j % PHASE2_FRAMES_PER_CAND
            render_phase2(ax, TARGET_POINTS, xlim, ylim, candidate_pairs,
                          cand_idx, sub)
        elif i < n_phase1 + n_phase2 + n_phase3:
            k = i - n_phase1 - n_phase2
            render_phase3(ax, TARGET_POINTS, xlim, ylim, winner_coords, k,
                          n_phase3)
        else:
            render_phase3_hold(ax, TARGET_POINTS, xlim, ylim, winner_coords)
        return []

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / FPS,
                         blit=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUTPUT_PATH} ...")
    writer = PillowWriter(fps=FPS)
    anim.save(OUTPUT_PATH, writer=writer)
    print("Done.")


if __name__ == "__main__":
    main()
