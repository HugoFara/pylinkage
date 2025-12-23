"""
Visualization of velocity and acceleration vectors.

This module provides functions to visualize kinematic quantities
(velocity and acceleration) as vector arrows overlaid on linkage diagrams.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from numpy.typing import NDArray

    from ..linkage.linkage import Linkage


def plot_velocity_vectors(
    linkage: "Linkage",
    axis: "Axes",
    positions: "NDArray[np.float64] | Sequence[tuple[float, float]]",
    velocities: "NDArray[np.float64] | Sequence[tuple[float, float]]",
    *,
    scale: float = 0.1,
    color: str = "blue",
    width: float = 0.005,
    label: str = "Velocity",
    skip_static: bool = True,
) -> "Quiver":
    """Plot velocity vectors as arrows at joint positions.

    Args:
        linkage: The linkage being visualized.
        axis: Matplotlib axes to draw on.
        positions: Joint positions, shape (n_joints, 2) or list of (x, y).
        velocities: Joint velocities, shape (n_joints, 2) or list of (vx, vy).
        scale: Scaling factor for arrow length. Smaller = longer arrows.
        color: Arrow color.
        width: Arrow shaft width as fraction of plot width.
        label: Legend label for the arrows.
        skip_static: Whether to skip drawing arrows for static joints.

    Returns:
        The Quiver object for further customization.
    """
    from ..joints import Static

    positions = np.asarray(positions)
    velocities = np.asarray(velocities)

    # Filter out static joints if requested
    if skip_static:
        mask = [not isinstance(j, Static) for j in linkage.joints]
        positions = positions[mask]
        velocities = velocities[mask]

    # Filter out NaN velocities
    valid = ~np.isnan(velocities).any(axis=1)
    positions = positions[valid]
    velocities = velocities[valid]

    if len(positions) == 0:
        # Return empty quiver if no valid vectors
        return axis.quiver([], [], [], [], scale=scale, color=color, width=width)

    quiver = axis.quiver(
        positions[:, 0],
        positions[:, 1],
        velocities[:, 0],
        velocities[:, 1],
        scale=scale,
        scale_units="xy",
        angles="xy",
        color=color,
        width=width,
        label=label,
    )
    return quiver


def plot_acceleration_vectors(
    linkage: "Linkage",
    axis: "Axes",
    positions: "NDArray[np.float64] | Sequence[tuple[float, float]]",
    accelerations: "NDArray[np.float64] | Sequence[tuple[float, float]]",
    *,
    scale: float = 0.01,
    color: str = "red",
    width: float = 0.004,
    label: str = "Acceleration",
    skip_static: bool = True,
) -> "Quiver":
    """Plot acceleration vectors as arrows at joint positions.

    Args:
        linkage: The linkage being visualized.
        axis: Matplotlib axes to draw on.
        positions: Joint positions, shape (n_joints, 2) or list of (x, y).
        accelerations: Joint accelerations, shape (n_joints, 2) or list of (ax, ay).
        scale: Scaling factor for arrow length. Smaller = longer arrows.
        color: Arrow color.
        width: Arrow shaft width as fraction of plot width.
        label: Legend label for the arrows.
        skip_static: Whether to skip drawing arrows for static joints.

    Returns:
        The Quiver object for further customization.
    """
    from ..joints import Static

    positions = np.asarray(positions)
    accelerations = np.asarray(accelerations)

    # Filter out static joints if requested
    if skip_static:
        mask = [not isinstance(j, Static) for j in linkage.joints]
        positions = positions[mask]
        accelerations = accelerations[mask]

    # Filter out NaN accelerations
    valid = ~np.isnan(accelerations).any(axis=1)
    positions = positions[valid]
    accelerations = accelerations[valid]

    if len(positions) == 0:
        return axis.quiver([], [], [], [], scale=scale, color=color, width=width)

    quiver = axis.quiver(
        positions[:, 0],
        positions[:, 1],
        accelerations[:, 0],
        accelerations[:, 1],
        scale=scale,
        scale_units="xy",
        angles="xy",
        color=color,
        width=width,
        label=label,
    )
    return quiver


def plot_kinematics_frame(
    linkage: "Linkage",
    axis: "Axes",
    positions: "NDArray[np.float64]",
    velocities: "NDArray[np.float64] | None" = None,
    accelerations: "NDArray[np.float64] | None" = None,
    *,
    velocity_scale: float = 0.1,
    acceleration_scale: float = 0.01,
    show_velocity: bool = True,
    show_acceleration: bool = False,
) -> None:
    """Plot a single frame of linkage with kinematic vectors.

    Args:
        linkage: The linkage being visualized.
        axis: Matplotlib axes to draw on.
        positions: Joint positions for this frame, shape (n_joints, 2).
        velocities: Joint velocities, shape (n_joints, 2). Optional.
        accelerations: Joint accelerations, shape (n_joints, 2). Optional.
        velocity_scale: Scaling factor for velocity arrows.
        acceleration_scale: Scaling factor for acceleration arrows.
        show_velocity: Whether to show velocity vectors.
        show_acceleration: Whether to show acceleration vectors.
    """
    from .static import plot_static_linkage

    # Convert single frame to loci format for plot_static_linkage
    loci = [tuple((float(positions[i, 0]), float(positions[i, 1]))
                  for i in range(len(linkage.joints)))]
    plot_static_linkage(linkage, axis, loci, show_legend=False)

    if show_velocity and velocities is not None:
        plot_velocity_vectors(
            linkage, axis, positions, velocities, scale=velocity_scale
        )

    if show_acceleration and accelerations is not None:
        plot_acceleration_vectors(
            linkage, axis, positions, accelerations, scale=acceleration_scale
        )


def show_kinematics(
    linkage: "Linkage",
    frame_index: int = 0,
    *,
    show_velocity: bool = True,
    show_acceleration: bool = False,
    velocity_scale: float | None = None,
    acceleration_scale: float | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> "Figure":
    """Display linkage with velocity and/or acceleration vectors.

    Runs simulation with kinematics computation and displays the result
    at a specific frame.

    Args:
        linkage: The linkage to visualize.
        frame_index: Which frame to display (0 = initial position).
        show_velocity: Whether to show velocity vectors.
        show_acceleration: Whether to show acceleration vectors.
        velocity_scale: Arrow scale for velocities. Auto-computed if None.
        acceleration_scale: Arrow scale for accelerations. Auto-computed if None.
        figsize: Figure size (width, height) in inches.
        title: Figure title.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> linkage.set_input_velocity(crank, omega=10.0)
        >>> fig = show_kinematics(linkage, frame_index=25, show_velocity=True)
    """
    import matplotlib.pyplot as plt

    from ..linkage.analysis import movement_bounding_box

    # Check that omega is set on at least one crank
    from ..joints import Crank
    has_omega = any(
        isinstance(j, Crank) and j.omega is not None and j.omega != 0
        for j in linkage.joints
    )
    if not has_omega:
        raise ValueError(
            "No crank has omega set. Use linkage.set_input_velocity(crank, omega=...) "
            "before calling show_kinematics()."
        )

    # Run simulation with kinematics
    positions, velocities, accelerations = linkage.step_fast_with_kinematics()

    # Get the specific frame
    n_frames = positions.shape[0]
    if frame_index < 0 or frame_index >= n_frames:
        raise ValueError(f"frame_index must be in [0, {n_frames}), got {frame_index}")

    pos = positions[frame_index]
    vel = velocities[frame_index]

    # Auto-compute scales based on data ranges
    if velocity_scale is None:
        vel_mag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
        max_vel = np.nanmax(vel_mag) if np.any(~np.isnan(vel_mag)) else 1.0
        # Scale so max velocity arrow is about 1/5 of linkage size
        bbox = movement_bounding_box(
            [tuple((float(pos[i, 0]), float(pos[i, 1]))
                   for i in range(len(linkage.joints)))]
        )
        linkage_size = max(bbox[2] - bbox[0], bbox[1] - bbox[3])
        velocity_scale = max_vel / (linkage_size * 0.2) if max_vel > 0 else 1.0

    if acceleration_scale is None:
        acceleration_scale = velocity_scale * 10  # Accelerations are typically larger

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to loci format
    loci = [tuple((float(positions[i, j, 0]), float(positions[i, j, 1]))
                  for j in range(len(linkage.joints)))
            for i in range(n_frames)]

    # Plot static linkage with all loci
    from .static import plot_static_linkage
    plot_static_linkage(linkage, ax, loci, show_legend=True)

    # Highlight current position
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=100, c='yellow', edgecolors='black', zorder=10, label='Current'
    )

    # Plot velocity vectors
    if show_velocity:
        plot_velocity_vectors(
            linkage, ax, pos, vel, scale=velocity_scale, label="Velocity"
        )

    # Plot acceleration vectors (would need to extend simulation)
    if show_acceleration:
        # For now, acceleration visualization requires extending the API
        pass

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(title or f"Kinematics at frame {frame_index}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def animate_kinematics(
    linkage: "Linkage",
    *,
    show_velocity: bool = True,
    show_acceleration: bool = False,
    velocity_scale: float | None = None,
    fps: int = 24,
    duration: float = 5.0,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    save_path: str | None = None,
) -> "Figure":
    """Create an animated visualization with velocity vectors.

    Args:
        linkage: The linkage to visualize.
        show_velocity: Whether to show velocity vectors.
        show_acceleration: Whether to show acceleration vectors.
        velocity_scale: Arrow scale for velocities. Auto-computed if None.
        fps: Frames per second.
        duration: Animation duration in seconds.
        figsize: Figure size (width, height) in inches.
        title: Figure title.
        save_path: If provided, save animation to this path (e.g., "animation.gif").

    Returns:
        The matplotlib Figure object.

    Example:
        >>> linkage.set_input_velocity(crank, omega=10.0)
        >>> fig = animate_kinematics(linkage, show_velocity=True, save_path="vel.gif")
    """
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt

    from ..joints import Crank, Static
    from ..linkage.analysis import movement_bounding_box
    from .core import _get_color

    # Check that omega is set
    has_omega = any(
        isinstance(j, Crank) and j.omega is not None and j.omega != 0
        for j in linkage.joints
    )
    if not has_omega:
        raise ValueError(
            "No crank has omega set. Use linkage.set_input_velocity(crank, omega=...) "
            "before calling animate_kinematics()."
        )

    # Run simulation
    positions, velocities, _ = linkage.step_fast_with_kinematics()
    n_frames = positions.shape[0]

    # Auto-compute velocity scale
    if velocity_scale is None:
        vel_mag = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2)
        max_vel = np.nanmax(vel_mag) if np.any(~np.isnan(vel_mag)) else 1.0
        loci = [tuple((float(positions[i, j, 0]), float(positions[i, j, 1]))
                      for j in range(len(linkage.joints)))
                for i in range(n_frames)]
        bbox = movement_bounding_box(loci)
        linkage_size = max(bbox[2] - bbox[0], bbox[1] - bbox[3])
        velocity_scale = max_vel / (linkage_size * 0.15) if max_vel > 0 else 1.0

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Set up axes limits
    loci = [tuple((float(positions[i, j, 0]), float(positions[i, j, 1]))
                  for j in range(len(linkage.joints)))
            for i in range(n_frames)]
    bbox = movement_bounding_box(loci)
    padding = max(bbox[2] - bbox[0], bbox[1] - bbox[3]) * 0.2
    for ax in (ax1, ax2):
        ax.set_xlim(bbox[3] - padding, bbox[1] + padding)
        ax.set_ylim(bbox[0] - padding, bbox[2] + padding)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Left plot: static with all loci
    from .static import plot_static_linkage
    plot_static_linkage(linkage, ax1, loci, show_legend=True)
    ax1.set_title("Trajectory paths")

    # Right plot: animated with velocity vectors
    ax2.set_title(title or "Kinematics Animation")

    # Initialize artists for animation
    link_lines = []
    for joint in linkage.joints:
        for parent in (joint.joint0, joint.joint1):
            if parent is not None:
                line, = ax2.plot([], [], c=_get_color(joint), linewidth=2)
                link_lines.append((joint, parent, line))

    # Joint markers
    joint_scatter = ax2.scatter([], [], s=80, c='white', edgecolors='black', zorder=5)

    # Velocity quiver (will be updated each frame)
    # Initialize with empty data
    non_static_mask = [not isinstance(j, Static) for j in linkage.joints]
    n_non_static = sum(non_static_mask)
    quiver = ax2.quiver(
        np.zeros(n_non_static), np.zeros(n_non_static),
        np.zeros(n_non_static), np.zeros(n_non_static),
        scale=velocity_scale,
        scale_units="xy",
        angles="xy",
        color="blue",
        width=0.008,
        label="Velocity",
    )

    ax2.legend(loc='upper right')

    def update(frame_idx: int) -> "Iterable[Artist]":
        """Update function for animation."""
        pos = positions[frame_idx]
        vel = velocities[frame_idx]

        # Update links
        for joint, parent, line in link_lines:
            j_idx = list(linkage.joints).index(joint)
            if isinstance(parent, Static):
                p_pos = parent.coord()
            else:
                p_idx = list(linkage.joints).index(parent)
                p_pos = (pos[p_idx, 0], pos[p_idx, 1])
            line.set_data([p_pos[0], pos[j_idx, 0]], [p_pos[1], pos[j_idx, 1]])

        # Update joint positions
        joint_scatter.set_offsets(pos)

        # Update velocity quiver
        if show_velocity:
            non_static_pos = pos[non_static_mask]
            non_static_vel = vel[non_static_mask]
            # Filter out NaNs
            valid = ~np.isnan(non_static_vel).any(axis=1)
            if np.any(valid):
                quiver.set_offsets(non_static_pos[valid])
                quiver.set_UVC(non_static_vel[valid, 0], non_static_vel[valid, 1])
            else:
                quiver.set_offsets(np.array([]).reshape(0, 2))
                quiver.set_UVC([], [])

        return [line for _, _, line in link_lines] + [joint_scatter, quiver]

    # Create animation
    frames_to_show = min(n_frames, int(fps * duration))
    frame_indices = np.linspace(0, n_frames - 1, frames_to_show, dtype=int)

    animation = anim.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    if save_path:
        if save_path.endswith('.gif'):
            animation.save(save_path, writer='pillow', fps=fps)
        else:
            animation.save(save_path, fps=fps)

    plt.tight_layout()
    if plt.isinteractive() or plt.get_backend() not in ("agg", "Agg"):
        plt.show(block=False)
        plt.pause(duration)

    return fig
