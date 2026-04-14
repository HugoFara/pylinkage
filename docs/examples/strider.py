"""Kinematic Strider linkage, a type of walking linkage.

The original linkage can be found at
https://www.diywalkers.com/strider-linkage-plans.html
"""

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

import pylinkage as pl
from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.simulation import Linkage

# --- Simulation parameters --------------------------------------------------
# Number of points for a crank complete turn
LAP_POINTS = 10
# Time (in seconds) for a crank revolution
LAP_PER_SECOND = 100

# Design parameters — can change without changing the connectivity.
DIM_NAMES = ("triangle", "aperture", "femur", "rockerL", "rockerS", "f", "tibia", "phi")

DIMENSIONS = (
    # AB = AB_p distance (triangle)
    2,
    # aperture
    np.pi / 4,
    # femur (3 for higher steps, 2 for standard, 1.8 is good enough)
    1.8,
    # rockerL
    2.6,
    # rockerS
    1.4,
    # phi
    np.pi + 0.2,
    # tibia
    2.5,
    # f
    1.8,
)

# Limits for the design parameters, passed to the optimizer.
BOUNDS = ((0, 0, 0, 0, 0, 0, 0, 0), (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6))

# Initial coordinates for (A, Y, B, B_p, C, D, E, F, G, H, I).
INIT_COORD = (
    (0, 0),
    (0, 1),
    (1.41, 1.41),
    (-1.41, 1.41),
    (0, -1),
    (-2.25, 0),
    (2.25, 0),
    (-1.4, -1.2),
    (1.4, -1.2),
    (-2.7, -2.7),
    (2.7, -2.7),
)


def param2dimensions(param=DIMENSIONS, *, flat: bool = False):
    """Expand the symmetric design vector into a full constraint list.

    The mechanism is symmetric so only half the parameters are exposed;
    this helper mirrors them back out in the order each component
    consumes its constraints.

    :param param: Short form of the design vector.
    :param flat: If ``True`` return a flat tuple of all constraint
        values; if ``False`` return a per-component tuple-of-tuples.
    """
    out = (
        # Static joints (A, Y) carry no constraints
        (),
        (),
        # B, B_p — (distance, angle) on FixedDyad
        (param[0], -param[1]),
        (param[0], param[1]),
        # Crank C — (radius,)
        (1,),
        # D and E — RRR dyads (distance1, distance2)
        (param[2], param[3]),
        (param[2], param[3]),
        # F and G — FixedDyads (distance, angle)
        (param[4], -param[5]),
        (param[4], param[5]),
        # H and I — RRR dyads (distance1, distance2)
        (param[6], param[7]),
        (param[6], param[7]),
    )
    if not flat:
        return out
    flat_dims: list[float] = []
    for constraint in out:
        flat_dims.extend(constraint)
    return tuple(flat_dims)


def _constraints_by_joint(dimensions):
    """Normalize *dimensions* into a per-joint tuple-of-tuples."""
    if dimensions and isinstance(dimensions[0], tuple):
        return dimensions
    return param2dimensions(dimensions)


def complete_strider(constraints, prev):
    """Build a Strider linkage from a per-joint constraint tuple.

    :param constraints: Per-joint constraint values (tuple-of-tuples)
        or a flat design vector.
    :param prev: Initial coordinates for each component
        (tuple of ``(x, y)`` pairs, in the same order as
        ``INIT_COORD``).

    :return: A strider ``Linkage``.
    """
    c = _constraints_by_joint(constraints)
    # c[0] and c[1] are empty — A and Y contribute no constraints.
    B_d, B_a = c[2]
    Bp_d, Bp_a = c[3]
    (crank_r,) = c[4]
    D_d1, D_d2 = c[5]
    E_d1, E_d2 = c[6]
    F_d, F_a = c[7]
    G_d, G_a = c[8]
    H_d1, H_d2 = c[9]
    I_d1, I_d2 = c[10]

    parts: list = []

    # Mechanism frame — A is the origin, Y is the vertical reference.
    A = Ground(*prev[0], name="A")
    Y = Ground(*prev[1], name="Point (0, 1)")
    parts += [A, Y]

    # Frame-side fixed dyads
    B = FixedDyad(anchor1=A, anchor2=Y, distance=B_d, angle=B_a, name="Frame right (B)")
    B_p = FixedDyad(
        anchor1=A, anchor2=Y, distance=Bp_d, angle=Bp_a, name="Frame left (B_p)",
    )
    parts += [B, B_p]

    # Crank
    C = Crank(
        anchor=A,
        radius=crank_r,
        angular_velocity=-2 * np.pi / LAP_POINTS,
        name="Crank link (C)",
    )
    parts.append(C)

    # Knee links
    D = RRRDyad(
        anchor1=B_p,
        anchor2=C.output,
        distance1=D_d1,
        distance2=D_d2,
        name="Left knee link (D)",
    )
    E = RRRDyad(
        anchor1=B,
        anchor2=C.output,
        distance1=E_d1,
        distance2=E_d2,
        name="Right knee link (E)",
    )
    parts += [D, E]

    # Ankle fixed dyads (F fixed to C/E, G fixed to C/D)
    F = FixedDyad(
        anchor1=C.output, anchor2=E, distance=F_d, angle=F_a, name="Left ankle link (F)",
    )
    G = FixedDyad(
        anchor1=C.output, anchor2=D, distance=G_d, angle=G_a, name="Right ankle link (G)",
    )
    parts += [F, G]

    # Feet
    H = RRRDyad(
        anchor1=D, anchor2=F, distance1=H_d1, distance2=H_d2, name="Left foot (H)",
    )
    I_ = RRRDyad(
        anchor1=E, anchor2=G, distance1=I_d1, distance2=I_d2, name="Right foot (I)",
    )
    parts += [H, I_]

    strider = Linkage(parts, name="Strider")
    # Initial coordinates are in the same order as INIT_COORD.
    strider.set_coords(list(prev))
    return strider


def sym_stride_evaluator(linkage, dimensions, initial_positions):
    """Score a set of dimensions by horizontal foot-stride length."""
    linkage.set_completely(
        list(param2dimensions(dimensions, flat=True)),
        list(initial_positions),
    )
    points = 12
    try:
        loci = tuple(
            map(tuple, linkage.step(iterations=points, dt=LAP_POINTS / points))
        )
    except pl.UnbuildableError:
        return 0
    # The foot (second-to-last component = H)
    foot_locus = tuple(x[-2] for x in loci)
    return max(k[0] for k in foot_locus) - min(k[0] for k in foot_locus)


def history_saver(evaluator, history, linkage, dims, pos):
    """Record each evaluation for animation/plotting."""
    score = evaluator(linkage, dims, pos)
    history.append((score, list(dims), pos))
    return score


def view_swarm_polar(linkage, dimensions=DIMENSIONS, save_each=0, n_agents=300, n_iterations=400):
    """Animate the swarm as a polar graph over the design parameters."""
    history: list = []
    ensemble = pl.particle_swarm_optimization(
        lambda *x: history_saver(sym_stride_evaluator, history, *x),
        linkage,
        center=dimensions,
        n_particles=n_agents,
        iters=n_iterations,
        bounds=BOUNDS,
        dimensions=len(dimensions),
    )
    best_score = ensemble[0].scores.get("score", 0.0)

    fig = plt.figure("Swarm in polar graph")
    fig.suptitle(f"Final best score: {-best_score:.2f}")
    formatted_history = [history[i : i + n_agents] for i in range(0, len(history), n_agents)]
    artists: list = []

    def init_polar_repr():
        ax = fig.add_subplot(111, projection="polar")
        artists.extend(ax.plot([], [], lw=0.5, animated=False)[0] for _ in range(n_agents))
        ax.set_rmax(7)
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, len(dimensions) + 1, endpoint=False), DIM_NAMES + ("score",)
        )
        artists.append(ax.text(1.9 * np.pi, 2, "", animated=True))
        return artists

    def repr_polar_swarm(current_swarm):
        t = np.linspace(0, 2 * np.pi, len(current_swarm[1][0][1]) + 2)[:-1]
        for line, agent in zip(artists, current_swarm[1], strict=False):
            line.set_data(t, agent[1] + [agent[0]])
        artists[-1].set_text(
            f"Best score: {max(x[0] for x in current_swarm[1]):.2f}\n"
            f"Iteration: {current_swarm[0]}"
        )
        return artists

    animation = anim.FuncAnimation(
        fig,
        func=repr_polar_swarm,
        frames=enumerate(formatted_history),
        init_func=init_polar_repr,
        blit=True,
        interval=400,
        repeat=True,
        save_count=(n_iterations - 1) * bool(save_each),
    )
    plt.show()
    if save_each:
        writer = anim.FFMpegWriter(
            fps=24,
            bitrate=1800,
            metadata={
                "title": "Particle swarm looking for R^8 in R application maximum",
                "comment": "Made with Python and Matplotlib",
                "description": "The swarm tries to find the best dimension "
                "set for the Strider legged mechanism",
            },
        )
        animation.save("Particle Swarm Optimization of Strider linkage.mp4", writer=writer)
    # Keep a reference so matplotlib doesn't garbage-collect the animation.
    _ = animation
    return ensemble


def view_swarm_tiled(linkage, dimensions=DIMENSIONS, save_each=0, n_agents=300, n_iterations=400):
    """Render the swarm as a grid of small linkage previews."""
    history: list = []
    ensemble = pl.particle_swarm_optimization(
        lambda *x: history_saver(sym_stride_evaluator, history, *x),
        linkage,
        center=dimensions,
        n_particles=n_agents,
        iters=n_iterations,
        bounds=BOUNDS,
        dimensions=len(dimensions),
    )

    fig = plt.figure("Swarm in tiled mode")
    cells = int(np.ceil(np.sqrt(n_agents)))
    axes = fig.subplots(cells, cells)
    formatted_history = [history[i : i + n_agents] for i in range(0, len(history), n_agents)]

    animation = anim.FuncAnimation(
        fig,
        lambda frame: pl.swarm_tiled_repr(
            linkage=linkage,
            swarm=frame,
            fig=fig,
            axes=axes,
            dimension_func=lambda dim: list(param2dimensions(dim, flat=True)),
        ),
        frames=enumerate(formatted_history),
        blit=False,
        interval=1000,
        repeat=False,
        save_count=(n_iterations - 1) * bool(save_each),
    )
    plt.show(block=not save_each)
    if save_each:
        writer = anim.FFMpegWriter(
            fps=24,
            bitrate=1800,
            metadata={
                "title": "Particle swarm looking for R^8 in R application maximum",
                "comment": "Made with Python and Matplotlib",
                "description": "The swarm looks for the best dimension "
                "set for the Strider legged mechanism",
            },
        )
        animation.save("Strider linkage - Particle swarm optimization.mp4", writer=writer)
    _ = animation
    return ensemble


def swarm_optimizer(
    linkage,
    dimensions=DIMENSIONS,
    show=0,
    save_each=0,
    n_agents=300,
    n_iterations=400,
    *args,
):
    """Optimise a strider geometry with PSO.

    :param linkage: The linkage to optimise.
    :param dimensions: Starting design vector.
    :param show: ``0`` no visualisation, ``1`` polar graph, ``2`` tiled preview.
    :param save_each: If non-zero, save every ``save_each`` iterations.
    :param n_agents: Number of particles.
    :param n_iterations: Iterations to run.
    """
    print("Initial dimensions:", dimensions)

    if show == 1:
        return view_swarm_polar(linkage, dimensions, save_each, n_agents, n_iterations)
    if show == 2:
        return view_swarm_tiled(linkage, dimensions, save_each, n_agents, n_iterations)

    return pl.particle_swarm_optimization(
        sym_stride_evaluator,
        linkage,
        *args,
        n_particles=n_agents,
        bounds=BOUNDS,
        dimensions=len(dimensions),
        iters=n_iterations,
    )


def show_optimized(linkage, ensemble, n_show=10, duration=5):
    """Display the top members of an optimization ensemble."""
    for i in range(min(n_show, ensemble.n_members)):
        member = ensemble[i]
        score = member.scores.get("score", 0.0)
        if score <= 0:
            continue
        linkage.set_constraints(list(param2dimensions(member.dimensions, flat=True)))
        pl.show_linkage(
            linkage,
            prev=list(INIT_COORD),
            title=str(score),
            duration=duration,
        )


def main() -> None:
    """Build a strider and run a quick PSO over its geometry."""
    strider = complete_strider(param2dimensions(DIMENSIONS), INIT_COORD)
    print("Initial striding score:", sym_stride_evaluator(strider, DIMENSIONS, INIT_COORD))
    pl.show_linkage(strider, iteration_factor=10)

    ensemble = swarm_optimizer(strider, show=1, save_each=0, n_agents=40, n_iterations=40)
    if ensemble.n_members:
        print(
            "Striding score after particle swarm optimization:",
            ensemble[0].scores.get("score", 0.0),
        )


if __name__ == "__main__":
    main()
