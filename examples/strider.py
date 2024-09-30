"""
Kinematic Strider linkage, a type of walking linkage.

The original linkage can be found at
https://www.diywalkers.com/strider-linkage-plans.html
"""

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

import pylinkage as pl

# Simulation parameters
# Number of points for crank complete turn
LAP_POINTS = 10
# Time (in seconds) for a crank revolution
LAP_PER_SECOND = 100

"""
Parameters that can change without changing joints between objects.

Can be distance between joints, or an angle.
Units are given relative to crank length, which is normalized to 1.
"""
DIM_NAMES = (
    "triangle", "aperture", "femur", "rockerL", "rockerS", "f", "tibia", "phi"
)

DIMENSIONS = (
    # AB distance (=AB_p) "triangle":
    2,
    # "aperture":
    np.pi / 4,
    # femur = 3 for higher steps, 2 for the standard size, but 1.8 is good enough
    1.8,
    # "rockerL":
    2.6,
    # "rockerS":
    1.4,
    # "phi":
    np.pi + .2,
    # "tibia":
    2.5,
    # "f":
    1.8,
)
# Optimized but useless strider with a step of size 5.05
# param = (2.62484195, 1.8450077, 2.41535873, 2.83669735, 2.75235715,
#         4.60386788, 3.49814371, 3.51517851)
# Limits for parameters, will be used in optimizers
BOUNDS = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6)
)

# Initial coordinates according to previous dimensions
INIT_COORD = (
    (0, 0), (0, 1), (1.41, 1.41), (-1.41, 1.41), (0, -1), (-2.25, 0),
    (2.25, 0), (-1.4, -1.2), (1.4, -1.2), (-2.7, -2.7), (2.7, -2.7)
)


def param2dimensions(param=DIMENSIONS, flat=False):
    """Expand dimensions them to fit in strider.set_num_constraints.

    Dimensions parameters are written in short form due to symmetry.

    :param param: Short form for dimensions (Default value = DIMENSIONS)
    :param flat: If the output should be a flat list (Default value = False)

    :return: Expanded dimensions
    """
    out = (
        # Static joints (A and Y)
        (), (),
        # B, B_p
        (param[0], -param[1]), (param[0], param[1]),
        # Crank (C)
        (1, ),
        # D and E
        (param[2], param[3]), (param[2], param[3]),
        # F and G
        (param[4], -param[5]), (param[4], param[5]),
        # H and I
        (param[6], param[7]), (param[6], param[7])
    )
    if not flat:
        return out
    flat_dims = []
    for constraint in out[2:]:
        flat_dims.extend(constraint)
    return tuple(flat_dims)


def complete_strider(constraints, prev):
    """
    Take two sequences to define strider linkage.

    :param constraints: The sequence of geometrical constraints.
    :type constraints: tuple[float] | tuple[tuple[float]]
    :param prev: Coordinates to set by default.
    :type prev: tuple[tuple[float, float]]

    :return pylinkage.Linkage: A strider linkage
    """
    linkage = {
        # Fixed points (mechanism body)
        # A is the origin
        "A": pl.Static(x=0, y=0, name="A"),
        # Vertical axis for convenience
        "Y": pl.Static(0, 1, name="Point (0, 1)"),
    }
    # For drawing only
    linkage["Y"].joint0 = linkage["A"]
    linkage.update({
        # Not fixed because we will optimize this position
        "B": pl.Fixed(joint0=linkage["A"], joint1=linkage["Y"], name="Frame right (B)"),
        "B_p": pl.Fixed(joint0=linkage["A"], joint1=linkage["Y"], name="Frame left (B_p)"),
        # Pivot joints, explicitly defined to be modified later
        # Joint linked to crank. Coordinates are chosen in each frame
        "C": pl.Crank(joint0=linkage["A"], angle=-2 * np.pi / LAP_POINTS, name="Crank link (C)")
    })
    linkage.update({
        "D": pl.Revolute(joint0=linkage["B_p"], joint1=linkage["C"], name="Left knee link (D)"),
        "E": pl.Revolute(joint0=linkage["B"], joint1=linkage["C"], name="Right knee link (E)")
    })
    linkage.update({
        # F is fixed relative to C and E
        "F": pl.Fixed(joint0=linkage["C"], joint1=linkage["E"], name='Left ankle link (F)'),
        # G fixed to C and D
        "G": pl.Fixed(joint0=linkage["C"], joint1=linkage["D"], name='Right ankle link (G)')
    })
    linkage.update({
        "H": pl.Revolute(joint0=linkage["D"], joint1=linkage["F"], name="Left foot (H)"),
        "I": pl.Revolute(joint0=linkage["E"], joint1=linkage["G"], name="Right foot (I)")
    })
    # Mechanism definition
    strider = pl.Linkage(
        joints=linkage.values(),
        order=linkage.values(),
        name="Strider"
    )
    strider.set_coords(prev)
    strider.set_num_constraints(constraints, flat=False)
    return strider


def sym_stride_evaluator(linkage, dimensions, initial_positions):
    """Give score to each dimension set for symmetric strider.

    :param linkage: Input linkage
    :type linkage: pylinkage.Linkage
    :param dimensions: Dimensions
    :type dimensions: tuple
    :param initial_positions: Initial positions
    :type initial_positions: tuple

    :return: Score
    :rtype: float
    """
    linkage.set_completely(param2dimensions(dimensions, flat=True), initial_positions)
    points = 12
    try:
        # Complete revolution with 12 points
        loci = tuple(
            map(
                tuple,
                linkage.step(
                    iterations=points, dt=LAP_POINTS / points
                )
            )
        )
    except pl.UnbuildableError:
        return 0
    foot_locus = tuple(x[-2] for x in loci)
    # Constraints check
    # Performances evaluation
    score = max(k[0] for k in foot_locus) - min(k[0] for k in foot_locus)
    return score


def history_saver(evaluator, history, linkage, dims, pos):
    """
    Save the history to a list.

    :param evaluator: Evaluation function
    :param history: History list
    :param linkage: Input linkage
    :param dims: Dimensions
    :param pos: Initial positions

    """
    score = evaluator(linkage, dims, pos)
    history.append((score, list(dims), pos))
    return score


def view_swarm_polar(
    linkage,
    dimensions=DIMENSIONS,
    save_each=0,
    n_agents=300,
    n_iterations=400
):
    """Draw an animation of the swarm in a polar graph.

    :param linkage: Input linkage
    :type linkage: pylinkage.Linkage
    :param dimensions: Dimensions (Default value = DIMENSIONS)
    :type dimensions: Sized
    :param save_each: To save the linkage state (Default value = 0)
    :type save_each: int | None
    :param n_agents: Number of agents (Default value = 300)
    :type n_agents: int
    :param n_iterations: NUmber of iterations (Default value = 400)
    :type n_iterations: int

    
    """
    history = []
    out = pl.particle_swarm_optimization(
        lambda *x: history_saver(sym_stride_evaluator, history, *x),
        linkage,
        center=dimensions, n_particles=n_agents, iters=n_iterations,
        bounds=BOUNDS, dimensions=len(dimensions)
    )

    fig = plt.figure(f"Swarm in polar graph")
    fig.suptitle(f"Final best score: {-out[0][0]:.2f}")
    formatted_history = [
        history[i:i + n_agents] for i in range(0, len(history), n_agents)
    ]
    artists = []

    def init_polar_repr():
        """Set the axis for the polar representation."""
        ax = fig.add_subplot(111, projection='polar')
        artists.extend(
            ax.plot([], [], lw=.5, animated=False)[0] for _ in range(n_agents)
        )
        ax.set_rmax(7)
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, len(dimensions) + 1, endpoint=False),
            DIM_NAMES + ("score",)
        )
        artists.append(ax.text(1.9 * np.pi, 2, "", animated=True))
        return artists

    def repr_polar_swarm(current_swarm):
        """Represent a swarm in a polar graph.

        :param current_swarm: Swarm of agents

        """
        t = np.linspace(0, 2 * np.pi, len(current_swarm[1][0][1]) + 2)[:-1]
        for line, agent in zip(artists, current_swarm[1]):
            line.set_data(t, agent[1] + [agent[0]])
        artists[-1].set_text(
            f"Best score: {max(x[0] for x in current_swarm[1]):.2f}"
            f"\nIteration: {current_swarm[0]}"
        )

        return artists

    animation = anim.FuncAnimation(
        fig,
        func=repr_polar_swarm,
        frames=enumerate(formatted_history),
        init_func=init_polar_repr,
        blit=True,
        interval=400, repeat=True,
        save_count=(n_iterations - 1) * bool(save_each)
    )
    plt.show()
    if save_each:
        writer = anim.FFMpegWriter(
            fps=24, bitrate=1800,
            metadata={
                'title': "Particle swarm looking for R^8 in R "
                "application maximum",
                'comment': "Made with Python and Matplotlib",
                'description': "The swarm tries to find best dimension"
                " set for Strider legged mechanism"
            }
        )
        animation.save(
            r"Particle Swarm Optimization of Strider linkage.mp4",
            writer=writer
        )
    # Prevent the garbage-collection of the animation
    if animation:
        pass
    return out


def view_swarm_tiled(
    linkage,
    dimensions=DIMENSIONS,
    save_each=0,
    n_agents=300,
    n_iterations=400
):
    """Represent the final state of the best linkages.

    :param linkage: Linkage to edit
    :type linkage: Linkage
    :param dimensions: Dimensions (Default value = DIMENSIONS)
    :type dimensions: Sized
    :param save_each: Period of state save (Default value = 0)
    :type save_each: int | None
    :param n_agents: Number of agents (Default value = 300)
    :type n_agents: int
    :param n_iterations: NUmber of iterations (Default value = 400)
    :type n_iterations: int

    
    """
    history = []

    out = pl.particle_swarm_optimization(
        lambda *x: history_saver(sym_stride_evaluator, history, *x),
        linkage,
        center=dimensions, n_particles=n_agents, iters=n_iterations,
        bounds=BOUNDS, dimensions=len(dimensions)
    )

    fig = plt.figure("Swarm in tiled mode")
    cells = int(np.ceil(np.sqrt(n_agents)))
    axes = fig.subplots(cells, cells)
    formatted_history = [
        history[i:i + n_agents] for i in range(0, len(history), n_agents)
    ]

    animation = anim.FuncAnimation(
        fig,
        lambda frame: pl.swarm_tiled_repr(
            linkage=linkage,
            swarm=frame,
            fig=fig,
            axes=axes,
            dimension_func=lambda dim: param2dimensions(dim, flat=True)
        ),
        frames=enumerate(formatted_history),
        blit=False,
        interval=1000, repeat=False,
        save_count=(n_iterations - 1) * bool(save_each)
    )
    plt.show(block=not save_each)
    if save_each:
        writer = anim.FFMpegWriter(
            fps=24, bitrate=1800,
            metadata={
                'title': "Particle swarm looking for R^8 in R "
                "application maximum",
                'comment': "Made with Python and Matplotlib",
                'description': "The swarm looks for best dimension "
                "set for Strider legged mechanism"
            }
        )

        animation.save(
            "Strider linkage - Particle swarm optimization.mp4",
            writer=writer
        )
    # Prevent the garbage-collection of the animation
    if animation:
        pass
    return out


def swarm_optimizer(
    linkage,
    dimensions=DIMENSIONS,
    show=False,
    save_each=0,
    n_agents=300,
    n_iterations=400,
    *args,
):
    """Optimize linkage geometrically using PSO.

    :param linkage: The linkage to optimize.
    :type linkage: pylinkage.linkage.Linkage
    :param dimensions: The dimensions that should vary.
    (Default value = DIMENSIONS).
    :type dimensions: list[float]
    :param show: Type of visualization.
        - 0 for None
        - 1 for polar graph
        - 2 for tiled 2D representation
        The default is False.
    :type show: int
    :param save_each: If show is 0, save the image each {save_each} frame. The default is 0.
    :type save_each: int
    :param n_agents: Number of agents to simulate. The default is 300.
    :type n_agents: int
    :param n_iterations: Number of iterations to run through. The default is 400.
    :type n_iterations: int
    :param args: Arguments to pass to the particle swarm optimization.
    :type args: list

    :return: List of fittest linkages.
    :rtype: list
    """
    print("Initial dimensions:", dimensions)

    if show == 1:
        return view_swarm_polar(linkage, dimensions, save_each, n_agents, n_iterations)
    elif show == 2:
        # Tiled representation of swarm
        return view_swarm_tiled(linkage, dimensions, save_each, n_agents, n_iterations)

    if save_each:
        for dim, i in pl.particle_swarm_optimization(
            sym_stride_evaluator,
            linkage,
            dimensions=len(dimensions),
            n_particles=n_agents,
            iters=n_iterations,
            bounds=BOUNDS
        ):
            if not i % save_each:
                f = open('PSO optimizer.txt', 'w')
                # We only keep the best results
                dim.sort(key=lambda x: x[1], reverse=True)
                for j in range(min(10, len(dim))):
                    par = {}
                    for k in range(len(dim[j][0])):
                        par[DIM_NAMES[k]] = dim[j][0][k]
                    f.write('{}\n{}\n{}\n'.format(par, dim[j][1], dim[j][2]))
                    f.write('----\n')
                f.close()
    else:
        out = pl.particle_swarm_optimization(
            sym_stride_evaluator,
            linkage,
            dimensions,
            n_particles=n_agents,
            bounds=BOUNDS,
            dimensions=len(dimensions),
            iters=n_iterations,
            *args
        )
        return tuple(out)


def show_optimized(linkage, data, n_show=10, duration=5, symmetric=True):
    """Show the optimized linkages.

    :param linkage: Linkage to modify.
    :param data: Linkage data
    :param n_show: Number of linkages to show (Default value = 10)
    :param duration: Duration of each animation (Default value = 5)
    :param symmetric: If the input dimensions should be symmetric (Default value = True)

    """
    for datum in data[:min(len(data), n_show)]:
        if datum[0] <= 0:
            continue
        if symmetric:
            linkage.set_num_constraints(param2dimensions(datum[1]), flat=False)
        else:
            linkage.set_num_constraints(datum[1], flat=False)
        pl.show_linkage(
            linkage, prev=INIT_COORD, title=str(datum[0]), duration=duration
        )


def main():
    """Build and optimize a strider linkage.
    
    You can find it at https://www.diywalkers.com/strider-linkage-plans.html
    """
    strider = complete_strider(param2dimensions(DIMENSIONS), INIT_COORD)
    print(
        "Initial striding score:",
        sym_stride_evaluator(strider, DIMENSIONS, INIT_COORD)
    )
    pl.show_linkage(strider, iteration_factor=10)

    # Particle swarm optimization
    optimized_striders = swarm_optimizer(
        strider, show=1, save_each=0, n_agents=40, n_iterations=40
    )
    print(
        "Striding score after particle swarm optimization:",
        optimized_striders[0][0]
    )


if __name__ == "__main__":
    main()
