Kinematics-Based Optimization
==============================

This tutorial covers how to use velocity and acceleration analysis in linkage
optimization. By incorporating kinematic quantities into fitness functions,
you can design linkages that not only follow a desired path but also meet
velocity and acceleration requirements.

Overview
--------

Pylinkage can compute:

- **Linear velocities** of all joints given crank angular velocity
- **Linear accelerations** of all joints given crank angular velocity and acceleration
- **Velocity vectors** for visualization

This enables optimization for:

- Minimizing peak velocities at output joints
- Achieving uniform velocity profiles
- Limiting accelerations to reduce wear and vibration
- Matching velocity requirements for specific applications

Setting Up Angular Velocity
---------------------------

Before computing kinematics, set the angular velocity on the input crank:

.. code-block:: python

   from pylinkage.actuators import Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage

   # Create a four-bar linkage
   A = Ground(0.0, 0.0, name="A")
   D = Ground(3.0, 0.0, name="D")
   crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="Crank")
   output = RRRDyad(
       anchor1=crank.output, anchor2=D,
       distance1=3.0, distance2=2.0, name="Output",
   )
   linkage = Linkage([A, D, crank, output])

   # Set angular velocity (rad/s) and optional angular acceleration (rad/s²)
   linkage.set_input_velocity(crank, omega=10.0, alpha=0.0)

``angular_velocity`` on the ``Crank`` is the rotation per simulation step;
``omega`` is the physical rate the velocities are computed for. With 0.1 rad
per step and 10 rad/s, one step is a hundredth of a second.

Running Kinematics Simulation
-----------------------------

Use ``step_fast_with_kinematics()`` to compute positions, velocities and
accelerations in one pass through the numba solver:

.. code-block:: python

   # Run simulation with kinematics
   positions, velocities, accelerations = linkage.step_fast_with_kinematics(
       iterations=100,
   )

   # positions.shape = (100, n_components, 2)      # (frames, components, x/y)
   # velocities.shape = (100, n_components, 2)     # (frames, components, vx/vy)
   # accelerations.shape = (100, n_components, 2)  # (frames, components, ax/ay)

   # Access velocity at a specific frame
   frame = 25
   for i, component in enumerate(linkage.components):
       vx, vy = velocities[frame, i]
       print(f"{component.name}: velocity = ({vx:.2f}, {vy:.2f})")

Positions the solver could not assemble come out as ``NaN`` rather than an
exception, so kinematic fitness functions below filter with ``numpy.isnan``.

Querying Joint Velocities
-------------------------

After running the kinematics simulation, joint velocities are accessible:

.. code-block:: python

   # Get velocities for all components (at the last simulated frame)
   all_velocities = linkage.get_velocities()
   # Returns: [(vx0, vy0), (vx1, vy1), ...]

   # Access an individual component's velocity
   output_velocity = linkage.components[-1].velocity
   if output_velocity is not None:
       vx, vy = output_velocity
       speed = (vx**2 + vy**2) ** 0.5
       print(f"Output speed: {speed:.2f} units/s")

Optimizing for Velocity Characteristics
---------------------------------------

Example: Minimize Peak Velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Design a linkage where the output joint has the lowest possible peak velocity:

.. code-block:: python

   import numpy as np
   import pylinkage as pl
   from pylinkage.actuators import Crank as _Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage


   def create_linkage():
       A = Ground(0.0, 0.0, name="A")
       D = Ground(3.0, 0.0, name="D")
       crank = _Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="Crank")
       output = RRRDyad(
           anchor1=crank.output, anchor2=D,
           distance1=3.0, distance2=2.0, name="Output",
       )
       return Linkage([A, D, crank, output])


   @pl.kinematic_minimization
   def minimize_peak_velocity(loci, linkage=None, **kwargs):
       """Minimize the peak velocity at the output joint."""
       crank = next(c for c in linkage.components if isinstance(c, _Crank))
       linkage.set_input_velocity(crank, omega=10.0)

       # Run kinematics
       positions, velocities, _ = linkage.step_fast_with_kinematics()

       # Calculate output joint speed at each frame
       output_velocities = velocities[:, -1, :]  # Last joint
       speeds = np.sqrt(output_velocities[:, 0]**2 + output_velocities[:, 1]**2)

       # Return peak velocity (we want to minimize this)
       peak_velocity = np.nanmax(speeds)
       return peak_velocity if not np.isnan(peak_velocity) else float('inf')


   # Run optimization
   linkage = create_linkage()
   bounds = pl.generate_bounds(linkage.get_constraints())

   results = pl.particle_swarm_optimization(
       eval_func=minimize_peak_velocity,
       linkage=linkage,
       bounds=bounds,
       order_relation=min,
       n_particles=30,
       iterations=30,
       verbose=False,
   )

   best = results[0]
   print(f"Minimum peak velocity: {best.score:.2f} units/s")
   print(f"Dimensions: {best.dimensions}")

The trivial answer to "lowest peak velocity" is a tiny crank, and that is
what the swarm finds unless the bounds or the fitness rule it out — the
examples below add such conditions.

Example: Uniform Velocity Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize for a constant output velocity (useful for conveyor mechanisms):

.. code-block:: python

   @pl.kinematic_minimization
   def uniform_velocity(loci, linkage=None, **kwargs):
       """Minimize velocity variation at the output joint."""
       crank = next(c for c in linkage.components if isinstance(c, _Crank))
       linkage.set_input_velocity(crank, omega=10.0)

       positions, velocities, _ = linkage.step_fast_with_kinematics()

       # Calculate output speeds
       output_velocities = velocities[:, -1, :]
       speeds = np.sqrt(output_velocities[:, 0]**2 + output_velocities[:, 1]**2)

       # Filter out NaN values
       valid_speeds = speeds[~np.isnan(speeds)]
       if len(valid_speeds) == 0:
           return float('inf')

       # Minimize standard deviation of speed (uniformity measure)
       velocity_variance = np.std(valid_speeds)

       # Also penalize very low average speed (we want motion, not stillness)
       avg_speed = np.mean(valid_speeds)
       if avg_speed < 5.0:
           return float('inf')

       return velocity_variance


   results = pl.particle_swarm_optimization(
       eval_func=uniform_velocity,
       linkage=create_linkage(),
       bounds=bounds,
       order_relation=min,
       n_particles=30,
       iterations=30,
       verbose=False,
   )
   print(f"Speed standard deviation: {results[0].score:.3f} units/s")

Example: Velocity Direction Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize for a specific velocity direction at certain positions:

.. code-block:: python

   @pl.kinematic_minimization
   def horizontal_velocity(loci, linkage=None, **kwargs):
       """Maximize horizontal velocity component at output."""
       crank = next(c for c in linkage.components if isinstance(c, _Crank))
       linkage.set_input_velocity(crank, omega=10.0)

       positions, velocities, _ = linkage.step_fast_with_kinematics()

       # Get output velocities
       output_vx = velocities[:, -1, 0]
       output_vy = velocities[:, -1, 1]

       # Calculate ratio of horizontal to total velocity
       speeds = np.sqrt(output_vx**2 + output_vy**2)
       horizontal_ratio = np.abs(output_vx) / (speeds + 1e-10)

       # Filter NaN and calculate average
       valid_ratio = horizontal_ratio[~np.isnan(horizontal_ratio)]
       if len(valid_ratio) == 0:
           return float('inf')

       # Return negative average (we want to maximize horizontal component)
       return -np.mean(valid_ratio)


   results = pl.particle_swarm_optimization(
       eval_func=horizontal_velocity,
       linkage=create_linkage(),
       bounds=bounds,
       order_relation=min,
       n_particles=30,
       iterations=30,
       verbose=False,
   )
   print(f"Horizontal fraction of the output velocity: {-results[0].score:.3f}")

Combined Path and Velocity Optimization
---------------------------------------

Optimize for both path shape and velocity characteristics:

.. code-block:: python

   @pl.kinematic_minimization
   def path_and_velocity(loci, linkage=None, **kwargs):
       """Optimize path shape while limiting peak velocity."""
       # Path shape objective
       output_path = [step[-1] for step in loci]
       bbox = pl.bounding_box(output_path)
       target_bbox = (0, 5, 2, 3)  # min_y, max_x, max_y, min_x
       path_error = sum((a - t)**2 for a, t in zip(bbox, target_bbox))

       # Velocity objective
       crank = next(c for c in linkage.components if isinstance(c, _Crank))
       linkage.set_input_velocity(crank, omega=10.0)
       positions, velocities, _ = linkage.step_fast_with_kinematics()

       output_velocities = velocities[:, -1, :]
       speeds = np.sqrt(output_velocities[:, 0]**2 + output_velocities[:, 1]**2)
       peak_velocity = np.nanmax(speeds) if np.any(~np.isnan(speeds)) else 100.0

       # Weighted combination
       # Penalize peak velocities above 15 units/s
       velocity_penalty = max(0, peak_velocity - 15) ** 2

       return path_error + 0.1 * velocity_penalty


   results = pl.particle_swarm_optimization(
       eval_func=path_and_velocity,
       linkage=create_linkage(),
       bounds=bounds,
       order_relation=min,
       n_particles=30,
       iterations=30,
       verbose=False,
   )
   print(f"Combined score: {results[0].score:.3f}")

Visualizing Velocity Vectors
----------------------------

Matplotlib Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import show_kinematics, animate_kinematics

   # Set up linkage with angular velocity
   linkage = create_linkage()
   crank = linkage.components[2]
   linkage.set_input_velocity(crank, omega=10.0)

   # Show single frame with velocity vectors
   fig = show_kinematics(linkage, frame_index=25, show_velocity=True)
   fig.savefig("velocity_frame.png")

   # Create animation with velocity vectors (saved as GIF via Pillow)
   fig = animate_kinematics(
       linkage,
       show_velocity=True,
       fps=24,
       duration=2.0,
       save_path="velocity_animation.gif",
   )

Plotly Interactive Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import plot_linkage_plotly_with_velocity

   # Runs the kinematics itself; needs set_input_velocity() beforehand
   fig = plot_linkage_plotly_with_velocity(
       linkage,
       frame_index=25,
       velocity_scale=0.1,
       title="Four-bar with Velocity Vectors",
   )
   fig.write_html("velocity_plot.html")

SVG Publication-Quality Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import save_linkage_svg_with_velocity

   positions, velocities, _ = linkage.step_fast_with_kinematics()

   # Save SVG with velocity vectors
   save_linkage_svg_with_velocity(
       linkage,
       "linkage_velocity.svg",
       positions[25],
       velocities[25],
       velocity_color="#0066CC"
   )

Performance Considerations
--------------------------

The kinematics computation adds minimal overhead:

1. Velocity is computed analytically (not numerically), so it's fast
2. The velocity solver is numba-compiled for maximum performance
3. Memory usage increases by 2x (storing velocities alongside positions)

For optimization loops, cache the kinematics results:

.. code-block:: python

   @pl.kinematic_minimization
   def optimized_fitness(loci, linkage=None, **kwargs):
       # Run kinematics once
       crank = next(c for c in linkage.components if isinstance(c, _Crank))
       linkage.set_input_velocity(crank, omega=10.0)
       positions, velocities, accelerations = linkage.step_fast_with_kinematics()
       if np.isnan(positions).any():
           return float("inf")

       # Compute all metrics from the same arrays
       speeds = np.linalg.norm(velocities[:, -1, :], axis=1)
       peak_vel = speeds.max()
       vel_uniformity = speeds.std()
       peak_acc = np.linalg.norm(accelerations[:, -1, :], axis=1).max()

       return peak_vel + 2.0 * vel_uniformity + 0.01 * peak_acc

Next Steps
----------

- See :doc:`advanced_optimization` for general optimization techniques
- Check :py:mod:`pylinkage.solver.velocity` for velocity solver implementation
- Explore :py:mod:`pylinkage.visualizer` for visualization options
