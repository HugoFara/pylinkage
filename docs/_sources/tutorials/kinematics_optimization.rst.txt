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

   import pylinkage as pl

   # Create a four-bar linkage
   crank = pl.Crank(
       x=0, y=1,
       joint0=(0, 0),
       angle=0.1,
       distance=1,
       name="Crank"
   )
   output = pl.Revolute(
       x=3, y=2,
       joint0=crank,
       joint1=(3, 0),
       distance0=3,
       distance1=1,
       name="Output"
   )
   linkage = pl.Linkage(
       joints=(crank, output),
       order=(crank, output),
   )

   # Set angular velocity (rad/s) and optional angular acceleration (rad/s²)
   linkage.set_input_velocity(crank, omega=10.0, alpha=0.0)

Running Kinematics Simulation
-----------------------------

Use ``step_fast_with_kinematics()`` to compute positions and velocities:

.. code-block:: python

   # Run simulation with kinematics
   positions, velocities = linkage.step_fast_with_kinematics(iterations=100)

   # positions.shape = (100, n_joints, 2)  # (frames, joints, x/y)
   # velocities.shape = (100, n_joints, 2)  # (frames, joints, vx/vy)

   # Access velocity at a specific frame
   frame = 25
   for i, joint in enumerate(linkage.joints):
       vx, vy = velocities[frame, i]
       print(f"{joint.name}: velocity = ({vx:.2f}, {vy:.2f})")

Querying Joint Velocities
-------------------------

After running the kinematics simulation, joint velocities are accessible:

.. code-block:: python

   # Get velocities for all joints (after simulation)
   all_velocities = linkage.get_velocities()
   # Returns: [(vx0, vy0), (vx1, vy1), ...]

   # Access individual joint velocity
   output_velocity = linkage.joints[-1].velocity
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


   def create_linkage():
       crank = pl.Crank(
           x=0, y=1,
           joint0=(0, 0),
           angle=0.1,
           distance=1,
           name="Crank"
       )
       output = pl.Revolute(
           x=3, y=2,
           joint0=crank,
           joint1=(3, 0),
           distance0=3,
           distance1=1,
           name="Output"
       )
       return pl.Linkage(
           joints=(crank, output),
           order=(crank, output),
       )


   @pl.kinematic_minimization
   def minimize_peak_velocity(loci, linkage=None, **kwargs):
       """Minimize the peak velocity at the output joint."""
       # Get the crank and set angular velocity
       crank = linkage.joints[0]
       linkage.set_input_velocity(crank, omega=10.0)

       # Run kinematics
       positions, velocities = linkage.step_fast_with_kinematics()

       # Calculate output joint speed at each frame
       output_velocities = velocities[:, -1, :]  # Last joint
       speeds = np.sqrt(output_velocities[:, 0]**2 + output_velocities[:, 1]**2)

       # Return peak velocity (we want to minimize this)
       peak_velocity = np.nanmax(speeds)
       return peak_velocity if not np.isnan(peak_velocity) else float('inf')


   # Run optimization
   linkage = create_linkage()
   bounds = pl.generate_bounds(linkage.get_num_constraints())

   results = pl.particle_swarm_optimization(
       eval_func=minimize_peak_velocity,
       linkage=linkage,
       bounds=bounds,
       n_particles=50,
       iters=100,
   )

   best_score, best_constraints, _ = results[0]
   print(f"Minimum peak velocity: {best_score:.2f} units/s")

Example: Uniform Velocity Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize for a constant output velocity (useful for conveyor mechanisms):

.. code-block:: python

   @pl.kinematic_minimization
   def uniform_velocity(loci, linkage=None, **kwargs):
       """Minimize velocity variation at the output joint."""
       crank = linkage.joints[0]
       linkage.set_input_velocity(crank, omega=10.0)

       positions, velocities = linkage.step_fast_with_kinematics()

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
       if avg_speed < 10.0:
           return float('inf')

       return velocity_variance

Example: Velocity Direction Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize for a specific velocity direction at certain positions:

.. code-block:: python

   @pl.kinematic_minimization
   def horizontal_velocity(loci, linkage=None, **kwargs):
       """Maximize horizontal velocity component at output."""
       crank = linkage.joints[0]
       linkage.set_input_velocity(crank, omega=10.0)

       positions, velocities = linkage.step_fast_with_kinematics()

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
       crank = linkage.joints[0]
       linkage.set_input_velocity(crank, omega=10.0)
       positions, velocities = linkage.step_fast_with_kinematics()

       output_velocities = velocities[:, -1, :]
       speeds = np.sqrt(output_velocities[:, 0]**2 + output_velocities[:, 1]**2)
       peak_velocity = np.nanmax(speeds) if np.any(~np.isnan(speeds)) else 100.0

       # Weighted combination
       # Penalize peak velocities above 50 units/s
       velocity_penalty = max(0, peak_velocity - 50) ** 2

       return path_error + 0.1 * velocity_penalty

Visualizing Velocity Vectors
----------------------------

Matplotlib Visualization
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import show_kinematics, animate_kinematics

   # Set up linkage with angular velocity
   linkage.set_input_velocity(crank, omega=10.0)

   # Show single frame with velocity vectors
   fig = show_kinematics(linkage, frame_index=25, show_velocity=True)

   # Create animation with velocity vectors
   fig = animate_kinematics(
       linkage,
       show_velocity=True,
       fps=24,
       save_path="velocity_animation.gif"
   )

Plotly Interactive Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import plot_linkage_plotly_with_velocity

   # Run kinematics
   positions, velocities = linkage.step_fast_with_kinematics()

   # Create interactive plot with velocity arrows
   fig = plot_linkage_plotly_with_velocity(
       linkage,
       positions[25],     # Position at frame 25
       velocities[25],    # Velocity at frame 25
       velocity_scale=0.1,
       title="Four-bar with Velocity Vectors"
   )
   fig.write_html("velocity_plot.html")

SVG Publication-Quality Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pylinkage.visualizer import save_linkage_svg_with_velocity

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
       crank = linkage.joints[0]
       linkage.set_input_velocity(crank, omega=10.0)
       positions, velocities = linkage.step_fast_with_kinematics()

       # Compute all metrics from cached results
       peak_vel = compute_peak_velocity(velocities)
       vel_uniformity = compute_uniformity(velocities)
       path_score = compute_path_score(positions)

       return weighted_combination(peak_vel, vel_uniformity, path_score)

Next Steps
----------

- See :doc:`advanced_optimization` for general optimization techniques
- Check :py:mod:`pylinkage.solver.velocity` for velocity solver implementation
- Explore :py:mod:`pylinkage.visualizer` for visualization options
