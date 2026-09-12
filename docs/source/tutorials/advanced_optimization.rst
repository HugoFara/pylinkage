Advanced Optimization Techniques
=================================

This tutorial covers advanced optimization techniques for linkage mechanisms
using pylinkage's Particle Swarm Optimization (PSO) and grid search capabilities.

Overview of Optimization
------------------------

Linkage optimization finds the best geometric parameters (distances, angles)
to achieve a desired motion. Pylinkage provides:

- **Particle Swarm Optimization (PSO)**: Efficient global optimization using swarm intelligence
- **Trials and Errors (Grid Search)**: Exhaustive search over a parameter grid

Defining a Fitness Function
---------------------------

The fitness function evaluates how well a linkage configuration meets your goals.
Use the ``@kinematic_minimization`` or ``@kinematic_maximization`` decorators.

Basic Fitness Function
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pylinkage as pl


   @pl.kinematic_minimization
   def fitness(loci, **kwargs):
       """Evaluate linkage fitness.

       :param loci: Joint positions for each simulation step.
           Structure: tuple[tuple[tuple[float, float], ...], ...]
           - Outer tuple: simulation steps
           - Middle tuple: joints at each step
           - Inner tuple: (x, y) coordinates
       :param kwargs: ``linkage``, ``params`` (the candidate constraints)
           and ``init_pos`` (the joint positions the simulation started from).
       :return: Fitness score (lower is better for minimization)
       """
       # Get the locus (path) of the last joint
       output_locus = [step[-1] for step in loci]

       # Calculate your fitness metric: here, how far the path strays
       # from a horizontal line.
       ys = [y for _, y in output_locus]
       return max(ys) - min(ys)

The decorated function has the signature every optimizer expects,
``(linkage, params, init_pos)``. The decorator:

- applies the candidate ``params`` with ``linkage.set_constraints()``;
- runs the simulation and passes the loci in;
- catches ``UnbuildableError`` and returns infinity (``-inf`` for
  ``@kinematic_maximization``), so impossible geometries lose without
  crashing the search.

Optimizers look for the **highest** score by default. A minimization
fitness therefore goes with ``order_relation=min``, as below.

Working with Loci Data
^^^^^^^^^^^^^^^^^^^^^^

Understanding the loci structure is key to writing good fitness functions:

.. code-block:: python

   @pl.kinematic_minimization
   def analyze_loci(loci, **kwargs):
       """Example showing loci structure."""
       # loci[step][joint] = (x, y)

       # Get all positions of joint 0 (usually the crank)
       crank_path = [step[0] for step in loci]

       # Get all positions of the last joint (output)
       output_path = [step[-1] for step in loci]

       # Get positions at a specific step
       positions_at_step_5 = loci[5]  # All joint positions at step 5

       # Calculate bounding box of output path
       bbox = pl.bounding_box(output_path)
       # bbox = (min_y, max_x, max_y, min_x)

       # Score: the crank's highest point should stay under the output's
       return max(y for _, y in crank_path) - bbox[2]

Example: Optimizing for Path Shape
----------------------------------

Let's optimize a four-bar linkage so its output traces a specific rectangular path:

.. code-block:: python

   import pylinkage as pl
   from pylinkage.actuators import Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage


   def create_linkage():
       """Create the base linkage to optimize."""
       A = Ground(0.0, 0.0, name="A")
       D = Ground(3.0, 0.0, name="D")
       crank = Crank(
           anchor=A, radius=1.0, angular_velocity=0.31, name="Crank",
       )
       output = RRRDyad(
           anchor1=crank.output, anchor2=D,
           distance1=3.0, distance2=1.0, name="Output",
       )
       return Linkage([A, D, crank, output])


   @pl.kinematic_minimization
   def rectangle_fitness(loci, **kwargs):
       """Minimize distance from a target rectangle."""
       output_path = [step[-1] for step in loci]
       bbox = pl.bounding_box(output_path)

       # Target rectangle: min_y=0, max_x=5, max_y=2, min_x=3
       target = (0, 5, 2, 3)

       # Sum of squared differences
       return sum((actual - target_val) ** 2
                  for actual, target_val in zip(bbox, target))


   # Run optimization
   linkage = create_linkage()

   # Generate search bounds around current constraints
   bounds = pl.generate_bounds(linkage.get_constraints())

   results = pl.particle_swarm_optimization(
       eval_func=rectangle_fitness,
       linkage=linkage,
       bounds=bounds,
       order_relation=min,   # rectangle_fitness is an error to minimize
       n_particles=30,
       iterations=50,
   )

   # Every optimizer returns an Ensemble: the optimized linkage plus one
   # Member per candidate kept, best first.
   best = results[0]
   print(f"Best score: {best.score}")
   print(f"Best constraints: {best.dimensions}")

   # Animate the best member: the Ensemble simulates it from the joint
   # positions the search started from, then hands the loci to show_linkage
   results.show(0)

   # Apply the best constraints to the linkage itself
   linkage.set_constraints(best.dimensions)

An optimum often sits right at the edge of what can be assembled, where a
four-bar has two mirror-image ways to close and a fine simulation from the
original joint positions may pick the wrong one. ``results.show()`` and
``results[0].trajectory`` avoid the question by simulating the member the
way the optimizer scored it; prefer them to re-stepping the linkage by hand.

Particle Swarm Optimization Parameters
--------------------------------------

Fine-tune PSO behavior for better results:

.. code-block:: python

   results = pl.particle_swarm_optimization(
       eval_func=rectangle_fitness,
       linkage=linkage,
       bounds=bounds,

       # Number of particles in the swarm
       n_particles=100,      # More particles = better exploration, slower

       # Number of iterations
       iterations=200,       # More iterations = better convergence, slower

       # Starting position (optional)
       center=None,          # None: particles start uniformly within bounds

       # Number of dimensions (usually auto-detected)
       dimensions=None,

       # Swarm dynamics
       leader=3.0,           # Pull towards the neighbourhood's best
       follower=0.1,         # Pull towards the particle's own best
       inertia=0.6,          # Momentum
       neighbors=17,         # Ring-topology neighbourhood size

       # Order relation for optimization
       order_relation=min,   # min for minimization, max for maximization
       verbose=False,        # No progress bar
   )

Generating Bounds
-----------------

The ``generate_bounds`` function creates search ranges around current values:

.. code-block:: python

   constraints = linkage.get_constraints()
   # Here: [1.0, 3.0, 1.0] — crank radius, coupler length, rocker length

   bounds = pl.generate_bounds(constraints)
   # Returns: (lower_bounds, upper_bounds)
   # Default: values / 5 to values * 5

   # Custom bounds
   bounds = pl.generate_bounds(
       constraints,
       min_ratio=1.25,   # Lower bound = value / 1.25
       max_factor=1.25,  # Upper bound = value * 1.25
   )

   # Or define bounds manually for precise control
   bounds = (
       [0.5, 2.0, 0.5],    # Lower bounds
       [2.0, 5.0, 2.0],    # Upper bounds
   )

Grid Search Optimization
------------------------

For simpler problems or exhaustive search:

.. code-block:: python

   results = pl.trials_and_errors_optimization(
       eval_func=rectangle_fitness,
       linkage=linkage,
       bounds=bounds,
       divisions=6,           # Points per dimension
       n_results=5,           # Members to keep
       order_relation=min,    # min or max
       verbose=False,
   )
   print(f"{len(results)} candidates kept, best score {results[0].score:.3f}")

   # Note: Grid search is O(divisions^n) where n = number of constraints
   # Use sparingly for high-dimensional problems

Multi-Objective Optimization
----------------------------

For true multi-objective optimization with Pareto fronts, see :doc:`multi_objective`.

For simple cases where objectives can be combined with weights:

.. code-block:: python

   @pl.kinematic_minimization
   def weighted_objectives(loci, **kwargs):
       """Combine objectives with weights (simple approach)."""
       output_path = [step[-1] for step in loci]

       # Objective 1: Match target bounding box
       bbox = pl.bounding_box(output_path)
       target = (0, 5, 2, 3)
       shape_error = sum((a - t) ** 2 for a, t in zip(bbox, target))

       # Objective 2: Minimize total mechanism size
       all_points = [p for step in loci for p in step]
       mech_bbox = pl.bounding_box(all_points)
       mechanism_size = (mech_bbox[1] - mech_bbox[3]) * (mech_bbox[2] - mech_bbox[0])

       # Weighted combination (requires choosing weights upfront)
       return shape_error + 0.1 * mechanism_size

For exploring trade-offs without committing to weights, use
``multi_objective_optimization()`` which returns the full Pareto front.

Constraint Preservation
-----------------------

Sometimes you want to optimize only certain constraints while keeping others fixed:

.. code-block:: python

   @pl.kinematic_minimization
   def constrained_fitness(loci, params, **kwargs):
       """Fitness function that enforces additional constraints."""
       # Penalize if the crank radius (first constraint of this
       # linkage) is too short
       if params[0] < 0.5:
           return float('inf')

       # Normal fitness calculation
       output_path = [step[-1] for step in loci]
       bbox = pl.bounding_box(output_path)
       return sum((a - t) ** 2 for a, t in zip(bbox, (0, 5, 2, 3)))

Optimizing Initial Positions
----------------------------

Sometimes the issue isn't the constraints but the initial joint positions:

.. code-block:: python

   # Save and restore initial positions
   init_coords = linkage.get_coords()

   # Optimize
   results = pl.particle_swarm_optimization(
       eval_func=rectangle_fitness,
       linkage=linkage,
       bounds=bounds,
       order_relation=min,
       n_particles=20,
       iterations=20,
       verbose=False,
   )

   # Apply results
   linkage.set_constraints(results[0].dimensions)
   linkage.set_coords(init_coords)  # Restore initial positions

Visualizing Optimization Progress
---------------------------------

Track optimization progress with the strider example pattern:

.. code-block:: python

   history = []


   def tracking_fitness(linkage, constraints, initial_positions):
       """Wrapper that records every evaluation."""
       score = rectangle_fitness(linkage, constraints, initial_positions)
       history.append((score, list(constraints), initial_positions))
       return score


   # Run optimization with tracking
   results = pl.particle_swarm_optimization(
       tracking_fitness,
       linkage,
       bounds=bounds,
       order_relation=min,
       n_particles=20,
       iterations=30,
       verbose=False,
   )

   # Analyze history: one entry per particle per iteration, after the
   # initial swarm
   scores = [h[0] for h in history]
   print(f"Evaluations: {len(scores)}, best score: {min(scores):.3f}")
   first_iteration = scores[:20]
   last_iteration = scores[-20:]
   print(f"Best of first iteration: {min(first_iteration):.3f}")
   print(f"Best of last iteration:  {min(last_iteration):.3f}")

The ``pylinkage.visualizer`` PSO plots (``parallel_coordinates_plot``,
``dashboard_layout``, ``animate_dashboard``) consume exactly this history,
grouped per iteration; see the strider example.

Async Optimization
------------------

For long-running optimizations, use the async version with progress callbacks:

.. code-block:: python

   import asyncio

   from pylinkage.optimization import particle_swarm_optimization_async


   async def optimize_with_progress():
       def on_progress(progress):
           if progress.current_iteration % 10 == 0:
               print(
                   f"Iteration {progress.current_iteration}"
                   f"/{progress.total_iterations}: "
                   f"best = {progress.best_score}"
               )

       results = await particle_swarm_optimization_async(
           eval_func=rectangle_fitness,
           linkage=linkage,
           bounds=bounds,
           order_relation=min,
           n_particles=20,
           iterations=30,
           on_progress=on_progress,
       )
       return results


   # Run async optimization
   results = asyncio.run(optimize_with_progress())
   print(f"Async result: {results[0].score:.3f}")

The optimization runs in a thread-pool executor, so the event loop stays
free; ``on_progress`` receives an ``OptimizationProgress`` with
``current_iteration``, ``total_iterations``, ``best_score`` and
``is_complete``.

Troubleshooting
---------------

**Optimization converges to poor solutions:**

- Increase ``n_particles`` for better exploration
- Widen the search bounds
- Check if your fitness function correctly penalizes bad configurations

**Many configurations are unbuildable:**

- Your bounds may include geometrically impossible regions
- Narrow the bounds around known-good configurations
- The ``@kinematic_minimization`` decorator returns ``inf`` for unbuildable configs

**Optimization is too slow:**

- Reduce ``n_particles`` or ``iterations``
- Use coarser simulation (fewer steps in ``linkage.step()``)
- Consider grid search for low-dimensional problems

Next Steps
----------

- See :doc:`../examples/examples` for complete optimization examples
- Check :py:mod:`pylinkage.optimization` for API details
- The strider example demonstrates advanced PSO visualization techniques
