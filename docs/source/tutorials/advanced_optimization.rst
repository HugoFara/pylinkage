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
       :param kwargs: Additional arguments (linkage, constraints, etc.)
       :return: Fitness score (lower is better for minimization)
       """
       # Get the locus (path) of the last joint
       output_locus = [step[-1] for step in loci]

       # Calculate your fitness metric
       score = calculate_score(output_locus)
       return score

The decorator handles:

- Setting up the linkage with candidate constraints
- Running the simulation
- Catching ``UnbuildableError`` and returning infinity

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

       return some_score

Example: Optimizing for Path Shape
----------------------------------

Let's optimize a four-bar linkage so its output traces a specific rectangular path:

.. code-block:: python

   import pylinkage as pl


   def create_linkage():
       """Create the base linkage to optimize."""
       crank = pl.Crank(
           x=0, y=1,
           joint0=(0, 0),
           angle=0.31,
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
   bounds = pl.generate_bounds(linkage.get_num_constraints())

   results = pl.particle_swarm_optimization(
       eval_func=rectangle_fitness,
       linkage=linkage,
       bounds=bounds,
   )

   # Best result
   best_score, best_constraints, best_coords = results[0]
   print(f"Best score: {best_score}")

   # Apply best constraints and visualize
   linkage.set_num_constraints(best_constraints)
   pl.show_linkage(linkage)

Particle Swarm Optimization Parameters
--------------------------------------

Fine-tune PSO behavior for better results:

.. code-block:: python

   results = pl.particle_swarm_optimization(
       eval_func=fitness_function,
       linkage=linkage,
       bounds=bounds,

       # Number of particles in the swarm
       n_particles=100,      # More particles = better exploration, slower

       # Number of iterations
       iters=200,            # More iterations = better convergence, slower

       # Starting position (optional)
       center=None,          # Use current linkage constraints as center

       # Number of dimensions (usually auto-detected)
       dimensions=None,

       # Order relation for optimization
       order_relation=min,   # min for minimization, max for maximization
   )

Generating Bounds
-----------------

The ``generate_bounds`` function creates search ranges around current values:

.. code-block:: python

   constraints = linkage.get_num_constraints()
   # Example: [0.31, 1.0, 3.0, 1.0]

   bounds = pl.generate_bounds(constraints)
   # Returns: (lower_bounds, upper_bounds)
   # Default: values * 0.5 to values * 2.0

   # Custom bounds
   bounds = pl.generate_bounds(
       constraints,
       min_ratio=0.8,    # Lower bound = value * 0.8
       max_ratio=1.2,    # Upper bound = value * 1.2
   )

   # Or define bounds manually for precise control
   bounds = (
       [0.0, 0.5, 2.0, 0.5],    # Lower bounds
       [6.28, 2.0, 5.0, 2.0],   # Upper bounds
   )

Grid Search Optimization
------------------------

For simpler problems or exhaustive search:

.. code-block:: python

   results = pl.trials_and_errors_optimization(
       eval_func=fitness_function,
       linkage=linkage,
       divisions=20,          # Points per dimension
       order_relation=min,    # min or max
   )

   # Note: Grid search is O(divisions^n) where n = number of constraints
   # Use sparingly for high-dimensional problems

Multi-Objective Optimization
----------------------------

Combine multiple objectives in your fitness function:

.. code-block:: python

   @pl.kinematic_minimization
   def multi_objective_fitness(loci, **kwargs):
       """Optimize for both path shape and mechanism size."""
       output_path = [step[-1] for step in loci]
       crank_path = [step[0] for step in loci]

       # Objective 1: Match target bounding box
       bbox = pl.bounding_box(output_path)
       target = (0, 5, 2, 3)
       shape_error = sum((a - t) ** 2 for a, t in zip(bbox, target))

       # Objective 2: Minimize total mechanism size
       all_points = [p for step in loci for p in step]
       mech_bbox = pl.bounding_box(all_points)
       mechanism_size = (mech_bbox[1] - mech_bbox[3]) * (mech_bbox[2] - mech_bbox[0])

       # Weighted combination
       return shape_error + 0.1 * mechanism_size

Constraint Preservation
-----------------------

Sometimes you want to optimize only certain constraints while keeping others fixed:

.. code-block:: python

   @pl.kinematic_minimization
   def constrained_fitness(loci, linkage=None, constraints=None, **kwargs):
       """Fitness function that enforces additional constraints."""
       # Penalize if crank arm (constraint 1) is too short
       if constraints[1] < 0.5:
           return float('inf')

       # Normal fitness calculation
       output_path = [step[-1] for step in loci]
       return calculate_path_score(output_path)

Optimizing Initial Positions
----------------------------

Sometimes the issue isn't the constraints but the initial joint positions:

.. code-block:: python

   # Save and restore initial positions
   init_coords = linkage.get_coords()

   # Optimize
   results = pl.particle_swarm_optimization(
       eval_func=fitness,
       linkage=linkage,
       bounds=bounds,
   )

   # Apply results
   linkage.set_num_constraints(results[0][1])
   linkage.set_coords(init_coords)  # Restore initial positions

Visualizing Optimization Progress
---------------------------------

Track optimization progress with the strider example pattern:

.. code-block:: python

   history = []


   def tracking_fitness(linkage, constraints, initial_positions):
       """Wrapper that records optimization history."""
       # Your actual fitness calculation
       score = my_fitness(linkage, constraints, initial_positions)
       history.append((score, list(constraints), initial_positions))
       return score


   # Run optimization with tracking
   results = pl.particle_swarm_optimization(
       lambda *args: tracking_fitness(*args),
       linkage,
       bounds=bounds,
       n_particles=50,
       iters=100,
   )

   # Analyze history
   scores = [h[0] for h in history]
   print(f"Best score: {min(scores)}")
   print(f"Score improvement: {scores[0]} -> {scores[-1]}")

Async Optimization
------------------

For long-running optimizations, use the async version with progress callbacks:

.. code-block:: python

   import asyncio


   async def optimize_with_progress():
       def on_progress(iteration, best_score):
           print(f"Iteration {iteration}: best = {best_score}")

       results = await pl.particle_swarm_optimization_async(
           eval_func=fitness,
           linkage=linkage,
           bounds=bounds,
           progress_callback=on_progress,
       )
       return results


   # Run async optimization
   results = asyncio.run(optimize_with_progress())

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

- Reduce ``n_particles`` or ``iters``
- Use coarser simulation (fewer steps in ``linkage.step()``)
- Consider grid search for low-dimensional problems

Next Steps
----------

- See :doc:`../examples/examples` for complete optimization examples
- Check :py:mod:`pylinkage.optimization` for API details
- The strider example demonstrates advanced PSO visualization techniques
