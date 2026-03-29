Multi-Objective Optimization
============================

This tutorial covers true multi-objective optimization using NSGA-II/NSGA-III
algorithms to find Pareto-optimal solutions when optimizing linkages with
competing objectives.

When to Use Multi-Objective Optimization
----------------------------------------

Use multi-objective optimization when you have competing goals that cannot
be combined into a single metric without losing important trade-off information:

- **Path accuracy vs. transmission angle**: A linkage that follows a path
  perfectly might have poor force transmission characteristics.
- **Compactness vs. range of motion**: Smaller mechanisms may have limited
  travel.
- **Speed vs. smoothness**: Fast motion may introduce jerky accelerations.

Unlike weighted-sum approaches (combining objectives into one score),
multi-objective optimization returns the entire **Pareto front**—the set of
solutions where improving one objective necessarily worsens another.

Installation
------------

Multi-objective optimization requires the ``pymoo`` library:

.. code-block:: bash

   pip install pylinkage[moo]

Basic Example
-------------

Let's optimize a four-bar linkage for both path accuracy and transmission angle:

.. code-block:: python

   import pylinkage as pl
   from pylinkage.optimization import (
       multi_objective_optimization,
       kinematic_minimization,
   )


   # Create the linkage
   def create_fourbar():
       crank = pl.Crank(
           x=0, y=1, joint0=(0, 0),
           angle=0.31, distance=1, name="Crank"
       )
       output = pl.Revolute(
           x=3, y=2, joint0=crank, joint1=(3, 0),
           distance0=3, distance1=1, name="Output"
       )
       return pl.Linkage(joints=(crank, output), order=(crank, output))


   # Objective 1: Minimize path error
   @kinematic_minimization
   def path_error(loci, **kwargs):
       """Distance from output path to target circle."""
       output_path = [step[-1] for step in loci]
       target_center = (4, 1)
       target_radius = 1.5

       error = 0.0
       for x, y in output_path:
           dist = ((x - target_center[0])**2 + (y - target_center[1])**2)**0.5
           error += (dist - target_radius)**2
       return error / len(output_path)


   # Objective 2: Minimize transmission angle deviation from 90 degrees
   @kinematic_minimization
   def transmission_penalty(loci, linkage=None, **kwargs):
       """Penalize poor transmission angles."""
       # Simple approximation: use the coupler angle variation
       output_path = [step[-1] for step in loci]
       crank_path = [step[0] for step in loci]

       angles = []
       for (cx, cy), (ox, oy) in zip(crank_path, output_path):
           import math
           angle = math.atan2(oy - cy, ox - cx)
           angles.append(abs(math.degrees(angle) % 180 - 90))

       return sum(angles) / len(angles)


   # Run multi-objective optimization
   linkage = create_fourbar()
   bounds = pl.generate_bounds(linkage.get_num_constraints())

   pareto = multi_objective_optimization(
       objectives=[path_error, transmission_penalty],
       linkage=linkage,
       bounds=bounds,
       objective_names=["Path Error", "Transmission Penalty"],
       n_generations=50,
       pop_size=50,
       verbose=True,
   )

   print(f"Found {len(pareto)} Pareto-optimal solutions")

Understanding the Pareto Front
------------------------------

The ``ParetoFront`` object contains all non-dominated solutions:

.. code-block:: python

   # Iterate over solutions
   for solution in pareto:
       print(f"Scores: {solution.scores}")
       print(f"Constraints: {solution.dimensions}")

   # Get scores as numpy array
   scores = pareto.scores_array()  # Shape: (n_solutions, n_objectives)

   # Number of objectives
   print(f"Objectives: {pareto.n_objectives}")

Pareto Dominance
^^^^^^^^^^^^^^^^

A solution **dominates** another if it is at least as good in all objectives
and strictly better in at least one:

.. code-block:: python

   sol_a = pareto[0]
   sol_b = pareto[1]

   if sol_a.dominates(sol_b):
       print("Solution A is strictly better than B")
   else:
       print("Solutions are non-dominated (trade-offs)")

Visualizing Trade-Offs
----------------------

Plot the Pareto front to understand the trade-offs:

2D Pareto Front
^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   # Default 2D plot
   fig = pareto.plot()
   plt.savefig("pareto_front.png")
   plt.show()

   # Custom styling
   fig = pareto.plot(s=100, alpha=0.8, c='blue')
   plt.show()

3D Pareto Front (3+ Objectives)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For 3 objectives, pass 3 indices
   fig = pareto.plot(objective_indices=(0, 1, 2))
   plt.show()

Selecting a Solution
--------------------

Best Compromise
^^^^^^^^^^^^^^^

Select a balanced solution using normalized weighted sum:

.. code-block:: python

   # Equal weight on all objectives
   best = pareto.best_compromise()

   # Custom weights (prioritize path error)
   best = pareto.best_compromise(weights=[0.8, 0.2])

   # Apply to linkage
   linkage.set_num_constraints(best.dimensions)
   pl.show_linkage(linkage)

Filtering Solutions
^^^^^^^^^^^^^^^^^^^

Reduce the Pareto front to a manageable subset using crowding distance:

.. code-block:: python

   # Keep 10 well-distributed solutions
   filtered = pareto.filter(max_solutions=10)

   print(f"Reduced from {len(pareto)} to {len(filtered)} solutions")

Hypervolume Indicator
^^^^^^^^^^^^^^^^^^^^^

Measure the quality of the Pareto front:

.. code-block:: python

   # Reference point should be worse than all solutions
   reference = [10.0, 90.0]  # Adjust based on your objective ranges

   hv = pareto.hypervolume(reference_point=reference)
   print(f"Hypervolume: {hv}")

Algorithm Selection
-------------------

NSGA-II (Default)
^^^^^^^^^^^^^^^^^

Best for 2-3 objectives:

.. code-block:: python

   pareto = multi_objective_optimization(
       objectives=[obj1, obj2],
       linkage=linkage,
       algorithm="nsga2",  # Default
       n_generations=100,
       pop_size=100,
   )

NSGA-III
^^^^^^^^

Better for many objectives (4+):

.. code-block:: python

   pareto = multi_objective_optimization(
       objectives=[obj1, obj2, obj3, obj4],
       linkage=linkage,
       algorithm="nsga3",
       n_generations=100,
       pop_size=100,
   )

Advanced Usage
--------------

Combining with Transmission Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the built-in transmission angle analysis:

.. code-block:: python

   @kinematic_minimization
   def transmission_objective(loci, linkage=None, **kwargs):
       """Use built-in transmission analysis."""
       analysis = linkage.analyze_transmission()
       # Minimize deviation from ideal 90 degrees
       return abs(90 - analysis.mean_angle)


   @kinematic_minimization
   def worst_transmission(loci, linkage=None, **kwargs):
       """Minimize the worst transmission angle."""
       analysis = linkage.analyze_transmission()
       # Return how far the worst angle is from acceptable range
       if analysis.min_angle < 40:
           return 40 - analysis.min_angle
       if analysis.max_angle > 140:
           return analysis.max_angle - 140
       return 0

Three-Objective Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @kinematic_minimization
   def path_error(loci, **kwargs):
       # ... path accuracy metric
       return error


   @kinematic_minimization
   def transmission_penalty(loci, linkage=None, **kwargs):
       # ... transmission angle metric
       return penalty


   @kinematic_minimization
   def compactness(loci, **kwargs):
       """Minimize mechanism bounding box area."""
       all_points = [p for step in loci for p in step]
       bbox = pl.bounding_box(all_points)
       width = bbox[1] - bbox[3]  # max_x - min_x
       height = bbox[2] - bbox[0]  # max_y - min_y
       return width * height


   pareto = multi_objective_optimization(
       objectives=[path_error, transmission_penalty, compactness],
       linkage=linkage,
       objective_names=["Path Error", "Transmission", "Size"],
       algorithm="nsga2",
       n_generations=100,
   )

   # 3D visualization
   fig = pareto.plot(objective_indices=(0, 1, 2))

Reproducibility
^^^^^^^^^^^^^^^

Set a random seed for reproducible results:

.. code-block:: python

   pareto = multi_objective_optimization(
       objectives=[obj1, obj2],
       linkage=linkage,
       seed=42,
   )

Comparison: Single vs Multi-Objective
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Approach
     - When to Use
     - Trade-offs
   * - **Weighted Sum**
     - Clear priority between objectives
     - Single solution; weights are subjective
   * - **Multi-Objective**
     - Exploring trade-offs; no clear priority
     - Multiple solutions; requires selection

Troubleshooting
---------------

**Empty Pareto front:**

- All candidate solutions may be unbuildable
- Widen the search bounds
- Increase population size

**Poor convergence:**

- Increase ``n_generations``
- Increase ``pop_size``
- Check that objectives have similar scales

**Slow optimization:**

- Reduce ``pop_size`` and ``n_generations``
- Simplify fitness functions
- Use fewer simulation steps

Next Steps
----------

- See :doc:`advanced_optimization` for single-objective techniques
- See :doc:`sensitivity_analysis` for understanding parameter sensitivity
- Check :py:mod:`pylinkage.optimization` for API reference
