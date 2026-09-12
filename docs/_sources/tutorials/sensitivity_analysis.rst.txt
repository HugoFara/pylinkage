Sensitivity & Tolerance Analysis
================================

This tutorial covers how to analyze how manufacturing variations affect
linkage behavior. Understanding sensitivity helps identify critical dimensions
and assess whether a design is robust to manufacturing tolerances.

Overview
--------

Pylinkage provides two complementary analysis tools:

- **Sensitivity Analysis**: Measures how much each constraint dimension affects
  the output path. Identifies which dimensions are most critical.

- **Tolerance Analysis**: Monte Carlo simulation that shows the statistical
  variation in output path given manufacturing tolerances.

These tools help answer questions like:

- Which link length is most critical to the output path accuracy?
- How much output variation should I expect with ±0.1mm tolerances?
- Is my design robust enough for the manufacturing process?

Basic Setup
-----------

First, create a linkage to analyze:

.. code-block:: python

   from pylinkage.actuators import Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage

   # Create a four-bar linkage (a crank-rocker: 1 + 3.5 < 3 + 2)
   A = Ground(0.0, 0.0, name="A")
   D = Ground(3.5, 0.0, name="D")
   crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="crank")
   coupler = RRRDyad(
       anchor1=crank.output, anchor2=D,
       distance1=3.0, distance2=2.0, name="coupler",
   )
   linkage = Linkage([A, D, crank, coupler])

Pick a design with some margin: a change-point four-bar (shortest plus
longest link exactly equal to the other two) becomes unbuildable under the
smallest perturbation, and its sensitivity comes out infinite.

Sensitivity Analysis
--------------------

Basic Usage
^^^^^^^^^^^

Sensitivity analysis measures how each constraint dimension affects the output:

.. code-block:: python

   # Run sensitivity analysis with 1% perturbation
   analysis = linkage.analyze_sensitivity(delta=0.01)

   # View the most sensitive constraint
   print(f"Most sensitive: {analysis.most_sensitive}")

   # View all constraints ranked by sensitivity
   for name, sensitivity in analysis.sensitivity_ranking:
       print(f"  {name}: {sensitivity:.4f}")

Output:

.. code-block:: text

   Most sensitive: coupler_dist2
     coupler_dist2: 1.1059
     coupler_dist1: 1.1058
     crank_radius: 0.7426

A sensitivity is the mean displacement of the output path per unit change
of the constraint: 1.1 means that lengthening ``coupler_dist2`` by 0.01
moves the output joint by 0.011 on average over a cycle.

Understanding Constraint Names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constraint names are auto-generated based on joint type and name:

- ``crank_radius``: Distance from crank pivot to crank end
- ``coupler_dist1``: Distance from first anchor to coupler joint
- ``coupler_dist2``: Distance from second anchor to coupler joint

The naming follows the pattern ``{joint_name}_{constraint_type}``.

Analyzing Specific Output Joints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the last joint is analyzed. You can specify a different output:

.. code-block:: python

   # Analyze sensitivity for the crank output, by component index...
   analysis = linkage.analyze_sensitivity(output_joint=2)

   # ...or by component object
   analysis = linkage.analyze_sensitivity(output_joint=crank)
   print(f"Crank path: {analysis.sensitivity_ranking}")

Including Transmission Angle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For four-bar linkages, you can also track transmission angle sensitivity:

.. code-block:: python

   analysis = linkage.analyze_sensitivity(
       delta=0.01,
       include_transmission=True,  # the default
   )

   print(f"Baseline transmission: {analysis.baseline_transmission:.1f}°")

   # Transmission angles for each perturbation
   if analysis.perturbed_transmission is not None:
       for name, trans in zip(analysis.constraint_names, analysis.perturbed_transmission):
           print(f"  {name}: {trans:.1f}°")

Exporting to DataFrame
^^^^^^^^^^^^^^^^^^^^^^

For detailed analysis, export to a pandas DataFrame:

.. code-block:: python

   # Requires: pip install pylinkage[analysis]
   df = analysis.to_dataframe()
   print(df)

Output:

.. code-block:: text

         constraint  sensitivity  perturbed_metric  perturbed_transmission
   0   crank_radius     0.742633          0.007426               91.335097
   1  coupler_dist1     1.105773          0.033173               90.266703
   2  coupler_dist2     1.105910          0.022118               90.791385

Tolerance Analysis
------------------

Basic Usage
^^^^^^^^^^^

Tolerance analysis uses Monte Carlo simulation to assess manufacturing variability:

.. code-block:: python

   # Define tolerances for each constraint
   tolerances = {
       "crank_radius": 0.1,     # +/- 0.1 mm
       "coupler_dist1": 0.2,    # +/- 0.2 mm
       "coupler_dist2": 0.2,    # +/- 0.2 mm
   }

   # Run Monte Carlo analysis
   result = linkage.analyze_tolerance(
       tolerances=tolerances,
       n_samples=1000,
       seed=42  # For reproducibility
   )

   # View statistics
   print(f"Mean deviation: {result.mean_deviation:.4f}")
   print(f"Max deviation:  {result.max_deviation:.4f}")
   print(f"Std deviation:  {result.std_deviation:.4f}")

Output:

.. code-block:: text

   Mean deviation: 0.1727
   Max deviation:  0.3510
   Std deviation:  0.0631

Understanding the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ToleranceAnalysis`` result contains:

- ``nominal_path``: Output path at nominal dimensions (n_steps, 2)
- ``output_cloud``: All Monte Carlo samples (n_samples, n_steps, 2)
- ``mean_deviation``: Average distance from nominal path
- ``max_deviation``: Worst-case deviation
- ``std_deviation``: Standard deviation of deviations
- ``position_std``: Per-position standard deviation (n_steps,)

Visualizing the Tolerance Cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``plot_cloud()`` to visualize the output variation:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Create scatter plot of output paths
   ax = result.plot_cloud(
       show_nominal=True,  # Show nominal path as red line
       alpha=0.1           # Transparency for sample points
   )
   plt.title("Output Path Tolerance Cloud")
   plt.savefig("tolerance_cloud.png", dpi=150)
   plt.show()

This creates a scatter plot showing:

- Blue dots: Individual sample output paths
- Red line: Nominal (ideal) output path
- The spread indicates manufacturing variation

Selective Tolerance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can analyze tolerance for specific constraints:

.. code-block:: python

   # Only analyze crank radius tolerance
   result = linkage.analyze_tolerance(
       tolerances={"crank_radius": 0.1},
       n_samples=500
   )

   print(f"Crank-only max deviation: {result.max_deviation:.4f}")

Use in Optimization
-------------------

Sensitivity as Fitness Penalty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Design linkages that are insensitive to manufacturing variation:

.. code-block:: python

   import pylinkage as pl


   @pl.kinematic_minimization
   def robust_linkage(loci, linkage=None, **kwargs):
       """Optimize for path shape while minimizing sensitivity."""

       # Path shape objective: a wide, flat output path
       output_path = [step[-1] for step in loci]
       min_y, max_x, max_y, min_x = pl.bounding_box(output_path)
       path_error = (max_y - min_y) - 0.5 * (max_x - min_x)

       # Sensitivity penalty
       analysis = linkage.analyze_sensitivity(delta=0.01)
       max_sensitivity = max(analysis.sensitivities.values())

       # Combined objective: good path + low sensitivity
       return path_error + 10.0 * max_sensitivity


   bounds = pl.generate_bounds(linkage.get_constraints(), min_ratio=1.5, max_factor=1.5)
   results = pl.particle_swarm_optimization(
       robust_linkage, linkage, bounds=bounds, order_relation=min,
       n_particles=10, iterations=10, verbose=False,
   )
   print(f"Best robust score: {results[0].score:.3f}")

Sensitivity analysis simulates the linkage once per constraint, so this
fitness costs a few times a plain one; keep the swarm small or the
perturbation coarse while exploring.

Tolerance-Based Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reject designs that exceed tolerance requirements:

.. code-block:: python

   @pl.kinematic_minimization
   def tolerance_constrained(loci, linkage=None, **kwargs):
       """Optimize path, rejecting designs with excessive variation."""

       # Check tolerance
       tolerances = {"crank_radius": 0.1, "coupler_dist1": 0.2, "coupler_dist2": 0.2}
       result = linkage.analyze_tolerance(tolerances, n_samples=100, seed=0)

       if result.max_deviation > 0.5:  # Reject if max deviation > 0.5mm
           return float("inf")

       # Path objective: a wide, flat output path
       output_path = [step[-1] for step in loci]
       min_y, max_x, max_y, min_x = pl.bounding_box(output_path)
       return (max_y - min_y) - 0.5 * (max_x - min_x)


   results = pl.particle_swarm_optimization(
       tolerance_constrained, linkage, bounds=bounds, order_relation=min,
       n_particles=10, iterations=10, verbose=False,
   )
   print(f"Best tolerance-constrained score: {results[0].score:.3f}")

Returning ``inf`` (or ``-inf`` under ``@kinematic_maximization``) is how a
fitness function rejects a candidate; it is what the decorator itself
returns for an unbuildable geometry.

Practical Guidelines
--------------------

Perturbation Size
^^^^^^^^^^^^^^^^^

The ``delta`` parameter controls the relative perturbation size:

- ``delta=0.01``: 1% perturbation (recommended default)
- ``delta=0.001``: 0.1% for fine sensitivity analysis
- ``delta=0.1``: 10% for coarse/fast analysis

Smaller perturbations give more accurate local sensitivity but may be
affected by numerical noise.

Sample Count
^^^^^^^^^^^^

For tolerance analysis, the number of samples affects accuracy:

- ``n_samples=100``: Quick estimate (useful in optimization loops)
- ``n_samples=1000``: Good accuracy for design validation
- ``n_samples=10000``: High accuracy for final verification

Typical Workflow
^^^^^^^^^^^^^^^^

1. **Design**: Create linkage meeting path requirements
2. **Sensitivity**: Run ``sensitivity_analysis()`` to identify critical dimensions
3. **Focus**: Tighten tolerances on most sensitive constraints
4. **Validate**: Run ``tolerance_analysis()`` to verify acceptable variation
5. **Iterate**: If variation is too high, modify design and repeat

Example Complete Workflow
-------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from pylinkage.actuators import Crank
   from pylinkage.components import Ground
   from pylinkage.dyads import RRRDyad
   from pylinkage.simulation import Linkage

   # Create linkage
   A = Ground(0.0, 0.0, name="A")
   D = Ground(3.5, 0.0, name="D")
   crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="crank")
   coupler = RRRDyad(
       anchor1=crank.output, anchor2=D,
       distance1=3.0, distance2=2.0, name="coupler",
   )
   linkage = Linkage([A, D, crank, coupler])

   # Step 1: Sensitivity analysis
   print("=== Sensitivity Analysis ===")
   sens = linkage.analyze_sensitivity(delta=0.01)
   print(f"Most sensitive: {sens.most_sensitive}")
   for name, val in sens.sensitivity_ranking:
       print(f"  {name}: {val:.4f}")

   # Step 2: Tolerance analysis with realistic tolerances
   print("\n=== Tolerance Analysis ===")
   tolerances = {
       "crank_radius": 0.05,     # Tight tolerance (sensitive)
       "coupler_dist1": 0.1,     # Normal tolerance
       "coupler_dist2": 0.1,     # Normal tolerance
   }
   tol = linkage.analyze_tolerance(tolerances, n_samples=500, seed=42)

   print(f"Mean deviation: {tol.mean_deviation:.4f}")
   print(f"Max deviation:  {tol.max_deviation:.4f}")
   print(f"Std deviation:  {tol.std_deviation:.4f}")

   # Step 3: Visualize
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # Plot tolerance cloud
   tol.plot_cloud(ax=ax1)
   ax1.set_title("Tolerance Cloud")

   # Plot per-position std
   ax2.plot(tol.position_std)
   ax2.set_xlabel("Simulation Step")
   ax2.set_ylabel("Position Std Dev")
   ax2.set_title("Per-Position Variation")
   ax2.grid(True)

   plt.tight_layout()
   plt.savefig("tolerance_analysis.png", dpi=150)
   plt.show()

API Reference
-------------

- :py:class:`pylinkage.linkage.SensitivityAnalysis` - Sensitivity analysis results
- :py:class:`pylinkage.linkage.ToleranceAnalysis` - Tolerance analysis results
- :py:func:`pylinkage.linkage.analyze_sensitivity` - Sensitivity analysis function
- :py:func:`pylinkage.linkage.analyze_tolerance` - Tolerance analysis function
- :py:meth:`pylinkage.simulation.Linkage.analyze_sensitivity` - Convenience method
- :py:meth:`pylinkage.simulation.Linkage.analyze_tolerance` - Convenience method

Next Steps
----------

- See :doc:`advanced_optimization` for optimization techniques
- See :doc:`kinematics_optimization` for velocity/acceleration analysis
- Explore :py:mod:`pylinkage.linkage` for the complete linkage API
