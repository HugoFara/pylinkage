#!/usr/bin/env python3
"""
Symbolic Linkage Computation Demo.

This demo shows how to use symbolic computation for linkage analysis,
including closed-form trajectory expressions, gradient-based optimization,
parameter sensitivity analysis, and performance comparisons.

To run:
    uv run python docs/examples/symbolic_linkage_demo.py

Theory:
    Symbolic computation represents linkage kinematics as algebraic expressions
    in terms of the input angle (theta) and link lengths. This enables:

    1. **Closed-form solutions**: Exact trajectory expressions like
       x(theta) = L1*cos(theta) + L2*cos(phi(theta))

    2. **Analytical gradients**: Derivatives computed symbolically, not numerically,
       providing exact gradients for optimization.

    3. **Parameter sensitivity**: Understanding how changes in link lengths
       affect the output motion through Jacobian analysis.

Applications:
    - Design space exploration with exact derivatives
    - Sensitivity analysis for tolerance specification
    - Understanding linkage behavior analytically
    - Efficient gradient-based optimization
    - Educational visualization of kinematic equations

References:
    - Erdman, A. & Sandor, G. "Mechanism Design: Analysis and Synthesis"
    - McCarthy, J.M. "Geometric Design of Linkages"
"""

import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import pylinkage as pl
from pylinkage.symbolic import (
    SymbolicLinkage,
    SymbolicOptimizer,
    SymCrank,
    SymRevolute,
    SymStatic,
    compute_trajectory_numeric,
    create_trajectory_functions,
    fourbar_symbolic,
    solve_linkage_symbolically,
    symbolic_gradient,
    theta,
)


def demo_creating_symbolic_linkages():
    """Demonstrate three methods to create symbolic linkages.

    Shows:
    1. Using fourbar_symbolic() convenience function
    2. Building from symbolic joint classes
    3. Converting from numeric linkage
    """
    print("=" * 70)
    print("Demo 1: Creating Symbolic Linkages (3 Methods)")
    print("=" * 70)

    # Method 1: Using fourbar_symbolic
    print("\n--- Method 1: fourbar_symbolic() convenience function ---")
    linkage1 = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",  # Symbolic parameter
        coupler_length="L2",
        rocker_length="L3",
    )
    print(f"Created linkage with {len(linkage1.joints)} joints")
    print(f"Symbolic parameters: {[str(p) for p in linkage1.parameters]}")
    print(f"Joint names: {[j.name for j in linkage1.joints]}")

    # Method 2: Building from joints
    print("\n--- Method 2: Building from SymJoint classes ---")
    L1, L2, L3 = sp.symbols("L1 L2 L3", positive=True, real=True)

    ground_A = SymStatic(x=0, y=0, name="A")
    crank_B = SymCrank(parent=ground_A, radius=L1, name="B")
    ground_D = SymStatic(x=4, y=0, name="D")
    coupler_C = SymRevolute(parent0=crank_B, parent1=ground_D, distance0=L2, distance1=L3, name="C")

    linkage2 = SymbolicLinkage(
        joints=[ground_A, crank_B, ground_D, coupler_C],
    )
    print(f"Created linkage with custom joint names: {[j.name for j in linkage2.joints]}")
    print(f"Parameters: {[str(p) for p in linkage2.parameters]}")

    print("\nBoth construction paths produce equivalent symbolic representations.")
    return linkage1


def demo_closed_form_expressions():
    """Demonstrate closed-form trajectory expressions.

    Shows:
    - Getting symbolic expressions for joint positions
    - Expression structure and complexity
    - Simplification techniques
    - Taking derivatives symbolically
    """
    print("\n" + "=" * 70)
    print("Demo 2: Closed-Form Trajectory Expressions")
    print("=" * 70)

    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    # Get symbolic solutions
    solutions = solve_linkage_symbolically(linkage)

    print("\nClosed-form expressions for each joint:")
    print("-" * 70)
    for joint_name, (x_expr, y_expr) in solutions.items():
        x_str = str(x_expr)
        y_str = str(y_expr)
        # Truncate long expressions for display
        if len(x_str) > 60:
            x_str = x_str[:57] + "..."
        if len(y_str) > 60:
            y_str = y_str[:57] + "..."
        print(f"\n{joint_name}:")
        print(f"  x(theta) = {x_str}")
        print(f"  y(theta) = {y_str}")

    # Show expression complexity
    print("\n" + "-" * 70)
    print("Expression complexity (operation count):")
    for joint_name, (x_expr, y_expr) in solutions.items():
        x_ops = sp.count_ops(x_expr)
        y_ops = sp.count_ops(y_expr)
        print(f"  {joint_name}: x has {x_ops} ops, y has {y_ops} ops")

    # Demonstrate derivatives
    print("\n" + "-" * 70)
    print("Symbolic derivatives (velocity expressions):")
    x_crank, y_crank = solutions["B"]
    dx_dtheta = sp.diff(x_crank, theta)
    dy_dtheta = sp.diff(y_crank, theta)
    print("\nCrank velocity (B):")
    print(f"  dx/dtheta = {dx_dtheta}")
    print(f"  dy/dtheta = {dy_dtheta}")
    print(f"  Speed = sqrt({sp.simplify(dx_dtheta**2 + dy_dtheta**2)})")

    # Show parameter sensitivity
    print("\n" + "-" * 70)
    print("Parameter sensitivity (partial derivatives):")
    x_output, y_output = solutions["C"]
    params = [sp.Symbol("L1"), sp.Symbol("L2"), sp.Symbol("L3")]

    print("\ndx_C/dL_i (sensitivity of output x to link lengths):")
    for p in params:
        dx_dp = sp.diff(x_output, p)
        # Evaluate at specific point for numerical insight
        dx_dp_val = dx_dp.subs(
            {sp.Symbol("L1"): 1, sp.Symbol("L2"): 3, sp.Symbol("L3"): 3, theta: 0}
        )
        print(f"  dx/d{p} at (1,3,3,theta=0) = {float(dx_dp_val):.4f}")

    return solutions


def demo_numeric_trajectory_evaluation():
    """Demonstrate numeric trajectory computation from symbolic expressions.

    Shows:
    - Evaluating symbolic expressions at specific parameter values
    - Trajectory shape and statistics
    - Multiple parameter configurations comparison
    """
    print("\n" + "=" * 70)
    print("Demo 3: Numeric Trajectory Evaluation")
    print("=" * 70)

    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    # Define parameter configurations
    configurations = [
        {"L1": 1.0, "L2": 3.0, "L3": 3.0, "name": "Standard (1, 3, 3)"},
        {"L1": 0.5, "L2": 3.5, "L3": 3.0, "name": "Short crank (0.5, 3.5, 3)"},
        {"L1": 1.5, "L2": 2.5, "L3": 3.5, "name": "Long crank (1.5, 2.5, 3.5)"},
    ]

    theta_values = np.linspace(0, 2 * np.pi, 360)

    print("\nTrajectory statistics for different configurations:")
    print("-" * 70)
    print(f"{'Configuration':<30} {'X range':<15} {'Y range':<15} {'Path length':<12}")
    print("-" * 70)

    all_trajectories = []
    for config in configurations:
        params = {k: v for k, v in config.items() if k != "name"}

        try:
            trajectories = compute_trajectory_numeric(linkage, params, theta_values)
            output = trajectories["C"]

            # Compute statistics
            x_min, x_max = np.min(output[:, 0]), np.max(output[:, 0])
            y_min, y_max = np.min(output[:, 1]), np.max(output[:, 1])
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Path length
            diffs = np.diff(output, axis=0)
            path_length = np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2))

            print(
                f"{config['name']:<30} {x_range:>6.3f}"
                f"       {y_range:>6.3f}       {path_length:>6.3f}"
            )
            all_trajectories.append((config["name"], output))
        except Exception as e:
            print(f"{config['name']:<30} UNBUILDABLE: {e}")

    # Show buildability checking
    print("\n" + "-" * 70)
    print("Buildability checking:")

    test_params = [
        {"L1": 1.0, "L2": 3.0, "L3": 3.0},  # Valid
        {"L1": 5.0, "L2": 1.0, "L3": 1.0},  # Invalid (crank too long)
    ]

    for params in test_params:
        is_buildable = True
        # Test at multiple angles
        for test_theta in np.linspace(0, 2 * np.pi, 8):
            try:
                traj = compute_trajectory_numeric(linkage, params, np.array([test_theta]))
                # Check for NaN
                if np.any(np.isnan(list(traj.values())[0])):
                    is_buildable = False
                    break
            except Exception:
                is_buildable = False
                break

        status = "Buildable" if is_buildable else "Not buildable"
        print(f"  L1={params['L1']}, L2={params['L2']}, L3={params['L3']}: {status}")

    return all_trajectories


def demo_performance_comparison():
    """Compare performance of different evaluation methods.

    Shows:
    - Direct symbolic evaluation time
    - Pre-compiled function evaluation time
    - Numeric (numba) solver time
    - Speedup factors
    """
    print("\n" + "=" * 70)
    print("Demo 4: Performance Comparison")
    print("=" * 70)

    # Number of evaluations for benchmarking
    n_evals = 50
    n_points = 360

    # Setup symbolic linkage
    linkage = fourbar_symbolic(ground_length=4, crank_length=1, coupler_length=3, rocker_length=3)
    params: dict[str, Any] = {}  # No symbolic params - all numeric
    theta_vals = np.linspace(0, 2 * np.pi, n_points)

    print(f"\nBenchmarking {n_evals} evaluations with {n_points} points each:")
    print("-" * 70)

    # Method 1: Direct symbolic evaluation
    start = time.perf_counter()
    for _ in range(n_evals):
        compute_trajectory_numeric(linkage, params, theta_vals)
    direct_time = time.perf_counter() - start
    print(
        f"Direct symbolic evaluation:   {direct_time:.4f}s"
        f" ({direct_time / n_evals * 1000:.2f}ms/eval)"
    )

    # Method 2: Pre-compiled functions
    traj_funcs = create_trajectory_functions(linkage)
    start = time.perf_counter()
    for _ in range(n_evals):
        for joint_name in traj_funcs:
            x_func, y_func, param_symbols = traj_funcs[joint_name]
            # No params for all-numeric linkage
            x_func(theta_vals)
            y_func(theta_vals)
    compiled_time = time.perf_counter() - start
    print(
        f"Pre-compiled functions:       {compiled_time:.4f}s"
        f" ({compiled_time / n_evals * 1000:.2f}ms/eval)"
    )

    # Method 3: Numeric (numba) solver for comparison
    from pylinkage.actuators import Crank as _Crank
    from pylinkage.components import Ground as _Ground
    from pylinkage.dyads import RRRDyad as _RRRDyad
    from pylinkage.simulation import Linkage as _SimLinkage

    _A = _Ground(0.0, 0.0, name="A")
    _D = _Ground(4.0, 0.0, name="D")
    _crank = _Crank(anchor=_A, radius=1.0, angular_velocity=0.0, name="B")
    _dyad = _RRRDyad(
        anchor1=_crank.output, anchor2=_D,
        distance1=3.0, distance2=3.0, name="C",
    )
    numeric_linkage = _SimLinkage([_A, _D, _crank, _dyad])

    start = time.perf_counter()
    for _ in range(n_evals):
        numeric_linkage.step(iterations=n_points)
    numeric_time = time.perf_counter() - start
    print(
        f"Numeric (numba) solver:       {numeric_time:.4f}s"
        f" ({numeric_time / n_evals * 1000:.2f}ms/eval)"
    )

    # Speedup analysis
    print("\n" + "-" * 70)
    print("Speedup analysis:")
    print(f"  Compiled vs Direct:     {direct_time / compiled_time:.1f}x faster")
    print(f"  Numeric vs Direct:      {direct_time / numeric_time:.1f}x faster")
    print(f"  Numeric vs Compiled:    {compiled_time / numeric_time:.1f}x faster")

    print("\nRecommendation:")
    print("  - Use numeric solver for optimization loops (fastest)")
    print("  - Use compiled functions for repeated symbolic evaluation")
    print("  - Use direct evaluation for one-off analyses")

    return {
        "direct": direct_time,
        "compiled": compiled_time,
        "numeric": numeric_time,
    }


def demo_gradient_optimization():
    """Demonstrate gradient-based optimization using symbolic gradients.

    Shows:
    - Setting up SymbolicOptimizer
    - Defining symbolic objective functions
    - Running optimization with statistics
    - Comparing initial vs optimized linkage

    Note: SymbolicOptimizer uses symbolic expressions, not numeric arrays.
    The objective function receives trajectory expressions (SymPy) and
    must return a symbolic expression that will be differentiated.
    """
    print("\n" + "=" * 70)
    print("Demo 5: Gradient-Based Optimization")
    print("=" * 70)

    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    # Define symbolic objective: minimize squared distance from target point (3, 1.5)
    # This is a symbolic expression, not a numeric computation!
    def minimize_distance_to_target(trajectories):
        """Minimize squared distance from output to target point (3, 1.5).

        This returns a SYMBOLIC expression that will be evaluated at many theta
        values and averaged. It is then differentiated for gradient optimization.
        """
        x, y = trajectories["C"]  # These are SymPy expressions
        target_x, target_y = 3.0, 1.5
        return (x - target_x) ** 2 + (y - target_y) ** 2

    initial_params = {"L1": 1.0, "L2": 3.0, "L3": 3.0}
    bounds = {"L1": (0.3, 2.0), "L2": (1.5, 5.0), "L3": (1.5, 5.0)}

    print("\nObjective: Minimize average squared distance to target point (3, 1.5)")
    print(
        f"Initial parameters: L1={initial_params['L1']},"
        f" L2={initial_params['L2']}, L3={initial_params['L3']}"
    )
    print(f"Bounds: L1={bounds['L1']}, L2={bounds['L2']}, L3={bounds['L3']}")

    # Compute initial objective numerically for comparison
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    initial_traj = compute_trajectory_numeric(linkage, initial_params, theta_vals)
    target = np.array([3.0, 1.5])
    initial_objective = np.mean(np.sum((initial_traj["C"] - target) ** 2, axis=1))
    print(f"\nInitial mean squared distance: {initial_objective:.4f}")

    # Run optimization
    print("\n" + "-" * 70)
    print("Running gradient-based optimization...")

    optimizer = SymbolicOptimizer(linkage, minimize_distance_to_target)

    start_time = time.perf_counter()
    result = optimizer.optimize(initial_params=initial_params, bounds=bounds)
    opt_time = time.perf_counter() - start_time

    print(f"\nOptimization completed in {opt_time:.3f}s")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")

    if result.success:
        print("\nOptimal parameters:")
        for param, value in result.params.items():
            print(f"  {param}: {value:.4f}")
        print(f"\nFinal mean squared distance: {result.objective_value:.4f}")
        print(f"Improvement: {(1 - result.objective_value / initial_objective) * 100:.1f}%")

        # Compare trajectories
        print("\n" + "-" * 70)
        print("Trajectory comparison (initial vs optimized):")

        optimal_traj = compute_trajectory_numeric(linkage, result.params, theta_vals)

        print(f"\n{'Metric':<20} {'Initial':<12} {'Optimized':<12} {'Change':<10}")
        print("-" * 55)

        init_out = initial_traj["C"]
        opt_out = optimal_traj["C"]

        comparisons = [
            (
                "Mean dist to (3,1.5)",
                np.mean(np.sqrt(np.sum((init_out - target) ** 2, axis=1))),
                np.mean(np.sqrt(np.sum((opt_out - target) ** 2, axis=1))),
            ),
            (
                "Min dist to (3,1.5)",
                np.min(np.sqrt(np.sum((init_out - target) ** 2, axis=1))),
                np.min(np.sqrt(np.sum((opt_out - target) ** 2, axis=1))),
            ),
            (
                "X range",
                np.ptp(init_out[:, 0]),
                np.ptp(opt_out[:, 0]),
            ),
            (
                "Y range",
                np.ptp(init_out[:, 1]),
                np.ptp(opt_out[:, 1]),
            ),
            (
                "Path length",
                np.sum(np.sqrt(np.sum(np.diff(init_out, axis=0) ** 2, axis=1))),
                np.sum(np.sqrt(np.sum(np.diff(opt_out, axis=0) ** 2, axis=1))),
            ),
        ]

        for metric, init_val, opt_val in comparisons:
            change = (opt_val - init_val) / init_val * 100 if init_val != 0 else 0
            print(f"{metric:<20} {init_val:<12.4f} {opt_val:<12.4f} {change:>+8.1f}%")

        return result.params, initial_traj, optimal_traj
    else:
        print(f"Optimization failed: {result.message}")
        return None, initial_traj, None


def demo_sensitivity_analysis():
    """Demonstrate parameter sensitivity analysis using symbolic Jacobian.

    Shows:
    - Computing symbolic Jacobian matrix
    - Evaluating sensitivity at specific configurations
    - Identifying most sensitive parameters
    - Tolerance implications
    """
    print("\n" + "=" * 70)
    print("Demo 6: Parameter Sensitivity Analysis")
    print("=" * 70)

    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    # Get symbolic expressions
    solutions = solve_linkage_symbolically(linkage)
    x_output, y_output = solutions["C"]

    # Parameters to analyze
    params = [sp.Symbol("L1"), sp.Symbol("L2"), sp.Symbol("L3")]
    param_values = {"L1": 1.0, "L2": 3.0, "L3": 3.0}

    print("\nComputing parameter sensitivity for output joint C...")
    print("-" * 70)

    # Compute symbolic gradients
    grad_x = symbolic_gradient(x_output, params)
    _ = symbolic_gradient(y_output, params)  # Can be used for y-sensitivity

    # Evaluate at multiple angles
    test_angles = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

    print(f"\n{'Angle (deg)':<12} {'dC_x/dL1':<12} {'dC_x/dL2':<12} {'dC_x/dL3':<12}")
    print("-" * 50)

    sensitivity_data = []
    for angle in test_angles:
        subs = {**{sp.Symbol(k): v for k, v in param_values.items()}, theta: angle}

        row = [math.degrees(angle)]
        for g in grad_x:
            try:
                val = float(g.subs(subs))
                row.append(val)
            except (TypeError, ValueError):
                row.append(float("nan"))

        if not any(np.isnan(row[1:])):
            sensitivity_data.append(row)
            print(f"{row[0]:<12.0f} {row[1]:<12.4f} {row[2]:<12.4f} {row[3]:<12.4f}")

    # Compute average sensitivities
    if sensitivity_data:
        sens_array = np.array(sensitivity_data)
        avg_sens = np.mean(np.abs(sens_array[:, 1:]), axis=0)

        print("\n" + "-" * 70)
        print("Average absolute sensitivity:")
        for i, p in enumerate(["L1", "L2", "L3"]):
            print(f"  |dC_x/d{p}| = {avg_sens[i]:.4f}")

        most_sensitive = ["L1", "L2", "L3"][np.argmax(avg_sens)]
        print(f"\nMost sensitive parameter: {most_sensitive}")
        print("(Requires tightest manufacturing tolerance)")

    # Tolerance analysis
    print("\n" + "-" * 70)
    print("Tolerance analysis:")
    print("If manufacturing tolerance is +/- 0.1 on each link length:")

    tolerance = 0.1
    theta_vals = np.linspace(0, 2 * np.pi, 100)

    # Nominal trajectory
    nominal_traj = compute_trajectory_numeric(linkage, param_values, theta_vals)
    nominal_out = nominal_traj["C"]

    # Perturbed trajectories
    max_deviations = []
    for param_name in ["L1", "L2", "L3"]:
        perturbed_params = param_values.copy()
        perturbed_params[param_name] += tolerance

        try:
            perturbed_traj = compute_trajectory_numeric(linkage, perturbed_params, theta_vals)
            perturbed_out = perturbed_traj["C"]

            # Max deviation
            deviation = np.max(
                np.sqrt(
                    (nominal_out[:, 0] - perturbed_out[:, 0]) ** 2
                    + (nominal_out[:, 1] - perturbed_out[:, 1]) ** 2
                )
            )
            max_deviations.append((param_name, deviation))
            print(f"  +{tolerance} on {param_name}: max output deviation = {deviation:.4f}")
        except Exception:
            print(f"  +{tolerance} on {param_name}: linkage becomes unbuildable")

    if max_deviations:
        worst_param = max(max_deviations, key=lambda x: x[1])
        print(f"\nWorst-case parameter: {worst_param[0]} (deviation: {worst_param[1]:.4f})")

    return sensitivity_data


def demo_visualize_results():
    """Visualize symbolic linkage analysis results.

    Creates a comprehensive figure showing:
    - Coupler curves for different configurations
    - Parameter space exploration
    - Optimization trajectory
    """
    print("\n" + "=" * 70)
    print("Demo 7: Visualization of Results")
    print("=" * 70)

    linkage = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Symbolic Linkage Analysis Results", fontsize=14, fontweight="bold")

    theta_vals = np.linspace(0, 2 * np.pi, 360)

    # Plot 1: Coupler curves for different L1 values
    ax1 = axes[0, 0]
    ax1.set_title("Coupler Curves: Varying Crank Length (L1)")

    L1_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(L1_values)))

    for L1_val, color in zip(L1_values, colors, strict=True):
        try:
            params = {"L1": L1_val, "L2": 3.0, "L3": 3.0}
            traj = compute_trajectory_numeric(linkage, params, theta_vals)
            output = traj["C"]
            ax1.plot(output[:, 0], output[:, 1], color=color, label=f"L1={L1_val}")
        except Exception:
            pass

    # Plot ground pivots
    ax1.plot([0, 4], [0, 0], "ko", markersize=10, label="Ground pivots")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coupler curves for different L2 values
    ax2 = axes[0, 1]
    ax2.set_title("Coupler Curves: Varying Coupler Length (L2)")

    L2_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(L2_values)))

    for L2_val, color in zip(L2_values, colors, strict=True):
        try:
            params = {"L1": 1.0, "L2": L2_val, "L3": 3.0}
            traj = compute_trajectory_numeric(linkage, params, theta_vals)
            output = traj["C"]
            ax2.plot(output[:, 0], output[:, 1], color=color, label=f"L2={L2_val}")
        except Exception:
            pass

    ax2.plot([0, 4], [0, 0], "ko", markersize=10)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Workspace area heatmap
    ax3 = axes[1, 0]
    ax3.set_title("Workspace Area: L1 vs L2 (L3=3.0)")

    L1_range = np.linspace(0.3, 1.8, 20)
    L2_range = np.linspace(1.5, 4.5, 20)
    workspace_areas = np.zeros((len(L2_range), len(L1_range)))

    for i, L2_val in enumerate(L2_range):
        for j, L1_val in enumerate(L1_range):
            try:
                params = {"L1": L1_val, "L2": L2_val, "L3": 3.0}
                traj = compute_trajectory_numeric(linkage, params, theta_vals)
                output = traj["C"]
                area = np.ptp(output[:, 0]) * np.ptp(output[:, 1])
                workspace_areas[i, j] = area
            except Exception:
                workspace_areas[i, j] = np.nan

    im = ax3.imshow(
        workspace_areas,
        extent=[L1_range[0], L1_range[-1], L2_range[0], L2_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    ax3.set_xlabel("Crank Length (L1)")
    ax3.set_ylabel("Coupler Length (L2)")
    plt.colorbar(im, ax=ax3, label="Workspace Area")

    # Plot 4: Path length vs crank angle for different configs
    ax4 = axes[1, 1]
    ax4.set_title("Cumulative Path Length vs Crank Angle")

    configs = [
        ({"L1": 0.5, "L2": 3.0, "L3": 3.0}, "Short crank"),
        ({"L1": 1.0, "L2": 3.0, "L3": 3.0}, "Standard"),
        ({"L1": 1.5, "L2": 3.0, "L3": 3.0}, "Long crank"),
    ]

    for params, label in configs:
        try:
            traj = compute_trajectory_numeric(linkage, params, theta_vals)
            output = traj["C"]

            # Cumulative path length
            diffs = np.diff(output, axis=0)
            segment_lengths = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
            cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])

            ax4.plot(np.degrees(theta_vals), cumulative, label=label)
        except Exception:
            pass

    ax4.set_xlabel("Crank Angle (degrees)")
    ax4.set_ylabel("Cumulative Path Length")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    print("\nDisplaying visualization...")
    print("Close the plot window to continue.")
    plt.show()

    return fig


def demo_symbolic_vs_pso_comparison():
    """Compare symbolic gradient-based optimization with PSO.

    Shows:
    - Setup for both methods
    - Convergence comparison
    - Solution quality comparison
    - When to use each approach
    """
    print("\n" + "=" * 70)
    print("Demo 8: Symbolic Gradient vs PSO Optimization")
    print("=" * 70)

    # Common setup
    initial_params = {"L1": 1.0, "L2": 3.0, "L3": 3.0}
    bounds_dict = {"L1": (0.3, 2.0), "L2": (1.5, 5.0), "L3": (1.5, 5.0)}

    # Symbolic objective function (receives SymPy expressions)
    def symbolic_objective(trajectories):
        """Symbolic objective: squared distance from y=1.5 line."""
        x, y = trajectories["C"]
        return (y - 1.5) ** 2

    print("\nObjective: Minimize average squared distance of output to y=1.5")
    print(
        f"Initial: L1={initial_params['L1']}, L2={initial_params['L2']}, L3={initial_params['L3']}"
    )

    # Method 1: Symbolic gradient-based
    print("\n" + "-" * 70)
    print("Method 1: Symbolic Gradient Optimization")

    linkage_sym = fourbar_symbolic(
        ground_length=4,
        crank_length="L1",
        coupler_length="L2",
        rocker_length="L3",
    )

    optimizer = SymbolicOptimizer(linkage_sym, symbolic_objective)

    start_time = time.perf_counter()
    sym_result = optimizer.optimize(initial_params=initial_params, bounds=bounds_dict)
    sym_time = time.perf_counter() - start_time

    print(f"  Time: {sym_time:.3f}s")
    print(f"  Iterations: {sym_result.iterations}")
    print(f"  Success: {sym_result.success}")
    if sym_result.success:
        print(
            f"  Result: L1={sym_result.params['L1']:.4f},"
            f" L2={sym_result.params['L2']:.4f},"
            f" L3={sym_result.params['L3']:.4f}"
        )
        print(f"  Objective: {sym_result.objective_value:.6f}")

    # Method 2: PSO (numeric)
    print("\n" + "-" * 70)
    print("Method 2: PSO Optimization")

    # Use bounds that are more likely to produce buildable linkages
    # The bounds should ensure L2 + L3 > ground_length for all configurations
    pso_bounds_dict = {"L1": (0.5, 1.5), "L2": (2.5, 4.0), "L3": (2.5, 4.0)}

    # angle is the rotation step per iteration (2*pi/100 for 100 steps per rotation)
    from pylinkage.actuators import Crank as _Crank
    from pylinkage.components import Ground as _Ground
    from pylinkage.dyads import RRRDyad as _RRRDyad
    from pylinkage.simulation import Linkage as _SimLinkage

    angle_step = 2 * np.pi / 100
    _A = _Ground(0.0, 0.0, name="A")
    _D = _Ground(4.0, 0.0, name="D")
    _crank = _Crank(anchor=_A, radius=initial_params["L1"], angular_velocity=angle_step)
    _dyad = _RRRDyad(
        anchor1=_crank.output,
        anchor2=_D,
        distance1=initial_params["L2"],
        distance2=initial_params["L3"],
    )
    numeric_linkage = _SimLinkage([_A, _D, _crank, _dyad])

    @pl.kinematic_minimization
    def pso_objective(loci, **kwargs):
        """PSO-compatible objective function."""
        output_path = np.array([step[-1] for step in loci])
        return np.mean((output_path[:, 1] - 1.5) ** 2)

    # bounds is a tuple of (lower_bounds, upper_bounds)
    bounds_tuple = (
        [pso_bounds_dict["L1"][0], pso_bounds_dict["L2"][0], pso_bounds_dict["L3"][0]],  # lower
        [pso_bounds_dict["L1"][1], pso_bounds_dict["L2"][1], pso_bounds_dict["L3"][1]],  # upper
    )

    start_time = time.perf_counter()
    try:
        pso_result = pl.particle_swarm_optimization(
            eval_func=pso_objective,
            linkage=numeric_linkage,
            bounds=bounds_tuple,
            n_particles=30,
            iters=50,
            verbose=False,
        )
        pso_time = time.perf_counter() - start_time

        if pso_result and len(pso_result) >= 2:
            pso_score, pso_params = pso_result[0], pso_result[1]
            print(f"  Time: {pso_time:.3f}s")
            print("  Iterations: 50 (30 particles each)")
            print(
                f"  Result: L1={pso_params[0]:.4f}, L2={pso_params[1]:.4f}, L3={pso_params[2]:.4f}"
            )
            print(f"  Objective: {pso_score:.6f}")
        else:
            print("  PSO returned empty result - no buildable linkages found in search space")
            pso_score = float("inf")
            pso_params = None
    except Exception as e:
        pso_time = time.perf_counter() - start_time
        print(f"  PSO failed: {e}")
        pso_score = float("inf")
        pso_params = None

    # Comparison summary
    print("\n" + "-" * 70)
    print("Comparison Summary:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Symbolic Gradient':<20} {'PSO':<20}")
    print("-" * 70)
    print(f"{'Time':<25} {sym_time:<20.3f} {pso_time:<20.3f}")

    sym_obj_str = f"{sym_result.objective_value:.6f}" if sym_result.success else "N/A"
    pso_obj_str = f"{pso_score:.6f}" if pso_params is not None else "N/A"
    print(f"{'Objective value':<25} {sym_obj_str:<20} {pso_obj_str:<20}")

    if sym_time > 0 and pso_params is not None:
        speedup = pso_time / sym_time
        print(f"\nSymbolic gradient is {speedup:.1f}x faster than PSO")

    print("\nWhen to use each:")
    print("  Symbolic Gradient:")
    print("    - Smooth, differentiable objectives")
    print("    - Need fast convergence")
    print("    - Local optimization from good starting point")
    print("  PSO:")
    print("    - Non-differentiable or noisy objectives")
    print("    - Global search needed")
    print("    - Multiple local minima expected")

    return sym_result, pso_result


def main():
    """Run all symbolic linkage demos."""
    print("\n" + "#" * 70)
    print("# SYMBOLIC LINKAGE COMPUTATION DEMOS")
    print("# Closed-form expressions, gradients, and optimization")
    print("#" * 70)

    # Run all demos
    demo_creating_symbolic_linkages()
    demo_closed_form_expressions()
    demo_numeric_trajectory_evaluation()
    demo_performance_comparison()
    demo_gradient_optimization()
    demo_sensitivity_analysis()
    demo_symbolic_vs_pso_comparison()

    # Visualization last (blocks until closed)
    print("\n" + "-" * 70)
    print("Final demo: Visualization (close plot window to finish)")
    demo_visualize_results()

    print("\n" + "=" * 70)
    print("Symbolic linkage demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
