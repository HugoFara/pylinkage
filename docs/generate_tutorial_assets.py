#!/usr/bin/env python3
"""Generate visualization assets for the tutorials.

Run with: uv run python docs/generate_tutorial_assets.py
"""

import os
import sys
import math

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pylinkage as pl

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Number of steps for smooth curves
NUM_STEPS = 100


def smooth_step(linkage, num_steps=NUM_STEPS):
    """Run simulation with enough steps for smooth curves.

    Uses a small dt to get many points over one full rotation.
    """
    n = linkage.get_rotation_period()
    dt = n / num_steps
    return list(linkage.step(iterations=num_steps, dt=dt))


def save_figure(fig, name, dpi=150):
    """Save figure to assets directory."""
    path = os.path.join(ASSETS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Getting Started Tutorial Assets
# =============================================================================

def create_fourbar_static():
    """Create a static four-bar linkage image."""
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="Crank")
    output = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0),
                         distance0=3, distance1=1, name="Output")
    linkage = pl.Linkage(joints=(crank, output), name="Four-bar")

    # Run simulation to get loci (use smooth_step for more points)
    loci = smooth_step(linkage)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot loci (paths)
    crank_path = [step[0] for step in loci]
    output_path = [step[1] for step in loci]

    ax.plot([p[0] for p in crank_path], [p[1] for p in crank_path],
            'b-', alpha=0.5, linewidth=1, label='Crank path')
    ax.plot([p[0] for p in output_path], [p[1] for p in output_path],
            'g-', alpha=0.5, linewidth=1, label='Output path')

    # Reset and plot current position
    linkage.set_num_constraints([0.31, 1, 3, 1])

    # Plot ground points
    ax.plot(0, 0, 'ks', markersize=12, label='Ground')
    ax.plot(3, 0, 'ks', markersize=12)

    # Plot joints
    crank_pos = crank.coord()
    output_pos = output.coord()

    ax.plot(crank_pos[0], crank_pos[1], 'bo', markersize=10)
    ax.plot(output_pos[0], output_pos[1], 'go', markersize=10)

    # Plot links
    ax.plot([0, crank_pos[0]], [0, crank_pos[1]], 'b-', linewidth=3)
    ax.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]], 'gray', linewidth=3)
    ax.plot([3, output_pos[0]], [0, output_pos[1]], 'g-', linewidth=3)

    # Labels
    ax.annotate('A (Ground)', (0, 0), textcoords="offset points", xytext=(10, -15))
    ax.annotate('B (Ground)', (3, 0), textcoords="offset points", xytext=(10, -15))
    ax.annotate('Crank', crank_pos, textcoords="offset points", xytext=(10, 5))
    ax.annotate('Output', output_pos, textcoords="offset points", xytext=(10, 5))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Four-bar Linkage with Joint Paths')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    save_figure(fig, 'fourbar_static.png')


def create_fourbar_positions():
    """Create multiple positions overlay."""
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="Crank")
    output = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0),
                         distance0=3, distance1=1, name="Output")
    linkage = pl.Linkage(joints=(crank, output), name="Four-bar")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ground
    ax.plot(0, 0, 'ks', markersize=15)
    ax.plot(3, 0, 'ks', markersize=15)

    # Plot multiple positions
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    for i, angle in enumerate(angles):
        linkage.set_num_constraints([angle, 1, 3, 1])
        try:
            crank.reload()
            output.reload()

            crank_pos = crank.coord()
            output_pos = output.coord()

            alpha = 0.3 if i != 0 else 1.0
            lw = 1.5 if i != 0 else 3

            # Links
            ax.plot([0, crank_pos[0]], [0, crank_pos[1]],
                    color=colors[i], linewidth=lw, alpha=alpha)
            ax.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]],
                    color='gray', linewidth=lw, alpha=alpha)
            ax.plot([3, output_pos[0]], [0, output_pos[1]],
                    color=colors[i], linewidth=lw, alpha=alpha)

            # Joints
            ax.plot(crank_pos[0], crank_pos[1], 'o', color=colors[i],
                    markersize=6 if i != 0 else 10, alpha=alpha)
            ax.plot(output_pos[0], output_pos[1], 'o', color=colors[i],
                    markersize=6 if i != 0 else 10, alpha=alpha)
        except:
            pass

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Four-bar Linkage - Multiple Positions')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'fourbar_multiposition.png')


# =============================================================================
# Synthesis Tutorial Assets
# =============================================================================

def create_synthesis_path_generation():
    """Create path generation visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Target precision points
    precision_points = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.5),
        (3.0, 0.5),
    ]

    # Plot precision points prominently
    for i, (x, y) in enumerate(precision_points):
        ax.plot(x, y, 'r*', markersize=20, zorder=10)
        ax.annotate(f'P{i+1}', (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=12, fontweight='bold')

    # Create a sample four-bar that traces a coupler curve
    # Use a known working four-bar configuration
    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1, name="Crank")
    output = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0),
                         distance0=3, distance1=1, name="Output")
    linkage = pl.Linkage(joints=(crank, output), order=(crank, output))

    # Get the coupler path (use smooth_step for more points)
    loci = smooth_step(linkage)
    coupler_path = [step[-1] for step in loci]

    # Plot coupler path
    ax.plot([p[0] for p in coupler_path], [p[1] for p in coupler_path],
            'b-', linewidth=2, alpha=0.7, label='Coupler curve')

    # Plot the linkage at initial position
    ax.plot(0, 0, 'ks', markersize=12)
    ax.plot(3, 0, 'ks', markersize=12)

    crank_pos = crank.coord()
    output_pos = output.coord()

    ax.plot([0, crank_pos[0]], [0, crank_pos[1]], 'gray', linewidth=2)
    ax.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]], 'gray', linewidth=2)
    ax.plot([3, output_pos[0]], [0, output_pos[1]], 'gray', linewidth=2)

    ax.plot(crank_pos[0], crank_pos[1], 'bo', markersize=8)
    ax.plot(output_pos[0], output_pos[1], 'go', markersize=10)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Path Generation: Synthesize Linkage for Precision Points', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    # Add text box explaining the concept
    textstr = 'Goal: Find 4-bar dimensions\nso coupler passes through\nall precision points (★)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    save_figure(fig, 'synthesis_path_generation.png')


def create_synthesis_function_generation():
    """Create function generation visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: The four-bar mechanism
    ax1 = axes[0]

    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.0, distance=1, name="Input")
    output = pl.Revolute(3, 1, joint0=crank, joint1=(4, 0),
                         distance0=2.5, distance1=2, name="Output")
    linkage = pl.Linkage(joints=(crank, output))

    # Plot ground
    ax1.plot(0, 0, 'ks', markersize=12)
    ax1.plot(4, 0, 'ks', markersize=12)

    # Plot multiple input/output angle pairs
    input_angles = [0, np.pi/6, np.pi/3, np.pi/2]
    colors = ['blue', 'green', 'orange', 'red']

    for angle, color in zip(input_angles, colors):
        linkage.set_num_constraints([angle, 1, 2.5, 2])
        try:
            crank.reload()
            output.reload()

            crank_pos = crank.coord()
            output_pos = output.coord()

            # Plot crank
            ax1.plot([0, crank_pos[0]], [0, crank_pos[1]],
                     color=color, linewidth=2, alpha=0.7)
            # Plot coupler
            ax1.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]],
                     color='gray', linewidth=1.5, alpha=0.5)
            # Plot rocker
            ax1.plot([4, output_pos[0]], [0, output_pos[1]],
                     color=color, linewidth=2, alpha=0.7, linestyle='--')

            ax1.plot(crank_pos[0], crank_pos[1], 'o', color=color, markersize=6)
            ax1.plot(output_pos[0], output_pos[1], 'o', color=color, markersize=6)
        except:
            pass

    # Add angle arcs
    theta = np.linspace(0, np.pi/2, 30)
    ax1.plot(0.3*np.cos(theta), 0.3*np.sin(theta), 'b-', linewidth=2)
    ax1.annotate('φ (input)', (0.4, 0.2), fontsize=10)

    ax1.plot(4 - 0.4*np.cos(theta), 0.4*np.sin(theta), 'r--', linewidth=2)
    ax1.annotate('ψ (output)', (3.2, 0.3), fontsize=10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Four-bar Mechanism', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 2.5)

    # Right plot: Input-output relationship
    ax2 = axes[1]

    # Simulate to get input-output relationship
    input_angles_full = np.linspace(0, 2*np.pi, 100)
    output_angles = []

    for angle in input_angles_full:
        linkage.set_num_constraints([angle, 1, 2.5, 2])
        try:
            crank.reload()
            output.reload()
            output_pos = output.coord()
            # Calculate output angle
            out_angle = np.arctan2(output_pos[1], output_pos[0] - 4)
            output_angles.append(out_angle)
        except:
            output_angles.append(np.nan)

    ax2.plot(np.degrees(input_angles_full), np.degrees(output_angles), 'b-', linewidth=2)

    # Mark the precision points
    precision_inputs = [0, 30, 60]
    precision_outputs = [0, 45, 90]
    ax2.scatter(precision_inputs, precision_outputs, c='red', s=150,
                marker='*', zorder=10, label='Precision points')

    ax2.set_xlabel('Input angle φ (degrees)', fontsize=11)
    ax2.set_ylabel('Output angle ψ (degrees)', fontsize=11)
    ax2.set_title('Function Generation: φ → ψ Mapping', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_figure(fig, 'synthesis_function_generation.png')


def create_grashof_types():
    """Create Grashof classification visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        ('Crank-Rocker', 1.0, 3.0, 3.0, 4.0, 'Shortest link = crank\n→ Crank rotates fully'),
        ('Double-Crank', 1.0, 2.0, 2.0, 1.5, 'Shortest link = ground\n→ Both links rotate'),
        ('Double-Rocker', 2.0, 1.0, 3.0, 4.0, 'Shortest link = coupler\n→ Both links oscillate'),
        ('Non-Grashof', 1.0, 4.0, 2.0, 2.0, 'S + L > P + Q\n→ Limited motion'),
    ]

    for ax, (name, L1, L2, L3, L4, desc) in zip(axes.flat, configs):
        # Create linkage
        crank = pl.Crank(0, L1, joint0=(0, 0), angle=0.3, distance=L1, name="Crank")
        output = pl.Revolute(L4, 1, joint0=crank, joint1=(L4, 0),
                             distance0=L2, distance1=L3, name="Output")
        linkage = pl.Linkage(joints=(crank, output))

        # Plot ground
        ax.plot(0, 0, 'ks', markersize=10)
        ax.plot(L4, 0, 'ks', markersize=10)

        # Try to plot multiple positions
        angles = np.linspace(0, 2*np.pi, 20)
        crank_paths_x, crank_paths_y = [], []
        output_paths_x, output_paths_y = [], []

        for angle in angles:
            linkage.set_num_constraints([angle, L1, L2, L3])
            try:
                crank.reload()
                output.reload()
                crank_paths_x.append(crank.x)
                crank_paths_y.append(crank.y)
                output_paths_x.append(output.x)
                output_paths_y.append(output.y)
            except:
                pass

        if crank_paths_x:
            ax.plot(crank_paths_x, crank_paths_y, 'b-', alpha=0.5, linewidth=1)
            ax.plot(output_paths_x, output_paths_y, 'g-', alpha=0.5, linewidth=1)

            # Plot initial position
            ax.plot([0, crank_paths_x[0]], [0, crank_paths_y[0]], 'b-', linewidth=2)
            ax.plot([crank_paths_x[0], output_paths_x[0]],
                    [crank_paths_y[0], output_paths_y[0]], 'gray', linewidth=2)
            ax.plot([L4, output_paths_x[0]], [0, output_paths_y[0]], 'g-', linewidth=2)

        ax.set_title(f'{name}\nL1={L1}, L2={L2}, L3={L3}, L4={L4}', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add description
        ax.text(0.02, 0.98, desc, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Grashof Classification of Four-bar Linkages', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'synthesis_grashof_types.png')


# =============================================================================
# Symbolic Tutorial Assets
# =============================================================================

def create_symbolic_trajectory():
    """Create symbolic trajectory visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Show trajectories for different parameter values
    ax1 = axes[0]

    ground_length = 4
    param_sets = [
        {'L1': 1.0, 'L2': 3.0, 'L3': 3.0, 'color': 'blue', 'label': 'L1=1, L2=3, L3=3'},
        {'L1': 1.5, 'L2': 2.5, 'L3': 2.5, 'color': 'green', 'label': 'L1=1.5, L2=2.5, L3=2.5'},
        {'L1': 0.8, 'L2': 3.5, 'L3': 2.0, 'color': 'red', 'label': 'L1=0.8, L2=3.5, L3=2'},
    ]

    for params in param_sets:
        crank = pl.Crank(0, params['L1'], joint0=(0, 0), angle=0.31,
                         distance=params['L1'])
        output = pl.Revolute(ground_length - 1, 1, joint0=crank, joint1=(ground_length, 0),
                             distance0=params['L2'], distance1=params['L3'])
        linkage = pl.Linkage(joints=(crank, output), order=(crank, output))

        # Get trajectory
        try:
            loci = smooth_step(linkage)
            output_path = [step[-1] for step in loci]

            ax1.plot([p[0] for p in output_path], [p[1] for p in output_path],
                     color=params['color'], linewidth=2, label=params['label'])
        except:
            # Skip unbuildable linkages
            pass

    ax1.plot(0, 0, 'ks', markersize=10)
    ax1.plot(ground_length, 0, 'ks', markersize=10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Coupler Curves for Different Parameters', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: Show symbolic expression concept
    ax2 = axes[1]
    ax2.axis('off')

    text = """
Symbolic Computation in Pylinkage

For a four-bar with symbolic parameters L₁, L₂, L₃:

    x(θ) = f(θ, L₁, L₂, L₃, L₄)
    y(θ) = g(θ, L₁, L₂, L₃, L₄)

Benefits:
• Closed-form expressions
• Analytical gradients: ∂x/∂L₁, ∂y/∂L₁, ...
• Parameter sensitivity analysis
• Faster optimization with exact derivatives

Example Output Position:
x_output(θ) = L₄ - L₃·cos(arccos((L₁²cos²θ + L₁²sin²θ
              - 2L₁L₄cosθ + L₄² + L₃² - L₂²) / (2L₃√(...))))
"""

    ax2.text(0.1, 0.9, text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax2.set_title('Symbolic Expressions', fontsize=12)

    plt.tight_layout()
    save_figure(fig, 'symbolic_trajectory.png')


def create_symbolic_optimization():
    """Create symbolic optimization visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Simulate an optimization process
    np.random.seed(42)
    iterations = 50

    # Generate fake optimization history
    best_scores = [10.0]
    for i in range(iterations - 1):
        improvement = np.random.exponential(0.3) * (1 - i/iterations)
        best_scores.append(max(0.5, best_scores[-1] - improvement))

    # Left: Convergence plot
    ax1 = axes[0]
    ax1.plot(range(iterations), best_scores, 'b-', linewidth=2)
    ax1.fill_between(range(iterations), best_scores, alpha=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optimization Convergence')
    ax1.grid(True, alpha=0.3)

    # Middle: Parameter evolution
    ax2 = axes[1]
    L1_history = 1.0 + 0.3 * np.cumsum(np.random.randn(iterations) * 0.1)
    L2_history = 3.0 + 0.5 * np.cumsum(np.random.randn(iterations) * 0.1)
    L3_history = 3.0 - 0.4 * np.cumsum(np.random.randn(iterations) * 0.1)

    ax2.plot(range(iterations), L1_history, 'r-', label='L₁', linewidth=2)
    ax2.plot(range(iterations), L2_history, 'g-', label='L₂', linewidth=2)
    ax2.plot(range(iterations), L3_history, 'b-', label='L₃', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Right: Before/After comparison
    ax3 = axes[2]

    # Initial linkage
    crank1 = pl.Crank(0, 1, joint0=(0, 0), angle=0.2, distance=1)
    output1 = pl.Revolute(4, 1, joint0=crank1, joint1=(4, 0), distance0=3, distance1=3)
    linkage1 = pl.Linkage(joints=(crank1, output1), order=(crank1, output1))
    loci1 = smooth_step(linkage1)
    path1 = [step[-1] for step in loci1]

    # Optimized linkage
    crank2 = pl.Crank(0, 1.2, joint0=(0, 0), angle=0.2, distance=1.2)
    output2 = pl.Revolute(4, 1, joint0=crank2, joint1=(4, 0), distance0=2.8, distance1=2.5)
    linkage2 = pl.Linkage(joints=(crank2, output2), order=(crank2, output2))
    loci2 = smooth_step(linkage2)
    path2 = [step[-1] for step in loci2]

    ax3.plot([p[0] for p in path1], [p[1] for p in path1],
             'r--', linewidth=2, alpha=0.7, label='Initial')
    ax3.plot([p[0] for p in path2], [p[1] for p in path2],
             'g-', linewidth=2, label='Optimized')

    ax3.plot(0, 0, 'ks', markersize=8)
    ax3.plot(4, 0, 'ks', markersize=8)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Before vs After Optimization')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Gradient-Based Symbolic Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'symbolic_optimization.png')


# =============================================================================
# Visualization Tutorial Assets
# =============================================================================

def create_visualization_comparison():
    """Create comparison of visualization backends."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1, name="A")
    output = pl.Revolute(3, 2, joint0=crank, joint1=(3, 0),
                         distance0=3, distance1=1.5, name="B")
    linkage = pl.Linkage(joints=(crank, output), order=(crank, output))

    loci = smooth_step(linkage)
    crank_path = [step[0] for step in loci]
    output_path = [step[1] for step in loci]

    crank_pos = crank.coord()
    output_pos = output.coord()

    # Matplotlib style
    ax1 = axes[0]
    ax1.plot([p[0] for p in crank_path], [p[1] for p in crank_path], 'b-', alpha=0.5)
    ax1.plot([p[0] for p in output_path], [p[1] for p in output_path], 'g-', alpha=0.5)
    ax1.plot(0, 0, 'ks', markersize=10)
    ax1.plot(3, 0, 'ks', markersize=10)
    ax1.plot([0, crank_pos[0]], [0, crank_pos[1]], 'b-', linewidth=3)
    ax1.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]], 'gray', linewidth=3)
    ax1.plot([3, output_pos[0]], [0, output_pos[1]], 'g-', linewidth=3)
    ax1.plot(crank_pos[0], crank_pos[1], 'bo', markersize=10)
    ax1.plot(output_pos[0], output_pos[1], 'go', markersize=10)
    ax1.set_title('Matplotlib\n(Animations, GIFs)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plotly style (simulated)
    ax2 = axes[1]
    ax2.fill([p[0] for p in output_path], [p[1] for p in output_path],
             alpha=0.2, color='green')
    ax2.plot([p[0] for p in crank_path], [p[1] for p in crank_path], 'b-', linewidth=2)
    ax2.plot([p[0] for p in output_path], [p[1] for p in output_path], 'g-', linewidth=2)
    ax2.plot(0, 0, 's', color='#636EFA', markersize=12)
    ax2.plot(3, 0, 's', color='#636EFA', markersize=12)
    ax2.plot([0, crank_pos[0]], [0, crank_pos[1]], color='#636EFA', linewidth=3)
    ax2.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]],
             color='#AB63FA', linewidth=3)
    ax2.plot([3, output_pos[0]], [0, output_pos[1]], color='#00CC96', linewidth=3)
    ax2.plot(crank_pos[0], crank_pos[1], 'o', color='#636EFA', markersize=12)
    ax2.plot(output_pos[0], output_pos[1], 'o', color='#00CC96', markersize=12)
    ax2.set_title('Plotly\n(Interactive HTML)', fontsize=12)
    ax2.set_aspect('equal')
    ax2.set_facecolor('#F8F9FA')

    # SVG style (simulated - clean lines)
    ax3 = axes[2]
    ax3.plot([p[0] for p in output_path], [p[1] for p in output_path],
             'k-', linewidth=1, alpha=0.5)
    ax3.plot(0, 0, 'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax3.plot(3, 0, 'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax3.plot([0, crank_pos[0]], [0, crank_pos[1]], 'k-', linewidth=2)
    ax3.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]], 'k-', linewidth=2)
    ax3.plot([3, output_pos[0]], [0, output_pos[1]], 'k-', linewidth=2)
    ax3.plot(crank_pos[0], crank_pos[1], 'ko', markersize=8,
             markerfacecolor='white', markeredgewidth=2)
    ax3.plot(output_pos[0], output_pos[1], 'ko', markersize=8,
             markerfacecolor='white', markeredgewidth=2)
    ax3.annotate('A', (crank_pos[0], crank_pos[1]), xytext=(5, 5),
                 textcoords='offset points', fontsize=10)
    ax3.annotate('B', (output_pos[0], output_pos[1]), xytext=(5, 5),
                 textcoords='offset points', fontsize=10)
    ax3.set_title('drawsvg\n(Publication SVG)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.set_facecolor('white')

    plt.suptitle('Visualization Backend Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'visualization_comparison.png')


def create_velocity_visualization():
    """Create velocity vectors visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    crank = pl.Crank(0, 1, joint0=(0, 0), angle=0.8, distance=1, name="Crank")
    output = pl.Revolute(3, 2, joint0=crank, joint1=(4, 0),
                         distance0=3, distance1=2, name="Output")
    linkage = pl.Linkage(joints=(crank, output))

    # Simulate
    crank.reload()
    output.reload()

    crank_pos = crank.coord()
    output_pos = output.coord()

    # Plot linkage
    ax.plot(0, 0, 'ks', markersize=12)
    ax.plot(4, 0, 'ks', markersize=12)
    ax.plot([0, crank_pos[0]], [0, crank_pos[1]], 'b-', linewidth=3)
    ax.plot([crank_pos[0], output_pos[0]], [crank_pos[1], output_pos[1]], 'gray', linewidth=3)
    ax.plot([4, output_pos[0]], [0, output_pos[1]], 'g-', linewidth=3)
    ax.plot(crank_pos[0], crank_pos[1], 'bo', markersize=12)
    ax.plot(output_pos[0], output_pos[1], 'go', markersize=12)

    # Simulate velocity vectors (approximate)
    omega = 10  # rad/s
    crank_vel = (-omega * crank_pos[1], omega * crank_pos[0])
    output_vel = (crank_vel[0] * 0.7, crank_vel[1] * 0.5)  # Approximate

    scale = 0.05
    ax.annotate('', xy=(crank_pos[0] + crank_vel[0]*scale,
                        crank_pos[1] + crank_vel[1]*scale),
                xytext=(crank_pos[0], crank_pos[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(output_pos[0] + output_vel[0]*scale,
                        output_pos[1] + output_vel[1]*scale),
                xytext=(output_pos[0], output_pos[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Labels
    ax.text(crank_pos[0] + crank_vel[0]*scale + 0.1,
            crank_pos[1] + crank_vel[1]*scale + 0.1,
            'v_crank', fontsize=10, color='red')
    ax.text(output_pos[0] + output_vel[0]*scale + 0.1,
            output_pos[1] + output_vel[1]*scale + 0.1,
            'v_output', fontsize=10, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linkage with Velocity Vectors (ω = 10 rad/s)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import FancyArrow
    ax.plot([], [], 'r-', linewidth=2, label='Velocity vectors')
    ax.legend()

    save_figure(fig, 'visualization_velocity.png')


# =============================================================================
# Graph Representation Tutorial Assets
# =============================================================================

def create_assur_decomposition():
    """Create Assur group decomposition visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Full linkage
    ax1 = axes[0]

    # Six-bar Stephenson mechanism (simplified visualization)
    joints = {
        'A': (0, 0),      # Ground
        'B': (5, 0),      # Ground
        'C': (6, 0),      # Ground
        'D': (1, 1),      # Crank tip
        'E': (3, 2),      # Coupler
        'F': (5.5, 1.5),  # Additional joint
    }

    links = [
        ('A', 'D', 'blue'),   # Crank
        ('D', 'E', 'gray'),   # Coupler
        ('B', 'E', 'green'),  # Rocker
        ('E', 'F', 'gray'),   # Extension
        ('C', 'F', 'orange'), # Additional rocker
    ]

    for start, end, color in links:
        ax1.plot([joints[start][0], joints[end][0]],
                 [joints[start][1], joints[end][1]],
                 color=color, linewidth=3)

    for name, (x, y) in joints.items():
        if name in ['A', 'B', 'C']:
            ax1.plot(x, y, 'ks', markersize=12)
        else:
            ax1.plot(x, y, 'ko', markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax1.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=11)

    ax1.set_title('Complete Linkage', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 7)
    ax1.set_ylim(-1, 3)

    # Middle: Driver
    ax2 = axes[1]
    ax2.plot([joints['A'][0], joints['D'][0]],
             [joints['A'][1], joints['D'][1]], 'blue', linewidth=4)
    ax2.plot(joints['A'][0], joints['A'][1], 'ks', markersize=12)
    ax2.plot(joints['D'][0], joints['D'][1], 'bo', markersize=12)
    ax2.annotate('A', joints['A'], xytext=(5, -15), textcoords='offset points', fontsize=11)
    ax2.annotate('D', joints['D'], xytext=(5, 5), textcoords='offset points', fontsize=11)

    # Add rotation arrow
    theta = np.linspace(0.3, 1.2, 20)
    ax2.plot(0.3*np.cos(theta), 0.3*np.sin(theta), 'b-', linewidth=2)
    ax2.annotate('', xy=(0.3*np.cos(1.2), 0.3*np.sin(1.2)),
                 xytext=(0.3*np.cos(1.0), 0.3*np.sin(1.0)),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax2.set_title('Driver (Crank)\nDOF = 1', fontsize=12)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-1, 2)

    # Right: Assur Groups
    ax3 = axes[2]

    # RRR Dyad 1
    ax3.plot([1, 3], [1, 2], 'purple', linewidth=3, label='Dyad 1 (RRR)')
    ax3.plot([3, 5], [2, 0], 'purple', linewidth=3)
    ax3.plot(1, 1, 'mo', markersize=10)
    ax3.plot(3, 2, 'mo', markersize=10)
    ax3.annotate('D', (1, 1), xytext=(-15, 5), textcoords='offset points', fontsize=10)
    ax3.annotate('E', (3, 2), xytext=(5, 5), textcoords='offset points', fontsize=10)

    # RRR Dyad 2
    ax3.plot([3, 5.5], [2, 1.5], 'orange', linewidth=3, linestyle='--', label='Dyad 2 (RRR)')
    ax3.plot([5.5, 6], [1.5, 0], 'orange', linewidth=3, linestyle='--')
    ax3.plot(5.5, 1.5, 'o', color='orange', markersize=10)
    ax3.annotate('F', (5.5, 1.5), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax3.plot(5, 0, 'ks', markersize=10)
    ax3.plot(6, 0, 'ks', markersize=10)
    ax3.annotate('B', (5, 0), xytext=(5, -15), textcoords='offset points', fontsize=10)
    ax3.annotate('C', (6, 0), xytext=(5, -15), textcoords='offset points', fontsize=10)

    ax3.set_title('Assur Group Decomposition\n2 × RRR Dyads (DOF = 0 each)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.set_xlim(-0.5, 7)
    ax3.set_ylim(-1, 3)

    plt.suptitle('Assur Group Decomposition of a Six-bar Linkage', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'assur_decomposition.png')


def create_hypergraph_components():
    """Create hypergraph components visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Component library
    ax1 = axes[0]
    ax1.axis('off')

    # Draw component boxes
    components = [
        ('FOURBAR', 0.1, 0.7, 'Four-bar linkage\n4 ports: input, output,\nground_a, ground_b'),
        ('DYAD', 0.1, 0.4, 'RRR Dyad\n3 ports: input_a,\ninput_b, output'),
        ('CRANK_SLIDER', 0.1, 0.1, 'Crank-slider\n3 ports: input, output,\nslider_ground'),
    ]

    for name, x, y, desc in components:
        rect = plt.Rectangle((x, y), 0.35, 0.2, fill=True,
                              facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + 0.175, y + 0.15, name, ha='center', va='center',
                 fontsize=11, fontweight='bold')
        ax1.text(x + 0.4, y + 0.1, desc, fontsize=9, va='center')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Component Library', fontsize=12)

    # Middle: Hierarchical composition
    ax2 = axes[1]
    ax2.axis('off')

    # Draw hierarchical structure
    ax2.add_patch(plt.Rectangle((0.1, 0.6), 0.8, 0.3, fill=True,
                                 facecolor='lightyellow', edgecolor='darkgoldenrod', linewidth=2))
    ax2.text(0.5, 0.85, 'HierarchicalLinkage', ha='center', fontsize=11, fontweight='bold')
    ax2.text(0.5, 0.72, '"Walking Robot Leg"', ha='center', fontsize=10, style='italic')

    # Sub-components
    ax2.add_patch(plt.Rectangle((0.15, 0.2), 0.3, 0.25, fill=True,
                                 facecolor='lightblue', edgecolor='navy', linewidth=2))
    ax2.text(0.3, 0.35, 'FOURBAR\ninstance_1', ha='center', fontsize=9)

    ax2.add_patch(plt.Rectangle((0.55, 0.2), 0.3, 0.25, fill=True,
                                 facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    ax2.text(0.7, 0.35, 'DYAD\ninstance_2', ha='center', fontsize=9)

    # Connection
    ax2.annotate('', xy=(0.55, 0.33), xytext=(0.45, 0.33),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(0.5, 0.38, 'Connection', ha='center', fontsize=8, color='red')

    # Arrows from parent
    ax2.annotate('', xy=(0.3, 0.45), xytext=(0.35, 0.6),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax2.annotate('', xy=(0.7, 0.45), xytext=(0.65, 0.6),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Hierarchical Composition', fontsize=12)

    # Right: Flattened result
    ax3 = axes[2]

    # Draw a combined mechanism
    joints = {
        'G1': (0, 0), 'G2': (4, 0), 'G3': (6, 0),
        'A': (1, 1), 'B': (3, 2), 'C': (5, 1.5)
    }

    for name, (x, y) in joints.items():
        if name.startswith('G'):
            ax3.plot(x, y, 'ks', markersize=10)
        else:
            ax3.plot(x, y, 'bo', markersize=8)
        ax3.annotate(name, (x, y), xytext=(3, 3), textcoords='offset points', fontsize=9)

    # Links
    links = [('G1', 'A'), ('A', 'B'), ('G2', 'B'), ('B', 'C'), ('G3', 'C')]
    for start, end in links:
        ax3.plot([joints[start][0], joints[end][0]],
                 [joints[start][1], joints[end][1]], 'gray', linewidth=2)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Flattened Linkage\n(Ready for simulation)', fontsize=12)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Hypergraph: Component-Based Linkage Definition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'hypergraph_components.png')


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all tutorial assets."""
    print("Generating tutorial assets...")
    print("=" * 50)

    # Getting Started
    print("\n[Getting Started]")
    create_fourbar_static()
    create_fourbar_positions()

    # Synthesis
    print("\n[Synthesis]")
    create_synthesis_path_generation()
    create_synthesis_function_generation()
    create_grashof_types()

    # Symbolic
    print("\n[Symbolic]")
    create_symbolic_trajectory()
    create_symbolic_optimization()

    # Visualization
    print("\n[Visualization]")
    create_visualization_comparison()
    create_velocity_visualization()

    # Graph Representation
    print("\n[Graph Representation]")
    create_assur_decomposition()
    create_hypergraph_components()

    print("\n" + "=" * 50)
    print("All assets generated successfully!")
    print(f"Output directory: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
