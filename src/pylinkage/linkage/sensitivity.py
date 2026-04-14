"""
Sensitivity and tolerance analysis for linkage mechanisms.

This module provides analysis tools for:

1. Sensitivity analysis:
   - Measures how each constraint dimension affects the output path
   - Identifies critical dimensions that most affect mechanism behavior

2. Tolerance analysis:
   - Monte Carlo simulation of manufacturing tolerances
   - Statistical analysis of output variation due to dimensional tolerances
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from .._compat import get_parts

if TYPE_CHECKING:
    from typing import Any as Linkage  # accepts legacy/sim Linkage and Mechanism

    from matplotlib.axes import Axes


# =============================================================================
# Constraint Naming Helpers
# =============================================================================


def _get_constraint_names(linkage: Linkage) -> tuple[str, ...]:
    """Generate human-readable names for each constraint in a linkage.

    Names are generated based on joint type and joint name:
    - Crank: "{name}_radius"
    - Revolute: "{name}_dist1", "{name}_dist2"
    - Fixed: "{name}_radius", "{name}_angle"
    - Prismatic: "{name}_radius"

    Args:
        linkage: The linkage to analyze.

    Returns:
        Tuple of constraint names matching the flat constraint order
        from get_constraints().
    """
    names: list[str] = []

    for joint in get_parts(linkage):
        # Get joint name, use class name + index if no name attribute
        joint_name = getattr(joint, "name", None)
        if joint_name is None:
            joint_name = f"{joint.__class__.__name__}_{id(joint) % 10000}"

        n_constraints = len(joint.get_constraints())

        if n_constraints == 0:
            # Ground/Static - no constraints
            pass
        elif n_constraints == 1:
            # Crank, Prismatic/RRPDyad, LinearActuator
            names.append(f"{joint_name}_radius")
        elif n_constraints == 2:
            # FixedDyad/Fixed (distance + angle) vs RRRDyad/Revolute (two distances)
            if hasattr(joint, "angle") and not hasattr(joint, "angular_velocity"):
                names.append(f"{joint_name}_radius")
                names.append(f"{joint_name}_angle")
            else:
                names.append(f"{joint_name}_dist1")
                names.append(f"{joint_name}_dist2")

    return tuple(names)


def _name_to_index(linkage: Linkage, names: tuple[str, ...]) -> dict[str, int]:
    """Create mapping from constraint names to indices.

    Args:
        linkage: The linkage being analyzed.
        names: Tuple of constraint names (from _get_constraint_names).

    Returns:
        Dictionary mapping constraint name to its index in flat constraint list.
    """
    return {name: i for i, name in enumerate(names)}


def _get_output_joint_index(linkage: Linkage, output_joint: object | int | None) -> int:
    """Resolve output joint to an index.

    Args:
        linkage: The linkage being analyzed.
        output_joint: Joint object, index, or None (auto-detect last).

    Returns:
        Index of the output joint in get_parts(linkage).

    Raises:
        ValueError: If joint not found.
    """
    if output_joint is None:
        # Default to last joint (typically the output/coupler point)
        return len(get_parts(linkage)) - 1
    if isinstance(output_joint, int):
        if output_joint < 0 or output_joint >= len(get_parts(linkage)):
            raise ValueError(f"Joint index {output_joint} out of range")
        return output_joint
    # Find joint object
    for i, joint in enumerate(get_parts(linkage)):
        if joint is output_joint:
            return i
    raise ValueError(f"Joint {output_joint} not found in linkage")


# =============================================================================
# Sensitivity Analysis
# =============================================================================


@dataclass(frozen=True)
class SensitivityAnalysis:
    """Results of sensitivity analysis for a linkage.

    Sensitivity measures how much the output path changes when each
    constraint dimension is perturbed by a small amount.

    Attributes:
        sensitivities: Mapping from constraint name to sensitivity coefficient.
            Higher values indicate the output is more sensitive to that constraint.
        baseline_path_metric: Mean path deviation at nominal constraints (should be 0).
        baseline_transmission: Mean transmission angle at nominal (degrees), or None.
        perturbed_path_metrics: Array of path deviation metrics, one per constraint.
        perturbed_transmission: Array of transmission angles per constraint, or None.
        constraint_names: Tuple of constraint names in order.
        perturbation_delta: Relative perturbation magnitude used (e.g., 0.01 for 1%).
    """

    sensitivities: dict[str, float]
    baseline_path_metric: float
    baseline_transmission: float | None
    perturbed_path_metrics: NDArray[np.float64]
    perturbed_transmission: NDArray[np.float64] | None
    constraint_names: tuple[str, ...]
    perturbation_delta: float

    @property
    def most_sensitive(self) -> str:
        """Return name of the most sensitive constraint."""
        return max(self.sensitivities, key=lambda k: abs(self.sensitivities[k]))

    @property
    def sensitivity_ranking(self) -> list[tuple[str, float]]:
        """Return constraints ranked by sensitivity (highest first)."""
        return sorted(
            self.sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

    def to_dataframe(self) -> object:
        """Export results as pandas DataFrame.

        Returns:
            DataFrame with columns: constraint, sensitivity, perturbed_metric

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pylinkage[analysis]"
            ) from e

        data = {
            "constraint": list(self.constraint_names),
            "sensitivity": [self.sensitivities[n] for n in self.constraint_names],
            "perturbed_metric": list(self.perturbed_path_metrics),
        }
        if self.perturbed_transmission is not None:
            data["perturbed_transmission"] = list(self.perturbed_transmission)

        return pd.DataFrame(data)


def _compute_path_deviation(
    path1: NDArray[np.float64],
    path2: NDArray[np.float64],
) -> float:
    """Compute mean Euclidean deviation between two paths.

    Args:
        path1: Array of shape (n_steps, 2) with (x, y) positions.
        path2: Array of shape (n_steps, 2) with (x, y) positions.

    Returns:
        Mean Euclidean distance between corresponding points.
    """
    diff = path1 - path2
    distances = np.sqrt(np.sum(diff * diff, axis=1))
    return float(np.mean(distances))


def _simulate_output_path(
    linkage: Linkage,
    output_joint_idx: int,
    iterations: int,
) -> NDArray[np.float64]:
    """Simulate linkage and extract output joint path.

    Args:
        linkage: The linkage to simulate.
        output_joint_idx: Index of the joint to track.
        iterations: Number of simulation steps.

    Returns:
        Array of shape (iterations, 2) with (x, y) positions.
    """
    path: list[tuple[float, float]] = []
    for coords in linkage.step(iterations=iterations):
        coord = coords[output_joint_idx]
        if coord[0] is not None and coord[1] is not None:
            path.append((coord[0], coord[1]))
        else:
            # Unbuildable configuration - use NaN
            path.append((float("nan"), float("nan")))
    return np.array(path, dtype=np.float64)


def analyze_sensitivity(
    linkage: Linkage,
    output_joint: object | int | None = None,
    delta: float = 0.01,
    include_transmission: bool = True,
    iterations: int | None = None,
) -> SensitivityAnalysis:
    """Analyze sensitivity of output path to each constraint dimension.

    For each constraint, this function:
    1. Perturbs the constraint by delta (relative)
    2. Simulates the linkage
    3. Measures the path deviation from nominal
    4. Computes sensitivity coefficient

    Args:
        linkage: The linkage to analyze.
        output_joint: Joint to measure, index, or None (auto-detect last joint).
        delta: Relative perturbation magnitude (e.g., 0.01 for 1%).
        include_transmission: Whether to also measure transmission angle sensitivity.
        iterations: Number of simulation steps. Defaults to one full rotation.

    Returns:
        SensitivityAnalysis with sensitivity coefficients and statistics.

    Example:
        >>> analysis = linkage.sensitivity_analysis(delta=0.01)
        >>> print(analysis.most_sensitive)
        >>> for name, sens in analysis.sensitivity_ranking:
        ...     print(f"{name}: {sens:.4f}")
    """
    from .transmission import analyze_transmission

    # Resolve output joint
    output_joint_idx = _get_output_joint_index(linkage, output_joint)

    # Get constraint names and nominal values
    constraint_names = _get_constraint_names(linkage)
    # Note: with flat=True, get_constraints returns list[float | None]
    # We cast to list[float] since we skip None values in the loop
    nominal_constraints = cast(list[float], list(linkage.get_constraints()))

    # Save initial state
    initial_coords = linkage.get_coords()

    if iterations is None:
        iterations = linkage.get_rotation_period()

    # Baseline simulation
    try:
        nominal_path = _simulate_output_path(linkage, output_joint_idx, iterations)
        linkage.set_coords(initial_coords)

        # Baseline transmission angle if requested
        baseline_transmission: float | None = None
        if include_transmission:
            try:
                trans_analysis = analyze_transmission(linkage, iterations=iterations)
                baseline_transmission = trans_analysis.mean_angle
            except ValueError:
                # No four-bar detected
                baseline_transmission = None
            linkage.set_coords(initial_coords)

        # Analyze sensitivity for each constraint
        sensitivities: dict[str, float] = {}
        perturbed_metrics: list[float] = []
        perturbed_trans: list[float | None] = []

        for i, name in enumerate(constraint_names):
            # Skip if nominal value is None or zero
            if nominal_constraints[i] is None or abs(nominal_constraints[i]) < 1e-10:
                sensitivities[name] = 0.0
                perturbed_metrics.append(0.0)
                perturbed_trans.append(baseline_transmission)
                continue

            # Perturb constraint
            perturbed = nominal_constraints.copy()
            perturbation = nominal_constraints[i] * delta
            perturbed[i] = nominal_constraints[i] + perturbation

            # Apply and simulate
            linkage.set_constraints(perturbed)
            linkage.set_coords(initial_coords)

            try:
                perturbed_path = _simulate_output_path(linkage, output_joint_idx, iterations)
                path_deviation = _compute_path_deviation(nominal_path, perturbed_path)

                # Sensitivity = (metric change) / (constraint change)
                sensitivity = path_deviation / abs(perturbation)
            except Exception:
                # Unbuildable with perturbation
                path_deviation = float("inf")
                sensitivity = float("inf")

            sensitivities[name] = sensitivity
            perturbed_metrics.append(path_deviation)

            # Transmission sensitivity
            if include_transmission:
                try:
                    linkage.set_coords(initial_coords)
                    trans = analyze_transmission(linkage, iterations=iterations)
                    perturbed_trans.append(trans.mean_angle)
                except (ValueError, Exception):
                    perturbed_trans.append(None)

            # Restore constraints
            linkage.set_constraints(nominal_constraints)
            linkage.set_coords(initial_coords)

    finally:
        # Always restore original state
        linkage.set_constraints(nominal_constraints)
        linkage.set_coords(initial_coords)

    # Build transmission array if available
    perturbed_transmission: NDArray[np.float64] | None = None
    if include_transmission and any(t is not None for t in perturbed_trans):
        perturbed_transmission = np.array(
            [t if t is not None else float("nan") for t in perturbed_trans],
            dtype=np.float64,
        )

    return SensitivityAnalysis(
        sensitivities=sensitivities,
        baseline_path_metric=0.0,  # Nominal path compared to itself
        baseline_transmission=baseline_transmission,
        perturbed_path_metrics=np.array(perturbed_metrics, dtype=np.float64),
        perturbed_transmission=perturbed_transmission,
        constraint_names=constraint_names,
        perturbation_delta=delta,
    )


# =============================================================================
# Tolerance Analysis
# =============================================================================


@dataclass(frozen=True)
class ToleranceAnalysis:
    """Results of Monte Carlo tolerance analysis.

    Tolerance analysis simulates how manufacturing variations in link
    dimensions affect the output path. Each sample represents a possible
    manufactured linkage within tolerance bounds.

    Attributes:
        nominal_path: Output path at nominal dimensions, shape (n_steps, 2).
        output_cloud: Monte Carlo results, shape (n_samples, n_steps, 2).
        tolerances: Dictionary mapping constraint names to their tolerances.
        mean_deviation: Mean path deviation from nominal across all samples.
        max_deviation: Maximum path deviation from nominal (worst case).
        std_deviation: Standard deviation of path deviations.
        position_std: Per-position standard deviation, shape (n_steps,).
    """

    nominal_path: NDArray[np.float64]
    output_cloud: NDArray[np.float64]
    tolerances: dict[str, float]
    mean_deviation: float
    max_deviation: float
    std_deviation: float
    position_std: NDArray[np.float64]

    def plot_cloud(
        self,
        ax: Axes | None = None,
        show_nominal: bool = True,
        alpha: float = 0.1,
    ) -> Axes:
        """Plot the tolerance cloud as a scatter plot.

        Args:
            ax: Matplotlib axes to plot on. Creates new figure if None.
            show_nominal: Whether to show the nominal path as a solid line.
            alpha: Transparency for sample points (0-1).

        Returns:
            Matplotlib axes with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # Plot sample clouds
        n_samples = self.output_cloud.shape[0]
        for i in range(min(n_samples, 100)):  # Limit to 100 for visibility
            sample_path = self.output_cloud[i]
            ax.scatter(
                sample_path[:, 0],
                sample_path[:, 1],
                c="blue",
                alpha=alpha,
                s=1,
            )

        # Plot nominal path
        if show_nominal:
            ax.plot(
                self.nominal_path[:, 0],
                self.nominal_path[:, 1],
                "r-",
                linewidth=2,
                label="Nominal",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Tolerance Cloud (max deviation: {self.max_deviation:.4f})")
        ax.legend()
        ax.set_aspect("equal")

        return ax

    def to_dataframe(self) -> object:
        """Export statistics as pandas DataFrame.

        Returns:
            DataFrame with tolerance statistics.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pylinkage[analysis]"
            ) from e

        data = {
            "constraint": list(self.tolerances.keys()),
            "tolerance": list(self.tolerances.values()),
        }
        return pd.DataFrame(data)


def analyze_tolerance(
    linkage: Linkage,
    tolerances: dict[str, float],
    output_joint: object | int | None = None,
    iterations: int | None = None,
    n_samples: int = 1000,
    seed: int | None = None,
) -> ToleranceAnalysis:
    """Analyze manufacturing tolerance effects via Monte Carlo simulation.

    For each sample:
    1. Randomly perturb constraints within tolerance bounds
    2. Simulate the linkage
    3. Record the output path

    Args:
        linkage: The linkage to analyze.
        tolerances: Dictionary mapping constraint names to tolerance values.
            Each constraint is perturbed by +/- tolerance uniformly.
        output_joint: Joint to measure, index, or None (auto-detect last joint).
        iterations: Number of simulation steps. Defaults to one full rotation.
        n_samples: Number of Monte Carlo samples to run.
        seed: Random seed for reproducibility.

    Returns:
        ToleranceAnalysis with statistics and the sample cloud.

    Example:
        >>> tolerances = {
        ...     "Crank_radius": 0.1,    # +/- 0.1 mm
        ...     "Revolute_dist1": 0.2,  # +/- 0.2 mm
        ... }
        >>> analysis = linkage.tolerance_analysis(tolerances, n_samples=500)
        >>> print(f"Max deviation: {analysis.max_deviation:.4f}")
        >>> analysis.plot_cloud()
    """
    # Set up random state
    rng = np.random.default_rng(seed)

    # Resolve output joint
    output_joint_idx = _get_output_joint_index(linkage, output_joint)

    # Get constraint names and nominal values
    constraint_names = _get_constraint_names(linkage)
    name_to_idx = _name_to_index(linkage, constraint_names)
    # Note: with flat=True, get_constraints returns list[float | None]
    # We cast to list[float] since constraints are always set for valid linkages
    nominal_constraints = cast(list[float], list(linkage.get_constraints()))

    # Validate tolerance names
    for name in tolerances:
        if name not in name_to_idx:
            available = ", ".join(constraint_names)
            raise ValueError(f"Unknown constraint name '{name}'. Available: {available}")

    # Save initial state
    initial_coords = linkage.get_coords()

    if iterations is None:
        iterations = linkage.get_rotation_period()

    # Nominal simulation
    try:
        nominal_path = _simulate_output_path(linkage, output_joint_idx, iterations)
        linkage.set_coords(initial_coords)

        # Monte Carlo sampling
        output_cloud: list[NDArray[np.float64]] = []
        deviations: list[float] = []

        for _ in range(n_samples):
            # Generate random perturbations
            perturbed = nominal_constraints.copy()
            for name, tol in tolerances.items():
                idx = name_to_idx[name]
                # Uniform perturbation within +/- tolerance
                perturbation = rng.uniform(-tol, tol)
                perturbed[idx] = perturbed[idx] + perturbation

            # Apply and simulate
            linkage.set_constraints(perturbed)
            linkage.set_coords(initial_coords)

            try:
                sample_path = _simulate_output_path(linkage, output_joint_idx, iterations)
                output_cloud.append(sample_path)

                # Compute deviation from nominal
                deviation = _compute_path_deviation(nominal_path, sample_path)
                deviations.append(deviation)
            except Exception:
                # Unbuildable configuration - skip
                pass

            # Restore for next iteration
            linkage.set_constraints(nominal_constraints)
            linkage.set_coords(initial_coords)

    finally:
        # Always restore original state
        linkage.set_constraints(nominal_constraints)
        linkage.set_coords(initial_coords)

    if not output_cloud:
        raise ValueError("No valid samples generated. Tolerances may be too large.")

    # Build output arrays
    output_cloud_array = np.array(output_cloud, dtype=np.float64)
    deviations_array = np.array(deviations, dtype=np.float64)

    # Compute per-position statistics
    # Standard deviation at each step across all samples
    position_std = np.std(output_cloud_array, axis=0)
    # Take norm of (std_x, std_y) for each position
    position_std_scalar = np.sqrt(np.sum(position_std * position_std, axis=1))

    return ToleranceAnalysis(
        nominal_path=nominal_path,
        output_cloud=output_cloud_array,
        tolerances=tolerances,
        mean_deviation=float(np.mean(deviations_array)),
        max_deviation=float(np.max(deviations_array)),
        std_deviation=float(np.std(deviations_array)),
        position_std=position_std_scalar,
    )
