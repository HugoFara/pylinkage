"""Core synthesis concepts and data structures.

This module provides the fundamental data structures for mechanism synthesis:
- SynthesisProblem: Definition of a synthesis problem
- SynthesisResult: Container for synthesis solutions
- Dyad: A kinematic dyad from Burmester theory
- BurmesterCurves: Circle point and center point curves
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._types import (
    AnglePair,
    ComplexPoint,
    DyadSolution,
    FourBarSolution,
    Point2D,
    Pose,
    PrecisionPoint,
    SynthesisType,
)

if TYPE_CHECKING:
    from ..linkage import Linkage
    from ..population import Ensemble


@dataclass
class SynthesisProblem:
    """Definition of a synthesis problem.

    A synthesis problem specifies the requirements that a mechanism
    must satisfy. Depending on the synthesis type, different inputs
    are required:

    - FUNCTION: angle_pairs specifying input/output angle relationships
    - PATH: precision_points the coupler must pass through
    - MOTION: poses specifying body positions and orientations

    Attributes:
        synthesis_type: Type of synthesis (FUNCTION, PATH, or MOTION).
        precision_points: Target points for path generation.
        angle_pairs: Input/output angle pairs for function generation.
        poses: Target poses for motion generation.
        ground_pivot_a: Optional fixed position of first ground pivot.
        ground_pivot_d: Optional fixed position of second ground pivot.
        ground_length: Optional fixed ground link length.
    """

    synthesis_type: SynthesisType
    precision_points: list[PrecisionPoint] = field(default_factory=list)
    angle_pairs: list[AnglePair] = field(default_factory=list)
    poses: list[Pose] = field(default_factory=list)
    ground_pivot_a: Point2D | None = None
    ground_pivot_d: Point2D | None = None
    ground_length: float | None = None

    @property
    def num_precision_positions(self) -> int:
        """Number of precision positions specified."""
        if self.synthesis_type == SynthesisType.FUNCTION:
            return len(self.angle_pairs)
        elif self.synthesis_type == SynthesisType.PATH:
            return len(self.precision_points)
        else:  # MOTION
            return len(self.poses)


@dataclass
class SynthesisResult:
    """Result of a synthesis operation.

    Contains the synthesized linkages along with metadata about
    the synthesis process including warnings, raw mathematical
    solutions, and branch information.

    The primary way to access the solutions is via the :attr:`ensemble`
    property, which returns an :class:`~pylinkage.population.Ensemble`
    for batch simulation, ranking, filtering, and visualization.

    .. deprecated:: 0.9.0
        Iterating, indexing, and calling ``len()`` directly on a
        ``SynthesisResult`` is deprecated. Use ``.ensemble`` instead.
        These proxies will be removed in version 1.0.0.

    Attributes:
        solutions: List of valid Linkage objects.
        raw_solutions: Raw mathematical solutions before filtering.
        problem: The original synthesis problem.
        warnings: Any warnings generated during synthesis.
        branch_info: Branch selection info for each solution.
    """

    solutions: list[Linkage]
    raw_solutions: list[FourBarSolution]
    problem: SynthesisProblem
    warnings: list[str] = field(default_factory=list)
    branch_info: list[dict[str, int]] = field(default_factory=list)

    # Cached Ensemble — built lazily on first access
    _ensemble: Ensemble | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Primary API — Ensemble access
    # ------------------------------------------------------------------

    @property
    def ensemble(self) -> Ensemble:
        """Solutions as an :class:`~pylinkage.population.Ensemble`.

        Built lazily on first access and cached. The Ensemble carries
        link lengths from ``raw_solutions`` as score columns
        (``crank_length``, ``coupler_length``, ``rocker_length``,
        ``ground_length``) when available.

        Raises:
            ValueError: If no valid solutions exist.
        """
        if self._ensemble is None:
            self._ensemble = self._build_ensemble()
        return self._ensemble

    # ------------------------------------------------------------------
    # Deprecated collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of valid solutions found.

        .. deprecated:: 0.9.0
            Use ``len(result.ensemble)`` instead.
        """
        warnings.warn(
            "len(SynthesisResult) is deprecated. "
            "Use len(result.ensemble) instead. "
            "Direct len() will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self.solutions)

    def __iter__(self) -> Iterator[Linkage]:
        """Iterate over valid solutions.

        .. deprecated:: 0.9.0
            Use ``result.ensemble`` instead.
        """
        warnings.warn(
            "Iterating over SynthesisResult is deprecated. "
            "Use result.ensemble instead. "
            "Direct iteration will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter(self.solutions)

    def __getitem__(self, index: int) -> Linkage:
        """Get solution by index.

        .. deprecated:: 0.9.0
            Use ``result.ensemble[index]`` instead.
        """
        warnings.warn(
            "SynthesisResult[i] is deprecated. "
            "Use result.ensemble[i] instead. "
            "Direct indexing will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.solutions[index]

    def __bool__(self) -> bool:
        """True if any solutions were found.

        .. deprecated:: 0.9.0
            Use ``len(result.solutions) > 0`` or
            ``len(result.ensemble) > 0`` instead.
        """
        warnings.warn(
            "bool(SynthesisResult) is deprecated. "
            "Use len(result.solutions) > 0 instead. "
            "Direct bool() will be removed in version 1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self.solutions) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ensemble(self) -> Ensemble:
        """Build an Ensemble from the solutions list."""
        from ..population import Ensemble

        if not self.solutions:
            raise ValueError(
                "Cannot build Ensemble from empty SynthesisResult"
            )

        template = self.solutions[0]
        n = len(self.solutions)
        n_constraints = len(template.get_num_constraints(flat=True))
        n_joints = len(template.joints)

        dims = np.empty((n, n_constraints), dtype=np.float64)
        positions = np.empty((n, n_joints, 2), dtype=np.float64)

        for i, linkage in enumerate(self.solutions):
            constraints = linkage.get_num_constraints(flat=True)
            dims[i] = [c if c is not None else 0.0 for c in constraints]
            for j, (x, y) in enumerate(linkage.get_coords()):
                positions[i, j, 0] = x if x is not None else 0.0
                positions[i, j, 1] = y if y is not None else 0.0

        # Extract link lengths from raw solutions as scores
        scores: dict[str, NDArray[np.float64]] = {}
        if self.raw_solutions and len(self.raw_solutions) >= n:
            raw = self.raw_solutions[:n]
            scores["crank_length"] = np.array(
                [s.crank_length for s in raw], dtype=np.float64,
            )
            scores["coupler_length"] = np.array(
                [s.coupler_length for s in raw], dtype=np.float64,
            )
            scores["rocker_length"] = np.array(
                [s.rocker_length for s in raw], dtype=np.float64,
            )
            scores["ground_length"] = np.array(
                [s.ground_length for s in raw], dtype=np.float64,
            )

        return Ensemble(
            linkage=template,
            dimensions=dims,
            initial_positions=positions,
            scores=scores,
        )


@dataclass(frozen=True, slots=True)
class Dyad:
    """A kinematic dyad (two-link chain) from Burmester theory.

    In Burmester theory, a dyad connects a moving point on the coupler
    (circle point) to a fixed pivot on the frame (center point) through
    a single revolute joint.

    Two dyads combined form a four-bar linkage: the circle points
    define attachment points on the coupler, and the center points
    define the fixed ground pivots.

    Attributes:
        circle_point: Point on the moving body (complex representation).
        center_point: Fixed pivot point (complex representation).
    """

    circle_point: ComplexPoint
    center_point: ComplexPoint

    @property
    def link_length(self) -> float:
        """Length of the dyad link."""
        return abs(self.circle_point - self.center_point)

    def to_cartesian(self) -> tuple[Point2D, Point2D]:
        """Convert to Cartesian coordinate tuples.

        Returns:
            Tuple of (circle_point_xy, center_point_xy).
        """
        return (
            (self.circle_point.real, self.circle_point.imag),
            (self.center_point.real, self.center_point.imag),
        )

    def to_dyad_solution(self) -> DyadSolution:
        """Convert to DyadSolution named tuple."""
        return DyadSolution(
            circle_point=self.circle_point,
            center_point=self.center_point,
        )


@dataclass
class BurmesterCurves:
    """Circle point and center point curves from Burmester theory.

    For a set of precision positions, these curves contain all possible
    dyad attachment points. The circle points lie on the moving body,
    and the center points are the corresponding fixed pivots.

    For 3 precision positions: continuous parametric curves
    For 4 precision positions: up to 6 discrete points (Ball's points)
    For 5 precision positions: typically 0-2 discrete solutions

    The curves are parameterized so that circle_curve[i] and
    center_curve[i] form a valid dyad pair.

    Attributes:
        circle_curve: Array of circle point positions (complex).
        center_curve: Array of corresponding center point positions.
        parameter: Parameter values along the curves.
        is_discrete: True if solutions are discrete points, not curves.
    """

    circle_curve: NDArray[np.complex128]
    center_curve: NDArray[np.complex128]
    parameter: NDArray[np.float64]
    is_discrete: bool = False

    def __len__(self) -> int:
        """Number of points on the curves."""
        return len(self.circle_curve)

    def __bool__(self) -> bool:
        """True if curves contain any points."""
        return len(self.circle_curve) > 0

    def get_dyad(self, index: int) -> Dyad:
        """Get dyad at specific parameter index.

        Args:
            index: Index into the curve arrays.

        Returns:
            Dyad at the specified index.
        """
        return Dyad(
            circle_point=self.circle_curve[index],
            center_point=self.center_curve[index],
        )

    def get_all_dyads(self) -> list[Dyad]:
        """Get all dyads from the curves.

        Returns:
            List of Dyad objects, one per curve point.
        """
        return [self.get_dyad(i) for i in range(len(self))]

    def sample(self, n_samples: int) -> BurmesterCurves:
        """Resample curves to specified number of points.

        Only meaningful for continuous curves (is_discrete=False).

        Args:
            n_samples: Number of samples desired.

        Returns:
            New BurmesterCurves with resampled points.
        """
        if self.is_discrete or len(self) <= n_samples:
            return self

        indices = np.linspace(0, len(self) - 1, n_samples, dtype=int)
        return BurmesterCurves(
            circle_curve=self.circle_curve[indices],
            center_curve=self.center_curve[indices],
            parameter=self.parameter[indices],
            is_discrete=self.is_discrete,
        )

    def filter_finite(self) -> BurmesterCurves:
        """Remove points with infinite or NaN coordinates.

        Returns:
            New BurmesterCurves with only finite points.
        """
        finite_mask = np.isfinite(self.circle_curve) & np.isfinite(self.center_curve)
        return BurmesterCurves(
            circle_curve=self.circle_curve[finite_mask],
            center_curve=self.center_curve[finite_mask],
            parameter=self.parameter[finite_mask],
            is_discrete=self.is_discrete,
        )
