"""Service layer for synthesis endpoints.

Calls pylinkage.synthesis functions and converts results
to frontend-compatible mechanism dicts.
"""

from __future__ import annotations

import logging
from typing import Any

from pylinkage.synthesis import (
    FourBarSolution,
    Pose,
    SynthesisResult,
    function_generation,
    grashof_check,
    motion_generation,
    path_generation,
    solution_to_linkage,
)
from pylinkage.mechanism.conversion import mechanism_from_linkage
from pylinkage.mechanism.serialization import mechanism_to_dict

from ..models.synthesis_schemas import (
    FourBarSolutionDTO,
    FunctionGenerationRequest,
    MotionGenerationRequest,
    PathGenerationRequest,
    SynthesisResponse,
)

logger = logging.getLogger(__name__)


def _solution_to_dto(sol: FourBarSolution) -> FourBarSolutionDTO:
    """Convert a FourBarSolution to a DTO."""
    gt = grashof_check(
        sol.crank_length, sol.coupler_length,
        sol.rocker_length, sol.ground_length,
    )
    return FourBarSolutionDTO(
        ground_pivot_a=list(sol.ground_pivot_a),
        ground_pivot_d=list(sol.ground_pivot_d),
        crank_pivot_b=list(sol.crank_pivot_b),
        coupler_pivot_c=list(sol.coupler_pivot_c),
        crank_length=sol.crank_length,
        coupler_length=sol.coupler_length,
        rocker_length=sol.rocker_length,
        ground_length=sol.ground_length,
        coupler_point=list(sol.coupler_point) if sol.coupler_point else None,
        grashof_type=gt.name,
    )


def _linkage_to_mechanism_dict(
    sol: FourBarSolution, index: int
) -> dict[str, Any] | None:
    """Convert a FourBarSolution to a mechanism dict for the frontend.

    Returns None if conversion fails (e.g. unbuildable geometry).
    """
    try:
        linkage = solution_to_linkage(sol, name=f"synthesis-{index}")
        mechanism = mechanism_from_linkage(linkage)
        mech_dict = mechanism_to_dict(mechanism)

        # For non-Grashof: patch the driver link to arc_driver with limits
        if sol.arc_limits is not None:
            arc_start, arc_end = sol.arc_limits
            for link in mech_dict.get("links", []):
                if link.get("type") == "driver":
                    link["type"] = "arc_driver"
                    link["arc_start"] = arc_start
                    link["arc_end"] = arc_end
                    break

        return mech_dict
    except Exception:
        logger.warning("Failed to convert solution %d to mechanism dict", index)
        return None


def _build_response(result: SynthesisResult) -> SynthesisResponse:
    """Build a SynthesisResponse from a SynthesisResult."""
    from pylinkage.synthesis.conversion import _compute_crank_limits

    solution_dtos: list[FourBarSolutionDTO] = []
    mechanism_dicts: list[dict[str, Any]] = []

    for i, raw_sol in enumerate(result.raw_solutions):
        # Compute arc limits for non-Grashof solutions
        arc_limits = _compute_crank_limits(
            raw_sol.crank_length, raw_sol.coupler_length,
            raw_sol.rocker_length, raw_sol.ground_length,
        )
        if arc_limits is not None:
            raw_sol = raw_sol._replace(arc_limits=arc_limits)

        mech_dict = _linkage_to_mechanism_dict(raw_sol, i)
        if mech_dict is not None:
            solution_dtos.append(_solution_to_dto(raw_sol))
            mechanism_dicts.append(mech_dict)

    return SynthesisResponse(
        solutions=solution_dtos,
        mechanism_dicts=mechanism_dicts,
        warnings=result.warnings,
        solution_count=len(solution_dtos),
    )


def run_path_generation(request: PathGenerationRequest) -> SynthesisResponse:
    """Run path generation synthesis."""
    points = [(p.x, p.y) for p in request.precision_points]
    result = path_generation(
        precision_points=points,
        max_solutions=request.max_solutions,
        require_grashof=request.require_grashof,
        require_crank_rocker=request.require_crank_rocker,
    )
    return _build_response(result)


def run_function_generation(
    request: FunctionGenerationRequest,
) -> SynthesisResponse:
    """Run function generation synthesis."""
    angle_pairs = [(ap.theta_in, ap.theta_out) for ap in request.angle_pairs]
    result = function_generation(
        angle_pairs=angle_pairs,
        ground_length=request.ground_length,
        require_grashof=request.require_grashof,
        require_crank_rocker=request.require_crank_rocker,
    )
    return _build_response(result)


def run_motion_generation(
    request: MotionGenerationRequest,
) -> SynthesisResponse:
    """Run motion generation synthesis."""
    poses = [Pose(p.x, p.y, p.angle) for p in request.poses]
    result = motion_generation(
        poses=poses,
        max_solutions=request.max_solutions,
        require_grashof=request.require_grashof,
        require_crank_rocker=request.require_crank_rocker,
    )
    return _build_response(result)
