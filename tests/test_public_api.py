"""The public surface of pylinkage, pinned name by name.

A name is public when it appears in the ``__all__`` of one of the packages
listed here. Everything else, including the modules inside those packages,
is implementation: it may keep working, but nothing promises it will.

This file is the freeze. Changing the surface means changing this file, so a
name cannot slip in or out of the public API without a visible diff:

- Adding a name to ``PUBLIC`` is an addition; document it in the changelog.
- Moving a name from ``PUBLIC`` to ``DEPRECATED`` is a deprecation. Serve it
  through ``pylinkage._deprecation`` so that reading it warns, and add it to
  the table in ``docs/source/deprecations.md``.
- Removing a name from ``DEPRECATED`` is a removal, which the policy allows no
  earlier than the next major release.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

PUBLIC: dict[str, set[str]] = {
    "pylinkage": {
        "actuators",
        "assur",
        "cam",
        "components",
        "dimensions",
        "dyads",
        "exceptions",
        "geometry",
        "hypergraph",
        "linkage",
        "mechanism",
        "optimization",
        "population",
        "simulation",
        "solver",
        "symbolic",
        "synthesis",
        "topology",
        "visualizer",
        "Ground",
        "PointTracker",
        "Crank",
        "ArcCrank",
        "LinearActuator",
        "RRRDyad",
        "RRPDyad",
        "PPDyad",
        "FixedDyad",
        "JointType",
        "NodeRole",
        "NodeId",
        "EdgeId",
        "HyperedgeId",
        "ComponentId",
        "PortId",
        "NotCompletelyDefinedError",
        "OptimizationError",
        "UnbuildableError",
        "UnderconstrainedError",
        "circle_intersect",
        "cyl_to_cart",
        "intersection",
        "norm",
        "sqr_dist",
        "Linkage",
        "Simulation",
        "bounding_box",
        "extract_trajectories",
        "extract_trajectory",
        "kinematic_default_test",
        "collections",
        "generate_bounds",
        "kinematic_maximization",
        "kinematic_minimization",
        "particle_swarm_optimization",
        "trials_and_errors_optimization",
        "Ensemble",
        "Member",
        "Population",
        "plot_kinematic_linkage",
        "plot_static_linkage",
        "show_linkage",
        "swarm_tiled_repr",
    },
    "pylinkage.actuators": {
        "ArcCrank",
        "Crank",
        "LinearActuator",
    },
    "pylinkage.assur": {
        "JointType",
        "NodeRole",
        "NodeId",
        "EdgeId",
        "AssurMechanism",
        "LinkageGraph",
        "Node",
        "Edge",
        "AssurGroup",
        "Dyad",
        "Triad",
        "DyadRRR",
        "DyadRRP",
        "DyadRPR",
        "DyadPRR",
        "AssurGroupClass",
        "AssurSignature",
        "parse_signature",
        "signature_to_hypergraph",
        "signature_to_group_class",
        "DecompositionResult",
        "decompose_assur_groups",
        "validate_decomposition",
        "graph_to_mechanism",
        "mechanism_to_graph",
        "from_hypergraph",
        "to_hypergraph",
        "graph_to_dict",
        "graph_from_dict",
        "graph_to_json",
        "graph_from_json",
    },
    "pylinkage.cam": {
        "CamProfile",
        "FunctionProfile",
        "PointArrayProfile",
        "MotionLaw",
        "HarmonicMotionLaw",
        "CycloidalMotionLaw",
        "ModifiedTrapezoidalMotionLaw",
        "PolynomialMotionLaw",
        "polynomial_345",
        "polynomial_4567",
    },
    "pylinkage.components": {
        "Component",
        "ConnectedComponent",
        "Ground",
        "PointTracker",
    },
    "pylinkage.dimensions": {
        "Dimensions",
        "DriverAngle",
    },
    "pylinkage.dyads": {
        "RRRDyad",
        "RRPDyad",
        "PPDyad",
        "FixedDyad",
        "BinaryDyad",
        "TranslatingCamFollower",
        "OscillatingCamFollower",
        "create_dyad",
        "to_mechanism",
        "get_isomer_geometry",
        "get_required_anchors",
        "get_required_constraints",
    },
    "pylinkage.exceptions": {
        "UnbuildableError",
        "UnderconstrainedError",
        "NotCompletelyDefinedError",
        "OptimizationError",
    },
    "pylinkage.geometry": {
        "circle_intersect",
        "circle_line_from_points_intersection",
        "circle_line_intersection",
        "cyl_to_cart",
        "get_nearest_point",
        "intersection",
        "norm",
        "sqr_dist",
    },
    "pylinkage.hypergraph": {
        "NodeId",
        "EdgeId",
        "HyperedgeId",
        "PortId",
        "JointType",
        "NodeRole",
        "Node",
        "Edge",
        "Hyperedge",
        "HypergraphLinkage",
        "ComponentInstance",
        "Connection",
        "HierarchicalLinkage",
        "to_mechanism",
        "from_mechanism",
        "from_sim_linkage",
        "graph_to_dict",
        "graph_from_dict",
        "graph_to_json",
        "graph_from_json",
        "hierarchical_to_dict",
        "hierarchical_from_dict",
        "hierarchical_to_json",
        "hierarchical_from_json",
    },
    "pylinkage.linkage": {
        "bounding_box",
        "extract_trajectories",
        "extract_trajectory",
        "kinematic_default_test",
        "TransmissionAngleAnalysis",
        "analyze_transmission",
        "compute_transmission_angle",
        "StrokeAnalysis",
        "analyze_stroke",
        "compute_slide_position",
        "SensitivityAnalysis",
        "ToleranceAnalysis",
        "analyze_sensitivity",
        "analyze_tolerance",
    },
    "pylinkage.mechanism": {
        "Joint",
        "RevoluteJoint",
        "PrismaticJoint",
        "GroundJoint",
        "TrackerJoint",
        "JointType",
        "Link",
        "GroundLink",
        "DriverLink",
        "ArcDriverLink",
        "LinkType",
        "Mechanism",
        "MechanismBuilder",
        "fourbar",
        "slider_crank",
        "mechanism_to_dict",
        "mechanism_from_dict",
        "mechanism_to_json",
        "mechanism_from_json",
    },
    "pylinkage.optimization": {
        "chain_optimizers",
        "co_optimize",
        "CoOptimizationConfig",
        "CoOptimizationResult",
        "CoOptSolution",
        "collections",
        "differential_evolution_optimization",
        "differential_evolution_optimization_async",
        "dual_annealing_optimization",
        "generate_bounds",
        "kinematic_maximization",
        "kinematic_minimization",
        "MixedChromosome",
        "minimize_linkage",
        "minimize_linkage_async",
        "multi_objective_optimization",
        "OptimizationProgress",
        "ParetoFront",
        "ParetoSolution",
        "particle_swarm_optimization",
        "particle_swarm_optimization_async",
        "TopologyNeighbor",
        "topology_neighbors",
        "trials_and_errors_optimization",
        "trials_and_errors_optimization_async",
        "warm_start_co_optimization",
    },
    "pylinkage.population": {
        "Ensemble",
        "Member",
        "Population",
    },
    "pylinkage.simulation": {
        "Linkage",
    },
    "pylinkage.solver": {
        "SolverData",
        "JOINT_STATIC",
        "JOINT_CRANK",
        "JOINT_REVOLUTE",
        "JOINT_FIXED",
        "JOINT_PRISMATIC",
        "MAX_PARENTS",
        "simulate",
        "simulate_with_kinematics",
        "linkage_to_solver_data",
        "solver_data_to_linkage",
        "update_solver_constraints",
        "update_solver_positions",
    },
    "pylinkage.symbolic": {
        "theta",
        "SymJoint",
        "SymStatic",
        "SymCrank",
        "SymRevolute",
        "SymbolicLinkage",
        "symbolic_circle_intersect",
        "symbolic_circle_line_intersect",
        "symbolic_cyl_to_cart",
        "symbolic_dist",
        "symbolic_sqr_dist",
        "solve_linkage_symbolically",
        "eliminate_theta",
        "compute_trajectory_numeric",
        "create_trajectory_functions",
        "check_buildability",
        "SymbolicOptimizer",
        "OptimizationResult",
        "symbolic_gradient",
        "symbolic_hessian",
        "generate_symbolic_bounds",
        "get_numeric_parameters",
        "fourbar_symbolic",
    },
    "pylinkage.synthesis": {
        "PrecisionPoint",
        "Pose",
        "SynthesisType",
        "FourBarSolution",
        "DyadSolution",
        "SynthesisProblem",
        "SynthesisResult",
        "BurmesterDyad",
        "BurmesterCurves",
        "function_generation",
        "verify_function_generation",
        "path_generation",
        "path_generation_with_timing",
        "verify_path_generation",
        "motion_generation",
        "motion_generation_3_poses",
        "NBarSolution",
        "GroupSynthesisResult",
        "QualityMetrics",
        "TopologySolution",
        "six_bar_path_generation",
        "generalized_synthesis",
        "multi_topology_synthesize",
        "nbar_solution_to_linkage",
        "compute_pole",
        "compute_all_poles",
        "compute_circle_point_curve",
        "select_compatible_dyads",
        "solution_to_linkage",
        "solutions_to_linkages",
        "linkage_to_synthesis_params",
        "fourbar_from_lengths",
        "watt_from_lengths",
        "stephenson_from_lengths",
        "crank_angle_limits",
        "grashof_check",
        "is_grashof",
        "is_crank_rocker",
        "GrashofType",
        "validate_fourbar",
    },
    "pylinkage.topology": {
        "compute_dof",
        "compute_mobility",
        "MobilityInfo",
        "are_isomorphic",
        "canonical_form",
        "canonical_hash",
        "enumerate_topologies",
        "enumerate_all",
        "TopologyCatalog",
        "CatalogEntry",
        "load_catalog",
    },
    "pylinkage.visualizer": {
        "animate_dashboard",
        "animate_parallel_coordinates",
        "dashboard_layout",
        "parallel_coordinates_plot",
        "plot_kinematic_linkage",
        "plot_static_linkage",
        "show_linkage",
        "swarm_tiled_repr",
        "animate_kinematics",
        "plot_acceleration_vectors",
        "plot_kinematics_frame",
        "plot_velocity_vectors",
        "show_kinematics",
        "animate_linkage_plotly",
        "interactive_linkage_plotly",
        "plot_linkage_plotly",
        "plot_linkage_plotly_with_velocity",
        "plot_linkage_svg",
        "plot_linkage_svg_with_velocity",
        "save_linkage_svg",
        "save_linkage_svg_with_velocity",
        "plot_linkage_dxf",
        "save_linkage_dxf",
        "build_linkage_3d",
        "save_linkage_step",
        "LinkProfile",
        "JointProfile",
    },
    "pylinkage.optimization.collections": {
        "Agent",
        "ParetoFront",
        "ParetoSolution",
    },
}

DEPRECATED: dict[str, set[str]] = {
    "pylinkage.assur": {
        "MobilityResult",
        "StructuralAnalysis",
    },
    "pylinkage.components": {
        "Dyad",
        "ConnectedDyad",
    },
    "pylinkage.dyads": {
        "Ground",
        "PointTracker",
        "Crank",
        "ArcCrank",
        "LinearActuator",
        "Linkage",
        "Component",
        "ConnectedComponent",
        "Dyad",
        "ConnectedDyad",
    },
    "pylinkage.optimization": {
        "Ensemble",
    },
    "pylinkage.synthesis": {
        "Dyad",
    },
}

FROZEN_PACKAGES = sorted(PUBLIC)


@pytest.mark.parametrize("package", FROZEN_PACKAGES)
def test_all_matches_frozen_surface(package: str) -> None:
    """``__all__`` lists exactly the public and deprecated names, nothing else."""
    module = importlib.import_module(package)
    expected = PUBLIC[package] | DEPRECATED.get(package, set())
    actual = set(module.__all__)
    assert actual == expected, (
        f"{package}.__all__ drifted from the frozen surface.\n"
        f"  unexpected: {sorted(actual - expected)}\n"
        f"  missing:    {sorted(expected - actual)}"
    )


@pytest.mark.parametrize("package", FROZEN_PACKAGES)
def test_public_names_resolve_silently(package: str) -> None:
    """Every public name is reachable and reading it warns about nothing."""
    module = importlib.import_module(package)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in PUBLIC[package]:
            getattr(module, name)


@pytest.mark.parametrize(
    ("package", "name"),
    [(package, name) for package, names in DEPRECATED.items() for name in sorted(names)],
)
def test_deprecated_names_warn(package: str, name: str) -> None:
    """Every deprecated name still resolves, and says so."""
    module = importlib.import_module(package)
    with pytest.warns(DeprecationWarning, match=f"{package}.{name} is deprecated"):
        getattr(module, name)


def test_no_underscore_names_are_public() -> None:
    """A leading underscore and a place in ``__all__`` contradict each other."""
    leaked = [
        f"{package}.{name}"
        for package, names in PUBLIC.items()
        for name in names
        if name.startswith("_")
    ]
    assert not leaked
