"""Mixed-variable co-optimization of topology and dimensions.

Jointly searches discrete topology space and continuous dimensional
space using NSGA-II/III with custom genetic operators. Each chromosome
encodes a topology index (discrete) and link lengths (continuous).

This is the Phase 4 entry point for co-optimization. For warm-starting
from Phase 3 synthesis results, see :mod:`~.warm_start`.

Example::

    from pylinkage.optimization.mixed_variable import co_optimize
    from pylinkage.synthesis.ranking import compute_path_accuracy, compute_compactness

    points = [(0, 0), (1, 2), (2, 3), (3, 2), (4, 0)]

    def accuracy_obj(linkage):
        return compute_path_accuracy(linkage, points)

    def compactness_obj(linkage):
        return compute_compactness(linkage)

    result = co_optimize(
        objectives=[accuracy_obj, compactness_obj],
        precision_points=points,
        objective_names=["Path Error", "Compactness"],
    )
    print(f"Found {len(result.pareto_front)} Pareto solutions")
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .co_optimization_types import (
    CoOptimizationConfig,
    CoOptimizationResult,
    CoOptSolution,
    MixedChromosome,
)
from .topology_neighborhood import build_neighborhood_graph

if TYPE_CHECKING:
    from ..synthesis._types import PrecisionPoint
    from ..topology.catalog import CatalogEntry, TopologyCatalog

logger = logging.getLogger(__name__)


def co_optimize(
    objectives: Sequence[Callable[..., float]],
    precision_points: list[PrecisionPoint] | None = None,
    catalog: TopologyCatalog | None = None,
    config: CoOptimizationConfig | None = None,
    initial_population: list[MixedChromosome] | None = None,
    objective_names: Sequence[str] | None = None,
) -> CoOptimizationResult:
    """Co-optimize topology and dimensions across all catalog topologies.

    Each candidate is a (topology, link_lengths) pair. The optimizer
    uses NSGA-II/III with custom operators that handle the mixed
    discrete/continuous encoding.

    Objectives receive a ``Linkage`` object and should return a float
    (minimized). Use ``float('inf')`` for infeasible solutions.

    Args:
        objectives: Callables ``(linkage) -> float``, all minimized.
        precision_points: Target points (passed to objectives that need them).
        catalog: Topology catalog. If None, loads the built-in catalog.
        config: Optimizer configuration. If None, uses defaults.
        initial_population: Seed chromosomes (e.g., from warm_start).
        objective_names: Names for each objective (for plotting).

    Returns:
        CoOptimizationResult with Pareto front and solution metadata.

    Raises:
        ImportError: If pymoo is not installed.
    """
    from .multi_objective import _check_pymoo_available

    _check_pymoo_available()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    if config is None:
        config = CoOptimizationConfig()

    if catalog is None:
        from ..topology.catalog import load_catalog
        catalog = load_catalog()

    # Build topology context
    compatible = catalog.compatible_topologies(max_links=config.max_links)
    if not compatible:
        from .collections.pareto import ParetoFront
        return CoOptimizationResult(
            pareto_front=ParetoFront(solutions=[]),
            config=config,
        )

    topo_context = _TopologyContext(catalog, compatible, config)
    neighborhood = build_neighborhood_graph(catalog)

    # Build neighbor index map (topology_idx -> list of neighbor indices)
    neighbor_map: dict[int, list[int]] = {}
    for entry in compatible:
        idx = topo_context.topo_to_idx[entry.id]
        neighbors_for = neighborhood.get(entry.id, [])
        neighbor_indices = []
        for n in neighbors_for:
            if n.target_id in topo_context.topo_to_idx:
                neighbor_indices.append(topo_context.topo_to_idx[n.target_id])
        neighbor_map[idx] = neighbor_indices

    n_obj = len(objectives)
    n_dim = topo_context.n_dim

    # Define the pymoo problem
    class _CoProblem(ElementwiseProblem):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(
                n_var=n_dim,
                n_obj=n_obj,
                xl=topo_context.xl,
                xu=topo_context.xu,
            )

        def _evaluate(
            self,
            x: NDArray[np.floating[Any]],
            out: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            scores = _evaluate_chromosome(
                x, topo_context, objectives, precision_points
            )
            out["F"] = np.array(scores)

    problem = _CoProblem()

    # Build initial population
    rng = np.random.default_rng(config.seed)
    from .operators import MixedCrossover, MixedMutation, warm_start_sampling

    if initial_population:
        X0 = warm_start_sampling(
            initial_population,
            config.pop_size,
            n_dim,
            topo_context.xl,
            topo_context.xu,
            topo_context.n_topologies,
            rng,
        )
    else:
        X0 = warm_start_sampling(
            [],
            config.pop_size,
            n_dim,
            topo_context.xl,
            topo_context.xu,
            topo_context.n_topologies,
            rng,
        )

    # Custom pymoo operators
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.sampling import Sampling

    crossover_op = MixedCrossover(crossover_prob=config.crossover_prob)
    mutation_op = MixedMutation(
        topology_rate=config.topology_mutation_rate,
        dimension_sigma=config.dimension_mutation_sigma,
    )

    class _Sampling(Sampling):  # type: ignore[misc]
        def _do(self, problem: Any, n_samples: int, **kwargs: Any) -> NDArray[np.floating[Any]]:
            return X0[:n_samples]

    class _Crossover(Crossover):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(n_parents=2, n_offsprings=2)

        def _do(
            self, problem: Any, X: NDArray[np.floating[Any]], **kwargs: Any
        ) -> NDArray[np.floating[Any]]:
            n_matings = X.shape[1]
            Y = np.empty((2, n_matings, X.shape[2]))
            for i in range(n_matings):
                Y[0, i], Y[1, i] = crossover_op(
                    X[0, i], X[1, i],
                    topo_context.xl, topo_context.xu,
                    topo_context.n_topologies, rng,
                )
            return Y

    class _Mutation(Mutation):  # type: ignore[misc]
        def _do(
            self, problem: Any, X: NDArray[np.floating[Any]], **kwargs: Any
        ) -> NDArray[np.floating[Any]]:
            for i in range(len(X)):
                mutation_op(
                    X[i], topo_context.xl, topo_context.xu,
                    topo_context.n_topologies, neighbor_map, rng,
                )
            return X

    # Configure algorithm
    if config.algorithm == "nsga2":
        algo = NSGA2(
            pop_size=config.pop_size,
            sampling=_Sampling(),
            crossover=_Crossover(),
            mutation=_Mutation(),
        )
    elif config.algorithm == "nsga3":
        n_partitions = max(4, 12 - n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algo = NSGA3(
            pop_size=config.pop_size,
            ref_dirs=ref_dirs,
            sampling=_Sampling(),
            crossover=_Crossover(),
            mutation=_Mutation(),
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # Run optimization
    result = minimize(
        problem,
        algo,
        ("n_gen", config.n_generations),
        seed=config.seed,
        verbose=config.verbose,
    )

    # Extract results
    from .collections.pareto import ParetoFront, ParetoSolution

    solutions: list[CoOptSolution] = []
    pareto_solutions: list[ParetoSolution] = []

    if result.F is not None and result.X is not None:
        # Ensure 2D arrays (pymoo may return 1D for single solutions)
        res_X = np.atleast_2d(result.X)
        res_F = np.atleast_2d(result.F)

        for i in range(len(res_F)):
            x = res_X[i]
            scores = tuple(float(v) for v in res_F[i])
            chromosome = _decode_chromosome(x, topo_context)

            # Build linkage for the solution
            linkage = _build_linkage_from_chromosome(x, topo_context)
            topo_entry = topo_context.idx_to_entry.get(int(round(x[0])))

            solutions.append(CoOptSolution(
                chromosome=chromosome,
                scores=scores,
                linkage=linkage,
                topology_entry=topo_entry,
            ))
            pareto_solutions.append(ParetoSolution(
                scores=scores,
                dimensions=x[1:].copy(),
                initial_positions=(),
            ))

    if objective_names is None:
        objective_names = tuple(f"Objective {i}" for i in range(n_obj))

    pareto_front = ParetoFront(
        solutions=pareto_solutions,
        objective_names=tuple(objective_names),
    )

    return CoOptimizationResult(
        pareto_front=pareto_front,
        solutions=solutions,
        config=config,
        n_evaluations=result.algorithm.evaluator.n_eval if hasattr(result, "algorithm") else 0,
    )


class _VirtualEdge:
    """A pairwise connection between two nodes (from edge or hyperedge)."""

    __slots__ = ("id", "source", "target")

    def __init__(self, edge_id: str, source: str, target: str) -> None:
        self.id = edge_id
        self.source = source
        self.target = target


class _TopologyContext:
    """Precomputed context for evaluating chromosomes.

    Caches topology metadata, dimension bounds, and edge mappings
    so that each fitness evaluation doesn't repeat this work.

    The dimension genes encode pairwise distances between connected
    joints. For binary links (edges), this is one distance per edge.
    For ternary links (hyperedges with 3 nodes), this expands to 3
    pairwise distances (the rigid triangle).
    """

    def __init__(
        self,
        catalog: TopologyCatalog,
        compatible: list[CatalogEntry],
        config: CoOptimizationConfig,
    ) -> None:
        self.catalog = catalog
        self.compatible = compatible
        self.config = config

        # Build bidirectional index maps
        self.topo_to_idx: dict[str, int] = {}
        self.idx_to_entry: dict[int, CatalogEntry] = {}
        for i, entry in enumerate(compatible):
            self.topo_to_idx[entry.id] = i
            self.idx_to_entry[i] = entry

        self.n_topologies = len(compatible)

        # Precompute virtual edges for each topology.
        # Binary edges map 1:1. Hyperedges expand to pairwise distances.
        self._virtual_edges: dict[int, list[_VirtualEdge]] = {}
        max_vedges = 0

        for i, entry in enumerate(compatible):
            graph = entry.to_graph()
            vedges: list[_VirtualEdge] = []

            # Ground link: pairwise distances between ground nodes.
            # These are on the same rigid frame but may not have an
            # explicit edge in the graph.
            g_nodes = [n.id for n in graph.nodes.values() if n.role.name == "GROUND"]
            for a_idx in range(len(g_nodes)):
                for b_idx in range(a_idx + 1, len(g_nodes)):
                    vedges.append(_VirtualEdge(
                        f"ground_{g_nodes[a_idx]}_{g_nodes[b_idx]}",
                        g_nodes[a_idx], g_nodes[b_idx],
                    ))

            for edge in graph.edges.values():
                vedges.append(_VirtualEdge(edge.id, edge.source, edge.target))

            for he in graph.hyperedges.values():
                nodes = list(he.nodes) if isinstance(he.nodes, (list, tuple)) else list(he.nodes)
                for a_idx in range(len(nodes)):
                    for b_idx in range(a_idx + 1, len(nodes)):
                        vedge_id = f"{he.id}_{nodes[a_idx]}_{nodes[b_idx]}"
                        vedges.append(_VirtualEdge(vedge_id, nodes[a_idx], nodes[b_idx]))

            self._virtual_edges[i] = vedges
            max_vedges = max(max_vedges, len(vedges))

        self._max_vedges = max_vedges

        # Total chromosome length: 1 (topology) + max_vedges (dimensions)
        self.n_dim = 1 + max_vedges

        # Build bounds
        # Use tighter bounds that produce assemblable mechanisms.
        # Link lengths typically range from 0.5 to 10 for unit-scale problems.
        xl = np.zeros(self.n_dim)
        xu = np.zeros(self.n_dim)

        xl[0] = 0.0
        xu[0] = float(max(self.n_topologies - 1, 0))

        for j in range(1, self.n_dim):
            xl[j] = 0.5
            xu[j] = 8.0

        self.xl = xl
        self.xu = xu

    def virtual_edges(self, topo_idx: int) -> list[_VirtualEdge]:
        """Virtual edges for a given topology."""
        return self._virtual_edges.get(topo_idx, [])

    def edge_count(self, topo_idx: int) -> int:
        """Number of active dimension genes for a given topology."""
        return len(self._virtual_edges.get(topo_idx, []))


def _evaluate_chromosome(
    x: NDArray[np.floating[Any]],
    ctx: _TopologyContext,
    objectives: Sequence[Callable[..., float]],
    precision_points: list[tuple[float, float]] | None,
) -> list[float]:
    """Evaluate a single chromosome against all objectives.

    Builds a Linkage from the chromosome and calls each objective.
    Returns inf for all objectives if the linkage can't be built.
    """
    linkage = _build_linkage_from_chromosome(x, ctx)

    if linkage is None:
        return [float("inf")] * len(objectives)

    scores: list[float] = []
    for obj in objectives:
        try:
            val = obj(linkage)
            if not math.isfinite(val):
                val = float("inf")
        except Exception:
            val = float("inf")
        scores.append(val)

    return scores


def _build_linkage_from_chromosome(
    x: NDArray[np.floating[Any]],
    ctx: _TopologyContext,
) -> Any:
    """Build a Linkage from a flat chromosome array.

    Returns None if construction fails.
    """
    topo_idx = int(round(np.clip(x[0], 0, ctx.n_topologies - 1)))
    entry = ctx.idx_to_entry.get(topo_idx)
    if entry is None:
        return None

    vedges = ctx.virtual_edges(topo_idx)
    n_vedges = len(vedges)
    dimensions = x[1 : 1 + n_vedges]

    # Ensure all dimensions are positive
    if np.any(dimensions <= 0):
        return None

    try:
        from ..synthesis.conversion import _generic_nbar_to_linkage
        from ..synthesis.topology_types import NBarSolution

        # Map virtual edge IDs to lengths
        vedge_lengths: dict[str, float] = {}
        for i, ve in enumerate(vedges):
            if i < len(dimensions):
                vedge_lengths[ve.id] = float(dimensions[i])
            else:
                vedge_lengths[ve.id] = 1.0

        # Build the link_lengths dict keyed by original edge IDs
        graph = entry.to_graph()
        link_lengths: dict[str, float] = {}
        for edge in graph.edges.values():
            link_lengths[edge.id] = vedge_lengths.get(edge.id, 1.0)

        # Place joints using forward kinematics
        joint_positions = _place_joints_from_vedges(vedges, vedge_lengths, graph)
        if joint_positions is None:
            return None

        nbar = NBarSolution(
            topology_id=entry.id,
            joint_positions=joint_positions,
            link_lengths=link_lengths,
        )

        # Always use the generic converter since joint names come from
        # the catalog graph (J0_1, etc.), not the synthesis convention (A, B, C...).
        return _generic_nbar_to_linkage(nbar, iterations=360)
    except Exception:
        return None


def _place_joints_from_vedges(
    vedges: list[_VirtualEdge],
    vedge_lengths: dict[str, float],
    graph: Any,
) -> dict[str, tuple[float, float]] | None:
    """Place joints in 2D using virtual edges and their lengths.

    Ground nodes are placed along the x-axis. Driver nodes at 45deg
    from their anchor. Driven nodes by circle-circle intersection
    from two placed parents.

    Returns None if placement fails.
    """
    positions: dict[str, tuple[float, float]] = {}

    # Build adjacency from virtual edges for fast lookup
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for ve in vedges:
        dist = vedge_lengths.get(ve.id, 1.0)
        adjacency.setdefault(ve.source, []).append((ve.target, dist))
        adjacency.setdefault(ve.target, []).append((ve.source, dist))

    # Identify node roles
    ground_nodes = [n.id for n in graph.nodes.values() if n.role.name == "GROUND"]
    driver_nodes = [n.id for n in graph.nodes.values() if n.role.name == "DRIVER"]
    driven_nodes = [n.id for n in graph.nodes.values() if n.role.name == "DRIVEN"]

    # Place ground nodes along x-axis
    x_pos = 0.0
    for i, node_id in enumerate(ground_nodes):
        if i == 0:
            positions[node_id] = (0.0, 0.0)
        else:
            ground_dist = _find_dist_to_placed(node_id, positions, adjacency)
            if ground_dist is None:
                ground_dist = 4.0
            positions[node_id] = (x_pos + ground_dist, 0.0)
            x_pos += ground_dist

    # Place driver nodes (cranks)
    for node_id in driver_nodes:
        driver_dist = _find_dist_to_placed(node_id, positions, adjacency)
        if driver_dist is None:
            driver_dist = 1.0
        anchor_id = _find_placed_neighbor(node_id, positions, adjacency)
        if anchor_id is not None:
            ax, ay = positions[anchor_id]
            positions[node_id] = (ax + driver_dist * 0.707, ay + driver_dist * 0.707)
        else:
            positions[node_id] = (driver_dist * 0.707, driver_dist * 0.707)

    # Place driven nodes by circle-circle intersection (multi-pass)
    for _ in range(5):
        progress = False
        for node_id in driven_nodes:
            if node_id in positions:
                continue

            parents = _find_two_placed_parents(node_id, positions, adjacency)
            if parents is None:
                continue

            (p1_id, d1), (p2_id, d2) = parents
            x1, y1 = positions[p1_id]
            x2, y2 = positions[p2_id]

            result = _circle_circle_intersect(x1, y1, d1, x2, y2, d2)
            if result is None:
                # Try with relaxed radii (scale to just reach)
                center_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if center_dist < 1e-12:
                    continue
                scale = center_dist / (d1 + d2) * 0.99
                result = _circle_circle_intersect(
                    x1, y1, d1 / scale, x2, y2, d2 / scale
                )
                if result is None:
                    continue  # Skip this node, try on next pass

            (ix1, iy1), (ix2, iy2) = result
            positions[node_id] = (ix1, iy1) if iy1 >= iy2 else (ix2, iy2)
            progress = True

        if not progress:
            break

    all_nodes = ground_nodes + driver_nodes + driven_nodes
    unplaced = [n for n in all_nodes if n not in positions]

    if unplaced:
        # Sequential placement stalled (e.g., triad circular dependency).
        # Try simultaneous placement of remaining nodes via optimization.
        solved = _solve_remaining_simultaneously(
            unplaced, positions, adjacency
        )
        if solved is None or not all(n in solved for n in all_nodes):
            return None
        positions = solved

    return positions


def _solve_remaining_simultaneously(
    unplaced: list[str],
    positions: dict[str, tuple[float, float]],
    adjacency: dict[str, list[tuple[str, float]]],
) -> dict[str, tuple[float, float]] | None:
    """Place remaining nodes by solving distance constraints simultaneously.

    Uses scipy.optimize.least_squares to find positions that satisfy
    all pairwise distance constraints between placed and unplaced nodes,
    and among unplaced nodes themselves.

    Returns updated positions dict, or None on failure.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None

    n = len(unplaced)
    if n == 0:
        return positions

    node_to_var = {nid: i for i, nid in enumerate(unplaced)}

    # Collect distance constraints
    constraints: list[tuple[int | None, int | None, str | None, str | None, float]] = []
    # Each constraint: (var_idx_a, var_idx_b, fixed_node_a, fixed_node_b, target_dist)
    # If var_idx is None, use fixed_node position instead.

    seen_pairs: set[tuple[str, str]] = set()
    for nid in unplaced:
        for neighbor, dist in adjacency.get(nid, []):
            pair = (min(nid, neighbor), max(nid, neighbor))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if neighbor in node_to_var:
                # Both unplaced
                constraints.append((node_to_var[nid], node_to_var[neighbor], None, None, dist))
            elif neighbor in positions:
                # One placed, one unplaced
                constraints.append((node_to_var[nid], None, None, neighbor, dist))

    if not constraints:
        return None

    # Initial guess: place unplaced nodes near centroid of placed nodes
    placed_xs = [p[0] for p in positions.values()]
    placed_ys = [p[1] for p in positions.values()]
    cx = sum(placed_xs) / len(placed_xs)
    cy = sum(placed_ys) / len(placed_ys)

    x0 = np.zeros(2 * n)
    for i in range(n):
        x0[2 * i] = cx + (i - n / 2) * 0.5
        x0[2 * i + 1] = cy + 1.0

    def residuals(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        res = []
        for va, vb, _fa, fb, target in constraints:
            if va is not None:
                ax, ay = x[2 * va], x[2 * va + 1]
            else:
                return np.full(len(constraints), 1e6)

            if vb is not None:
                bx, by = x[2 * vb], x[2 * vb + 1]
            elif fb is not None and fb in positions:
                bx, by = positions[fb]
            else:
                return np.full(len(constraints), 1e6)

            dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            res.append(dist - target)
        return np.array(res)

    result = least_squares(residuals, x0, method="lm", max_nfev=200)

    if result.cost > 1e-4:
        return None

    new_positions = dict(positions)
    for i, nid in enumerate(unplaced):
        new_positions[nid] = (float(result.x[2 * i]), float(result.x[2 * i + 1]))

    return new_positions


def _find_dist_to_placed(
    node_id: str,
    positions: dict[str, tuple[float, float]],
    adjacency: dict[str, list[tuple[str, float]]],
) -> float | None:
    """Find distance from node_id to any already-placed neighbor."""
    for neighbor, dist in adjacency.get(node_id, []):
        if neighbor in positions:
            return dist
    return None


def _find_placed_neighbor(
    node_id: str,
    positions: dict[str, tuple[float, float]],
    adjacency: dict[str, list[tuple[str, float]]],
) -> str | None:
    """Find a placed node adjacent to node_id."""
    for neighbor, _ in adjacency.get(node_id, []):
        if neighbor in positions:
            return neighbor
    return None


def _find_two_placed_parents(
    node_id: str,
    positions: dict[str, tuple[float, float]],
    adjacency: dict[str, list[tuple[str, float]]],
) -> tuple[tuple[str, float], tuple[str, float]] | None:
    """Find two placed neighbors with distances."""
    parents: list[tuple[str, float]] = []
    seen: set[str] = set()
    for neighbor, dist in adjacency.get(node_id, []):
        if neighbor in positions and neighbor not in seen:
            parents.append((neighbor, dist))
            seen.add(neighbor)
            if len(parents) == 2:
                return parents[0], parents[1]
    return None


def _circle_circle_intersect(
    x1: float, y1: float, r1: float,
    x2: float, y2: float, r2: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute intersection points of two circles.

    Returns two intersection points, or None if circles don't intersect.
    """
    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx * dx + dy * dy)

    if d < 1e-12 or d > r1 + r2 or d < abs(r1 - r2):
        return None

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < 0:
        h_sq = 0.0
    h = math.sqrt(h_sq)

    mx = x1 + a * dx / d
    my = y1 + a * dy / d

    px = -dy * h / d
    py = dx * h / d

    return (mx + px, my + py), (mx - px, my - py)


def _decode_chromosome(
    x: NDArray[np.floating[Any]],
    ctx: _TopologyContext,
) -> MixedChromosome:
    """Decode a flat numpy array back into a MixedChromosome."""
    topo_idx = int(round(np.clip(x[0], 0, ctx.n_topologies - 1)))
    return MixedChromosome(
        topology_idx=topo_idx,
        dimensions=x[1:].copy(),
    )
