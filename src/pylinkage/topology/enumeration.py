"""Systematic enumeration of planar linkage topologies.

Enumerates all non-isomorphic 1-DOF planar linkage topologies up to a given
number of links, using link-adjacency graph generation and isomorphism
filtering. Validated against published atlases (Mruthyunjaya 1984).

The enumeration works at the link-adjacency level (links = vertices,
joints = edges), which is the standard representation in mechanism atlas
literature. Results are converted to the joint-first HypergraphLinkage
representation used by the rest of pylinkage.

Expected topology counts for DOF=1 (revolute only):
    - 4 links:  1  (four-bar)
    - 6 links:  2  (Watt, Stephenson)
    - 8 links: 16  (per Mruthyunjaya 1984)
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, permutations

from ..hypergraph.core import Edge, Hyperedge, Node
from ..hypergraph.graph import HypergraphLinkage
from .analysis import compute_dof


def enumerate_topologies(
    num_links: int,
    *,
    dof: int = 1,
) -> list[HypergraphLinkage]:
    """Enumerate all non-isomorphic planar linkage topologies.

    Args:
        num_links: Number of links including ground (4, 6, 8, ...).
        dof: Target degree of freedom (default 1).

    Returns:
        List of HypergraphLinkage topologies, one per isomorphism class.

    Raises:
        ValueError: If num_links doesn't yield integer joint count,
            or if num_links < 4.
    """
    if num_links < 4:
        raise ValueError(f"Need at least 4 links, got {num_links}")

    num_joints = (3 * num_links - 4 + (dof - 1) * 2) // 2
    # Verify integer: DOF = 3*(n-1) - 2*j => j = (3*(n-1) - dof) / 2
    j_exact = (3 * (num_links - 1) - dof) / 2
    if j_exact != int(j_exact):
        raise ValueError(
            f"No integer joint count for {num_links} links with DOF={dof}. "
            f"Computed j={j_exact}"
        )
    num_joints = int(j_exact)

    # Generate all non-isomorphic link-adjacency graphs
    seen: set[tuple[tuple[int, ...], ...]] = set()
    results: list[HypergraphLinkage] = []

    for deg_seq in _valid_degree_sequences(num_links, num_joints):
        for adj in _generate_adjacency_matrices(deg_seq):
            if not _is_connected(adj):
                continue
            if _is_degenerate(adj):
                # Exclude degenerate chains: any proper sub-chain with
                # DOF <= 0 means a rigid substructure. Standard mechanism
                # atlases (Mruthyunjaya 1984) exclude these.
                continue

            # Canonical form at the link-adjacency level (uncolored)
            link_canon = _link_adj_canonical(adj)
            if link_canon in seen:
                continue
            seen.add(link_canon)

            # Convert to HypergraphLinkage and verify DOF
            hg = _link_adjacency_to_hypergraph(adj)
            if compute_dof(hg) == dof:
                results.append(hg)

    return results


def enumerate_all(
    *,
    max_links: int = 8,
    dof: int = 1,
) -> dict[int, list[HypergraphLinkage]]:
    """Enumerate all topologies up to max_links.

    Args:
        max_links: Maximum number of links (default 8).
        dof: Target degree of freedom (default 1).

    Returns:
        Dict mapping num_links -> list of topologies.
    """
    result: dict[int, list[HypergraphLinkage]] = {}
    for n in range(4, max_links + 1):
        j_exact = (3 * (n - 1) - dof) / 2
        if j_exact != int(j_exact):
            continue
        result[n] = enumerate_topologies(n, dof=dof)
    return result


# ---------------------------------------------------------------------------
# Internal: degree sequence generation
# ---------------------------------------------------------------------------


def _valid_degree_sequences(
    num_links: int,
    num_joints: int,
) -> list[tuple[int, ...]]:
    """Generate all valid link-degree sequences.

    Each link must have degree >= 2 (no pendant links).
    The sum of degrees = 2 * num_joints.
    Returns sorted tuples in non-decreasing order.
    """
    target_sum = 2 * num_joints
    results: list[tuple[int, ...]] = []

    def _backtrack(
        remaining: int,
        count: int,
        min_val: int,
        current: list[int],
    ) -> None:
        if count == 0:
            if remaining == 0:
                results.append(tuple(current))
            return
        # Max degree for a simple graph with num_links nodes
        max_deg = num_links - 1
        for d in range(min_val, min(remaining - 2 * (count - 1), max_deg) + 1):
            current.append(d)
            _backtrack(remaining - d, count - 1, d, current)
            current.pop()

    _backtrack(target_sum, num_links, 2, [])
    return results


# ---------------------------------------------------------------------------
# Internal: adjacency matrix generation
# ---------------------------------------------------------------------------


def _generate_adjacency_matrices(
    deg_seq: tuple[int, ...],
) -> list[tuple[tuple[int, ...], ...]]:
    """Generate all symmetric 0-1 adjacency matrices matching a degree sequence.

    Uses recursive upper-triangle fill with pruning.
    """
    n = len(deg_seq)
    # Working matrix
    mat = [[0] * n for _ in range(n)]
    remaining = list(deg_seq)
    results: list[tuple[tuple[int, ...], ...]] = []

    def _fill(row: int, col: int) -> None:
        if row >= n:
            # Verify all degrees exhausted
            if all(r == 0 for r in remaining):
                results.append(tuple(tuple(r) for r in mat))
            return

        next_row, next_col = (row, col + 1) if col + 1 < n else (row + 1, row + 2)

        # Can we place 0 here? Check remaining degree budget
        # For row i, remaining[i] must be fillable with the cells left in row i
        can_skip = True
        # remaining[row] must be <= number of unfilled cells in row
        # remaining[col] must be <= number of unfilled cells in col

        # Try placing 1 (edge between row and col)
        if remaining[row] > 0 and remaining[col] > 0:
            mat[row][col] = 1
            mat[col][row] = 1
            remaining[row] -= 1
            remaining[col] -= 1

            if _is_feasible(mat, remaining, row, col, n):
                _fill(next_row, next_col)

            mat[row][col] = 0
            mat[col][row] = 0
            remaining[row] += 1
            remaining[col] += 1

        # Try placing 0 (no edge)
        if can_skip and _is_feasible_skip(mat, remaining, row, col, n):
            _fill(next_row, next_col)

    _fill(0, 1)
    return results


def _is_feasible(
    mat: list[list[int]],
    remaining: list[int],
    row: int,
    col: int,
    n: int,
) -> bool:
    """Check if the current partial matrix can still be completed."""
    for i in range(n):
        if remaining[i] < 0:
            return False
        # Count unfilled cells for node i (upper triangle positions not yet visited)
        unfilled = 0
        for j in range(i + 1, n):
            if (i, j) > (row, col) and mat[i][j] == 0:
                unfilled += 1
        for j in range(0, i):
            if (j, i) > (row, col) and mat[j][i] == 0:
                unfilled += 1
        if remaining[i] > unfilled:
            return False
    return True


def _is_feasible_skip(
    mat: list[list[int]],
    remaining: list[int],
    row: int,
    col: int,
    n: int,
) -> bool:
    """Check feasibility after skipping (placing 0)."""
    return _is_feasible(mat, remaining, row, col, n)


# ---------------------------------------------------------------------------
# Internal: isomorphism at link-adjacency level
# ---------------------------------------------------------------------------


def _link_adj_canonical(adj: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Canonical form of a link-adjacency matrix (uncolored graph).

    Finds the lexicographically smallest adjacency matrix over all
    degree-preserving permutations. For n <= 8, this is fast enough
    to brute-force.
    """
    n = len(adj)
    if n == 0:
        return ()

    degrees = tuple(sum(adj[i]) for i in range(n))

    # Group nodes by degree
    by_degree: dict[int, list[int]] = {}
    for i, d in enumerate(degrees):
        by_degree.setdefault(d, []).append(i)

    # Generate all permutations that preserve degree
    classes = [by_degree[d] for d in sorted(by_degree.keys())]

    best: tuple[tuple[int, ...], ...] | None = None

    def _search(cls_idx: int, perm: list[int], positions: list[list[int]]) -> None:
        nonlocal best
        if cls_idx == len(classes):
            inv = [0] * n
            for i, p in enumerate(perm):
                inv[p] = i
            candidate = tuple(
                tuple(adj[inv[a]][inv[b]] for b in range(n))
                for a in range(n)
            )
            if best is None or candidate < best:
                best = candidate
            return

        nodes = classes[cls_idx]
        target_pos = positions[cls_idx]
        for p_tuple in permutations(target_pos):
            p_list = list(p_tuple)
            for idx, node in enumerate(nodes):
                perm[node] = p_list[idx]
            _search(cls_idx + 1, perm, positions)

    # The target positions for each class are all positions that
    # correspond to that degree. But for uncolored graphs, any degree-d
    # node can go to any degree-d position. Since we want the CANONICAL
    # form, the target positions are simply the sorted positions for
    # each degree class.
    target_positions = []
    pos = 0
    for d in sorted(by_degree.keys()):
        count = len(by_degree[d])
        target_positions.append(list(range(pos, pos + count)))
        pos += count

    perm = [0] * n
    _search(0, perm, target_positions)

    return best if best is not None else ()


# ---------------------------------------------------------------------------
# Internal: connectivity check
# ---------------------------------------------------------------------------


def _is_degenerate(adj: tuple[tuple[int, ...], ...]) -> bool:
    """Check if the chain contains a rigid sub-chain (DOF <= 0).

    A kinematic chain is degenerate if any proper subset of k >= 3 links
    has j' joints between them where j' > (3k-4)/2, i.e., the sub-chain
    has DOF <= 0 when isolated. This includes triangles (k=3, j'=3) and
    more complex rigid sub-structures.

    Standard mechanism atlases (Mruthyunjaya 1984) exclude degenerate chains.
    """
    n = len(adj)
    for k in range(3, n):
        max_joints = (3 * k - 4) // 2  # max joints for DOF >= 1
        for subset in combinations(range(n), k):
            # Count joints (edges) between links in this subset
            joints = 0
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    if adj[subset[i]][subset[j]]:
                        joints += 1
            if joints > max_joints:
                return True
    return False


def _is_connected(adj: tuple[tuple[int, ...], ...]) -> bool:
    """Check if the adjacency matrix represents a connected graph."""
    n = len(adj)
    if n <= 1:
        return True

    visited = set()
    queue = deque([0])
    visited.add(0)

    while queue:
        node = queue.popleft()
        for j in range(n):
            if adj[node][j] and j not in visited:
                visited.add(j)
                queue.append(j)

    return len(visited) == n


# ---------------------------------------------------------------------------
# Internal: link-adjacency to HypergraphLinkage conversion
# ---------------------------------------------------------------------------


def _link_adjacency_to_hypergraph(
    adj: tuple[tuple[int, ...], ...],
    ground_link: int = 0,
) -> HypergraphLinkage:
    """Convert a link-adjacency matrix to a joint-node HypergraphLinkage.

    In the link-adjacency representation:
    - Rows/columns = links (rigid bodies)
    - Non-zero entries = joints connecting two links

    In the HypergraphLinkage representation:
    - Nodes = joints
    - Edges = binary links (2 joints)
    - Hyperedges = ternary+ links (3+ joints)
    - Ground link is implicit (its joints get GROUND role)

    Args:
        adj: Link-adjacency matrix (n x n, symmetric, 0-1).
        ground_link: Index of the ground link (default 0).

    Returns:
        HypergraphLinkage with joints as nodes and links as edges/hyperedges.
    """
    from ..hypergraph._types import JointType, NodeRole

    n = len(adj)
    hg = HypergraphLinkage()

    # Create a joint node for each pair of adjacent links
    joint_map: dict[tuple[int, int], str] = {}  # (link_i, link_j) -> node_id
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                node_id = f"J{i}_{j}"
                is_ground = (i == ground_link or j == ground_link)
                # Find the driver: first non-ground link adjacent to ground
                is_driver = False
                if is_ground:
                    other = j if i == ground_link else i
                    # Check if this is the first ground joint (smallest other link index)
                    ground_neighbors = [
                        k for k in range(n) if k != ground_link and adj[ground_link][k]
                    ]
                    if ground_neighbors and other == min(ground_neighbors):
                        is_driver = True

                if is_driver:
                    role = NodeRole.DRIVER
                elif is_ground:
                    role = NodeRole.GROUND
                else:
                    role = NodeRole.DRIVEN

                hg.add_node(Node(
                    id=node_id,
                    role=role,
                    joint_type=JointType.REVOLUTE,
                ))
                joint_map[(i, j)] = node_id

    # Create edges (binary links) and hyperedges (ternary+ links) for each
    # non-ground link
    for link_idx in range(n):
        if link_idx == ground_link:
            continue

        # Collect all joints on this link
        joints_on_link = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] and (i == link_idx or j == link_idx):
                    joints_on_link.append(joint_map[(i, j)])

        link_id = f"L{link_idx}"

        if len(joints_on_link) == 2:
            hg.add_edge(Edge(
                id=link_id,
                source=joints_on_link[0],
                target=joints_on_link[1],
            ))
        elif len(joints_on_link) >= 3:
            hg.add_hyperedge(Hyperedge(
                id=link_id,
                nodes=tuple(joints_on_link),
            ))
        # len < 2 shouldn't happen for valid mechanisms (degree >= 2)

    return hg
