"""Topology neighborhood graph for evolutionary search.

Defines adjacency between catalog topologies so that topology mutations
can navigate smoothly between related topologies (e.g., four-bar to
six-bar via dyad addition).

Used by the mixed-variable co-optimizer (Phase 4) to implement
topology-aware mutation operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..topology.catalog import TopologyCatalog


@dataclass(frozen=True)
class TopologyNeighbor:
    """A neighboring topology reachable via a single structural mutation.

    Attributes:
        target_id: Catalog topology ID of the neighbor.
        operation: Type of structural change.
        description: Human-readable explanation.
    """

    target_id: str
    operation: str  # "add_dyad" | "remove_dyad" | "swap_variant" | "restructure"
    description: str


def build_neighborhood_graph(
    catalog: TopologyCatalog,
) -> dict[str, list[TopologyNeighbor]]:
    """Precompute adjacency between all catalog topologies.

    Adjacency rules:

    - **add_dyad**: A topology with N links connects to topologies
      with N+2 links whose Assur decomposition contains the smaller
      topology's decomposition as a prefix.
    - **remove_dyad**: Inverse of add_dyad.
    - **swap_variant**: Same family, different topology (e.g., Watt
      to Stephenson — both are six-bars).
    - **restructure**: Same link count, different Assur decomposition
      (connects across eight-bar variants).

    The graph is small (19 topologies in the default catalog) so
    exhaustive pair-checking is fine.

    Args:
        catalog: Topology catalog to build adjacency for.

    Returns:
        Dict mapping topology_id to list of neighbors.
    """
    entries = list(catalog.entries.values())
    graph: dict[str, list[TopologyNeighbor]] = {e.id: [] for e in entries}

    # Group by family for swap_variant connections
    by_family: dict[str, list[str]] = {}
    for e in entries:
        by_family.setdefault(e.family, []).append(e.id)

    # Group by num_links for restructure connections
    by_links: dict[int, list[str]] = {}
    for e in entries:
        by_links.setdefault(e.num_links, []).append(e.id)

    for entry in entries:
        neighbors = graph[entry.id]

        # add_dyad: connect to topologies with +2 links
        for other in entries:
            if (
                other.num_links == entry.num_links + 2
                and _is_decomposition_extension(entry.assur_groups, other.assur_groups)
            ):
                    neighbors.append(
                        TopologyNeighbor(
                            target_id=other.id,
                            operation="add_dyad",
                            description=(
                                f"Add dyad to {entry.name} -> {other.name}"
                            ),
                        )
                    )

        # remove_dyad: connect to topologies with -2 links (inverse)
        for other in entries:
            if (
                other.num_links == entry.num_links - 2
                and _is_decomposition_extension(other.assur_groups, entry.assur_groups)
            ):
                    neighbors.append(
                        TopologyNeighbor(
                            target_id=other.id,
                            operation="remove_dyad",
                            description=(
                                f"Remove dyad from {entry.name} -> {other.name}"
                            ),
                        )
                    )

        # swap_variant: same family, different topology
        for peer_id in by_family.get(entry.family, []):
            if peer_id != entry.id:
                peer = catalog.entries[peer_id]
                neighbors.append(
                    TopologyNeighbor(
                        target_id=peer_id,
                        operation="swap_variant",
                        description=(
                            f"Swap variant: {entry.name} -> {peer.name}"
                        ),
                    )
                )

        # restructure: same num_links, different family or different ID
        for peer_id in by_links.get(entry.num_links, []):
            if peer_id != entry.id:
                # Avoid duplicate with swap_variant
                already = {n.target_id for n in neighbors}
                if peer_id not in already:
                    peer = catalog.entries[peer_id]
                    neighbors.append(
                        TopologyNeighbor(
                            target_id=peer_id,
                            operation="restructure",
                            description=(
                                f"Restructure: {entry.name} -> {peer.name}"
                            ),
                        )
                    )

    return graph


def topology_neighbors(
    topology_id: str,
    catalog: TopologyCatalog,
    neighborhood: dict[str, list[TopologyNeighbor]] | None = None,
) -> list[TopologyNeighbor]:
    """Return neighbors of a given topology.

    Args:
        topology_id: ID to look up.
        catalog: The topology catalog.
        neighborhood: Pre-built neighborhood graph (optional).
            If None, builds it on the fly.

    Returns:
        List of TopologyNeighbor reachable in one step.
    """
    if neighborhood is None:
        neighborhood = build_neighborhood_graph(catalog)
    return neighborhood.get(topology_id, [])


def topology_distance(
    id_a: str,
    id_b: str,
    catalog: TopologyCatalog,
    neighborhood: dict[str, list[TopologyNeighbor]] | None = None,
) -> int:
    """Shortest path distance between two topologies.

    Uses BFS on the neighborhood graph. Returns -1 if unreachable.

    Args:
        id_a: Source topology ID.
        id_b: Target topology ID.
        catalog: The topology catalog.
        neighborhood: Pre-built neighborhood graph (optional).

    Returns:
        Number of steps, or -1 if no path exists.
    """
    if id_a == id_b:
        return 0

    if neighborhood is None:
        neighborhood = build_neighborhood_graph(catalog)

    if id_a not in neighborhood or id_b not in neighborhood:
        return -1

    # BFS
    from collections import deque

    visited: set[str] = {id_a}
    queue: deque[tuple[str, int]] = deque([(id_a, 0)])

    while queue:
        current, dist = queue.popleft()
        for neighbor in neighborhood.get(current, []):
            if neighbor.target_id == id_b:
                return dist + 1
            if neighbor.target_id not in visited:
                visited.add(neighbor.target_id)
                queue.append((neighbor.target_id, dist + 1))

    return -1


def _is_decomposition_extension(
    prefix: tuple[str, ...],
    full: tuple[str, ...],
) -> bool:
    """Check if ``prefix`` Assur groups are a subset of ``full``.

    We check subset rather than strict prefix because decomposition
    order may vary. The larger topology should contain all groups
    from the smaller topology plus at least one additional group.

    Returns True if every group in prefix appears in full (with
    multiplicity) and full has at least one extra group.
    """
    if not prefix or not full:
        # Empty prefix: any topology can grow from the ground up
        return len(full) > len(prefix)

    if len(full) <= len(prefix):
        return False

    # Count group signatures
    remaining = list(full)
    for sig in prefix:
        if sig in remaining:
            remaining.remove(sig)
        else:
            return False

    return len(remaining) > 0
