"""Topology catalog: a collection of known planar linkage topologies.

Provides a built-in catalog of all 1-DOF planar linkage topologies up to
8 links (19 total: 1 four-bar, 2 six-bars, 16 eight-bars), validated
against published atlases (Mruthyunjaya 1984).

The catalog is stored as a JSON file and loaded at runtime via
:func:`load_catalog`. Each entry contains the topology as a serialized
HypergraphLinkage plus metadata (link count, Assur group decomposition,
human-readable name, etc.).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ..hypergraph.graph import HypergraphLinkage
from ..hypergraph.serialization import graph_from_dict, graph_to_dict


@dataclass(frozen=True)
class CatalogEntry:
    """A single topology in the catalog.

    Attributes:
        id: Unique identifier (e.g., "four-bar", "watt", "stephenson").
        name: Human-readable name.
        num_links: Number of links including ground.
        num_joints: Number of joints.
        dof: Degree of freedom.
        link_assortment: Sorted degree sequence of links.
        assur_groups: List of Assur group signatures in solving order.
        family: Family name ("four-bar", "six-bar", "eight-bar").
    """

    id: str
    name: str
    num_links: int = 0
    num_joints: int = 0
    dof: int = 1
    link_assortment: tuple[int, ...] = ()
    assur_groups: tuple[str, ...] = ()
    family: str = ""
    _graph_data: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_graph(self) -> HypergraphLinkage:
        """Deserialize the stored graph data into a HypergraphLinkage."""
        return graph_from_dict(self._graph_data)


@dataclass
class TopologyCatalog:
    """Collection of known linkage topologies.

    Loaded from the built-in JSON catalog or constructed programmatically.
    """

    entries: dict[str, CatalogEntry] = field(default_factory=dict)

    @classmethod
    def load_builtin(cls) -> TopologyCatalog:
        """Load the built-in catalog shipped with pylinkage."""
        ref = resources.files("pylinkage.topology.data").joinpath("catalog.json")
        with resources.as_file(ref) as path:
            return cls.from_json(path)

    @classmethod
    def from_json(cls, path: str | Path) -> TopologyCatalog:
        """Load a catalog from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        catalog = cls()
        for entry_data in data["topologies"]:
            entry = CatalogEntry(
                id=entry_data["id"],
                name=entry_data["name"],
                num_links=entry_data["num_links"],
                num_joints=entry_data["num_joints"],
                dof=entry_data["dof"],
                link_assortment=tuple(entry_data["link_assortment"]),
                assur_groups=tuple(entry_data.get("assur_groups", [])),
                family=entry_data["family"],
                _graph_data=entry_data["graph"],
            )
            catalog.entries[entry.id] = entry

        return catalog

    def to_json(self, path: str | Path) -> None:
        """Save catalog to a JSON file."""
        data = {
            "version": "1.0",
            "topologies": [
                {
                    "id": e.id,
                    "name": e.name,
                    "num_links": e.num_links,
                    "num_joints": e.num_joints,
                    "dof": e.dof,
                    "link_assortment": list(e.link_assortment),
                    "assur_groups": list(e.assur_groups),
                    "family": e.family,
                    "graph": e._graph_data,
                }
                for e in self.entries.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, topology_id: str) -> CatalogEntry | None:
        """Look up a topology by ID."""
        return self.entries.get(topology_id)

    def by_num_links(self, n: int) -> list[CatalogEntry]:
        """Get all topologies with n links."""
        return [e for e in self.entries.values() if e.num_links == n]

    def by_family(self, family: str) -> list[CatalogEntry]:
        """Get all topologies in a family."""
        return [e for e in self.entries.values() if e.family == family]

    def compatible_topologies(
        self,
        *,
        max_links: int = 8,
    ) -> list[CatalogEntry]:
        """Get all topologies up to max_links."""
        return [e for e in self.entries.values() if e.num_links <= max_links]

    def all_graphs(self) -> list[HypergraphLinkage]:
        """Return all topologies as HypergraphLinkage objects."""
        return [e.to_graph() for e in self.entries.values()]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self.entries.values())

    def __contains__(self, topology_id: str) -> bool:
        return topology_id in self.entries

    def topology_index(self, topology_id: str) -> int:
        """Return the integer index of a topology (stable within a catalog instance).

        Args:
            topology_id: ID to look up.

        Returns:
            Integer index (0-based).

        Raises:
            KeyError: If topology_id is not in the catalog.
        """
        for i, entry_id in enumerate(self.entries):
            if entry_id == topology_id:
                return i
        raise KeyError(f"Topology '{topology_id}' not in catalog")

    def topology_by_index(self, index: int) -> CatalogEntry:
        """Look up a topology by integer index.

        Args:
            index: Integer index (0-based).

        Returns:
            The CatalogEntry at that index.

        Raises:
            IndexError: If index is out of range.
        """
        entries_list = list(self.entries.values())
        if index < 0 or index >= len(entries_list):
            raise IndexError(
                f"Topology index {index} out of range (catalog has {len(entries_list)} entries)"
            )
        return entries_list[index]


def load_catalog() -> TopologyCatalog:
    """Load the built-in topology catalog.

    Convenience function equivalent to ``TopologyCatalog.load_builtin()``.

    Returns:
        A TopologyCatalog with all built-in topologies.
    """
    return TopologyCatalog.load_builtin()


def generate_catalog(max_links: int = 8) -> TopologyCatalog:
    """Generate the catalog by running the enumerator.

    This is used to produce the static JSON file. Not needed at runtime.

    Args:
        max_links: Maximum number of links to enumerate (default 8).

    Returns:
        A TopologyCatalog with all enumerated topologies.
    """
    from .enumeration import enumerate_all

    all_topos = enumerate_all(max_links=max_links)
    catalog = TopologyCatalog()

    # Named topologies for known families
    names_4 = ["four-bar"]
    human_names_4 = ["Four-bar linkage"]

    names_6 = ["watt", "stephenson"]
    human_names_6 = ["Watt six-bar", "Stephenson six-bar"]

    for n_links, topos in sorted(all_topos.items()):
        if n_links == 4:
            names, human_names = names_4, human_names_4
        elif n_links == 6:
            names, human_names = names_6, human_names_6
        else:
            names = [f"eight-bar-{i + 1:02d}" for i in range(len(topos))]
            human_names = [f"Eight-bar type {i + 1}" for i in range(len(topos))]

        family = {4: "four-bar", 6: "six-bar", 8: "eight-bar"}.get(
            n_links, f"{n_links}-bar"
        )

        for i, graph in enumerate(topos):
            # Compute link assortment from the link-adjacency structure
            # Each node's degree in the HypergraphLinkage = number of edges
            # + hyperedge memberships. But we need link degrees.
            # Simpler: count from the graph structure.
            link_assortment = _compute_link_assortment(graph)

            # Try Assur decomposition
            assur_sigs = _try_decompose(graph)

            entry = CatalogEntry(
                id=names[i] if i < len(names) else f"{family}-{i + 1:02d}",
                name=human_names[i] if i < len(human_names) else f"{family} type {i + 1}",
                num_links=n_links,
                num_joints=len(graph.nodes),
                dof=1,
                link_assortment=link_assortment,
                assur_groups=assur_sigs,
                family=family,
                _graph_data=graph_to_dict(graph),
            )
            catalog.entries[entry.id] = entry

    return catalog


def _compute_link_assortment(graph: HypergraphLinkage) -> tuple[int, ...]:
    """Compute the sorted link degree sequence from a HypergraphLinkage.

    In the joint-first representation:
    - Each Edge is a binary link (degree 2)
    - Each Hyperedge with k nodes is a k-ary link (degree k)
    - Ground is a link whose degree = number of GROUND-role nodes
    """
    degrees = []

    # Ground link degree = number of joints on the ground link
    # (includes both GROUND-role and DRIVER-role nodes)
    n_ground = len(graph.ground_nodes()) + len(graph.driver_nodes())
    degrees.append(n_ground)

    # Non-ground links
    for _edge in graph.edges.values():
        degrees.append(2)  # binary link
    for he in graph.hyperedges.values():
        degrees.append(len(he.nodes))  # k-ary link

    return tuple(sorted(degrees))


def _try_decompose(graph: HypergraphLinkage) -> tuple[str, ...]:
    """Try to Assur-decompose the topology and return group signatures."""
    try:
        from ..assur.decomposition import decompose_assur_groups
        from ..assur.hypergraph_conversion import from_hypergraph

        assur_graph = from_hypergraph(graph)
        result = decompose_assur_groups(assur_graph)
        return tuple(g.joint_signature for g in result.groups)
    except Exception:
        return ()
