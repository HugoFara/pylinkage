"""Tests for pylinkage.population._population."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylinkage.population import Ensemble, Member, Population
from pylinkage.synthesis.topology_types import (
    NBarSolution,
    QualityMetrics,
    TopologySolution,
)


def _make_ensemble(linkage, dims, scores=None):
    coords = [
        [x if x is not None else 0.0, y if y is not None else 0.0]
        for (x, y) in linkage.get_coords()
    ]
    base = np.array(coords, dtype=np.float64)
    positions = np.stack([base] * len(dims))
    dims_arr = np.asarray(dims, dtype=np.float64)
    return Ensemble(
        linkage=linkage,
        dimensions=dims_arr,
        initial_positions=positions,
        scores=scores,
    )


def test_population_init_empty():
    pop = Population()
    assert len(pop) == 0
    assert pop.n_topologies == 0
    assert list(pop) == []
    assert pop.topologies == {}
    assert repr(pop) == "Population()"


def test_population_init_with_ensembles(fourbar, fourbar_variant):
    ens_a = _make_ensemble(
        fourbar,
        [[1.0, 2.5, 2.0]],
        scores={"score": np.array([1.0])},
    )
    ens_b = _make_ensemble(
        fourbar_variant,
        [[1.2, 3.0, 2.5, 0.5, 1.0], [1.3, 3.1, 2.6, 0.5, 1.0]],
        scores={"score": np.array([2.0, 3.0])},
    )
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    assert pop.n_topologies == 2
    assert len(pop) == 3
    assert pop.topologies.keys() == {"a", "b"}
    assert pop.ensemble("a") is ens_a
    r = repr(pop)
    assert "a: 1 members" in r
    assert "b: 2 members" in r


def test_population_topologies_returns_copy(fourbar):
    ens = _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])
    pop = Population(ensembles={"a": ens})
    topos = pop.topologies
    topos["new"] = ens
    # Internal state must not change
    assert pop.n_topologies == 1


def test_population_iter_yields_all_members(fourbar, fourbar_variant):
    ens_a = _make_ensemble(
        fourbar, [[1.0, 2.5, 2.0], [1.1, 2.5, 2.0]],
        scores={"score": np.array([1.0, 2.0])},
    )
    ens_b = _make_ensemble(
        fourbar_variant, [[1.2, 3.0, 2.5, 0.5, 1.0]],
        scores={"score": np.array([3.0])},
    )
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    members = list(pop)
    assert len(members) == 3
    assert all(isinstance(m, Member) for m in members)


def test_population_ensemble_keyerror(fourbar):
    pop = Population(ensembles={"a": _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])})
    with pytest.raises(KeyError):
        pop.ensemble("missing")


def test_population_add_ensemble(fourbar, fourbar_variant):
    pop = Population()
    ens = _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])
    pop.add_ensemble("a", ens)
    assert pop.n_topologies == 1
    ens2 = _make_ensemble(fourbar_variant, [[1.2, 3.0, 2.5, 0.5, 1.0]])
    pop.add_ensemble("a", ens2)  # replaces
    assert pop.n_topologies == 1
    assert pop.ensemble("a") is ens2


def test_population_simulate_all(fourbar, fourbar_variant):
    ens_a = _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])
    ens_b = _make_ensemble(fourbar_variant, [[1.2, 3.0, 2.5, 0.5, 1.0]])
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    pop.simulate_all(iterations=6)
    assert ens_a.trajectories is not None
    assert ens_b.trajectories is not None
    assert ens_a.trajectories.shape[1] == 6
    assert ens_b.trajectories.shape[1] == 6


def test_population_simulate_all_default_iterations(fourbar):
    ens = _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])
    pop = Population(ensembles={"a": ens})
    pop.simulate_all()
    assert ens.trajectories is not None
    assert ens.trajectories.shape[1] == fourbar.get_rotation_period()


def test_population_flatten_returns_list(fourbar, fourbar_variant):
    ens_a = _make_ensemble(
        fourbar, [[1.0, 2.5, 2.0]],
        scores={"score": np.array([1.0])},
    )
    ens_b = _make_ensemble(
        fourbar_variant, [[1.2, 3.0, 2.5, 0.5, 1.0]],
        scores={"score": np.array([2.0])},
    )
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    flat = pop.flatten()
    assert isinstance(flat, list)
    assert len(flat) == 2
    assert {m.scores["score"] for m in flat} == {1.0, 2.0}


def test_population_rank_across_topologies(fourbar, fourbar_variant):
    ens_a = _make_ensemble(
        fourbar,
        [[1.0, 2.5, 2.0], [1.1, 2.5, 2.0]],
        scores={"score": np.array([5.0, 3.0])},
    )
    ens_b = _make_ensemble(
        fourbar_variant,
        [[1.2, 3.0, 2.5, 0.5, 1.0]],
        scores={"score": np.array([4.0])},
    )
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    asc = pop.rank("score", ascending=True)
    assert [m.scores["score"] for m in asc] == [3.0, 4.0, 5.0]

    desc = pop.rank("score", ascending=False)
    assert [m.scores["score"] for m in desc] == [5.0, 4.0, 3.0]


def test_population_rank_key_fallback_for_single_score(fourbar):
    # Members have a single score under a different name; rank("score")
    # should fall back to the only available score.
    ens = _make_ensemble(
        fourbar,
        [[1.0, 2.5, 2.0], [1.1, 2.5, 2.0]],
        scores={"only": np.array([7.0, 2.0])},
    )
    pop = Population(ensembles={"a": ens})
    ranked = pop.rank("score", ascending=True)
    assert [m.scores["only"] for m in ranked] == [2.0, 7.0]


def test_population_rank_with_missing_key_pushes_to_end(fourbar):
    # Members have multiple scores but not the requested key → sort key is inf
    ens = _make_ensemble(
        fourbar,
        [[1.0, 2.5, 2.0], [1.1, 2.5, 2.0]],
        scores={
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 4.0]),
        },
    )
    pop = Population(ensembles={"x": ens})
    ranked = pop.rank("missing", ascending=True)
    # Both members go to the same inf; just ensure call succeeds
    assert len(ranked) == 2


def test_population_top(fourbar, fourbar_variant):
    ens_a = _make_ensemble(
        fourbar,
        [[1.0, 2.5, 2.0], [1.1, 2.5, 2.0]],
        scores={"score": np.array([5.0, 3.0])},
    )
    ens_b = _make_ensemble(
        fourbar_variant,
        [[1.2, 3.0, 2.5, 0.5, 1.0]],
        scores={"score": np.array([4.0])},
    )
    pop = Population(ensembles={"a": ens_a, "b": ens_b})
    top2 = pop.top(2, "score", ascending=False)
    assert [m.scores["score"] for m in top2] == [5.0, 4.0]


def test_population_from_ensembles_autolabels(fourbar, fourbar_variant):
    ens_a = _make_ensemble(fourbar, [[1.0, 2.5, 2.0]])
    ens_b = _make_ensemble(fourbar_variant, [[1.2, 3.0, 2.5, 0.5, 1.0]])
    pop = Population.from_ensembles([ens_a, ens_b])
    assert pop.n_topologies == 2
    assert pop.ensemble("topology_0") is ens_a
    assert pop.ensemble("topology_1") is ens_b


def test_population_from_members_groups_by_topology(fourbar, fourbar_variant):
    coords_a = [
        [x if x is not None else 0.0, y if y is not None else 0.0]
        for (x, y) in fourbar.get_coords()
    ]
    coords_b = [
        [x if x is not None else 0.0, y if y is not None else 0.0]
        for (x, y) in fourbar_variant.get_coords()
    ]
    pos_a = np.array(coords_a, dtype=np.float64)
    pos_b = np.array(coords_b, dtype=np.float64)

    m1 = Member(
        dimensions=np.array([1.0, 2.5, 2.0]),
        initial_positions=pos_a,
        scores={"score": 1.5},
    )
    m2 = Member(
        dimensions=np.array([1.05, 2.5, 2.0]),
        initial_positions=pos_a,
        scores={"score": 2.0, "extra": 9.0},
    )
    m3 = Member(
        dimensions=np.array([1.2, 3.0, 2.5]),
        initial_positions=pos_b,
        scores={"score": 3.0},
    )

    pop = Population.from_members([
        (fourbar, m1),
        (fourbar, m2),
        (fourbar_variant, m3),
    ])
    # Same fourbar instance gets grouped; variant is separate
    assert pop.n_topologies == 2
    total = sum(len(e) for e in pop.topologies.values())
    assert total == 3

    # Find the group containing 2 members
    two_member_group = next(e for e in pop.topologies.values() if len(e) == 2)
    assert set(two_member_group.scores.keys()) == {"extra", "score"}
    # Member 1 did not set "extra" → NaN
    extra = two_member_group.scores["extra"]
    assert math.isnan(extra[0]) or math.isnan(extra[1])


def test_population_from_topology_solutions(fourbar, fourbar_variant):
    def _sol(linkage, topo_id, metrics):
        nb = NBarSolution(
            topology_id=topo_id,
            joint_positions={},
            link_lengths={},
        )
        return TopologySolution(
            solution=nb,
            linkage=linkage,
            topology_entry=None,  # unused in from_topology_solutions
            metrics=metrics,
        )

    sol_a1 = _sol(
        fourbar, "four_bar",
        QualityMetrics(
            path_accuracy=0.1,
            min_transmission_angle=70.0,
            link_ratio=1.5,
            compactness=5.0,
            overall_score=0.4,
        ),
    )
    sol_a2 = _sol(
        fourbar, "four_bar",
        QualityMetrics(
            path_accuracy=0.2,
            min_transmission_angle=65.0,
            link_ratio=1.6,
            compactness=6.0,
            overall_score=0.5,
        ),
    )
    sol_b = _sol(
        fourbar_variant, "variant",
        QualityMetrics(
            path_accuracy=0.3,
            min_transmission_angle=60.0,
            link_ratio=1.7,
            compactness=7.0,
            overall_score=0.6,
        ),
    )

    pop = Population.from_topology_solutions([sol_a1, sol_a2, sol_b])
    assert pop.n_topologies == 2

    ens_a = pop.ensemble("four_bar")
    assert ens_a.n_members == 2
    # Score columns carry the QualityMetrics values
    assert set(ens_a.scores.keys()) == {
        "path_accuracy", "min_transmission_angle", "link_ratio",
        "compactness", "overall_score",
    }
    np.testing.assert_allclose(ens_a.scores["path_accuracy"], [0.1, 0.2])
    np.testing.assert_allclose(ens_a.scores["overall_score"], [0.4, 0.5])

    ens_b = pop.ensemble("variant")
    assert ens_b.n_members == 1
    np.testing.assert_allclose(ens_b.scores["compactness"], [7.0])

    # Dimensions and positions reflect each template linkage
    assert ens_a.dimensions.shape == (2, len(fourbar.get_constraints()))
    assert ens_b.dimensions.shape == (1, len(fourbar_variant.get_constraints()))
