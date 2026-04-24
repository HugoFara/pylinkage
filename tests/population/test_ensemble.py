"""Tests for pylinkage.population._ensemble."""

from __future__ import annotations

import numpy as np
import pytest

from pylinkage.optimization.collections.agent import Agent
from pylinkage.optimization.collections.pareto import ParetoFront, ParetoSolution
from pylinkage.population import Ensemble, Member


def _make_ensemble(fourbar, dims, positions, scores=None):
    return Ensemble(
        linkage=fourbar,
        dimensions=dims,
        initial_positions=positions,
        scores=scores,
    )


def test_ensemble_properties(fourbar, member_dims, member_positions, member_scores):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    assert ens.n_members == 3
    assert ens.n_constraints == 3
    assert ens.n_joints == 4
    assert ens.dimensions.shape == (3, 3)
    assert ens.initial_positions.shape == (3, 4, 2)
    assert set(ens.scores.keys()) == {"score", "extra"}
    assert ens.trajectories is None
    assert ens.linkage is fourbar


def test_ensemble_len_and_repr(fourbar, member_dims, member_positions, member_scores):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    assert len(ens) == 3
    r = repr(ens)
    assert "n_members=3" in r
    assert "n_joints=4" in r
    assert "score" in r
    assert "extra" in r


def test_ensemble_repr_with_no_scores(fourbar, member_dims, member_positions):
    ens = _make_ensemble(fourbar, member_dims, member_positions)
    r = repr(ens)
    assert "none" in r


def test_ensemble_topology_key_is_hashable(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    key = ens.topology_key
    assert isinstance(key, tuple)
    # Hashable
    {key: 1}


def test_ensemble_indexing_returns_member(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    m = ens[0]
    assert isinstance(m, Member)
    assert m.scores == {"score": 3.0, "extra": 0.5}
    np.testing.assert_allclose(m.dimensions, member_dims[0])

    m_np = ens[np.int64(2)]
    assert m_np.scores["score"] == 2.0


def test_ensemble_slice_returns_new_ensemble(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    sub = ens[:2]
    assert isinstance(sub, Ensemble)
    assert sub.n_members == 2
    np.testing.assert_allclose(sub.scores["score"], [3.0, 1.0])


def test_ensemble_iter_yields_members(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    members = list(ens)
    assert len(members) == 3
    assert all(isinstance(m, Member) for m in members)


def test_ensemble_simulate_stores_trajectories(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    traj = ens.simulate(iterations=10, dt=1.0)
    assert traj.shape == (3, 10, 4, 2)
    assert ens.trajectories is not None
    np.testing.assert_allclose(ens.trajectories, traj)
    # Members now return trajectories
    m = ens[0]
    assert m.trajectory is not None
    assert m.trajectory.shape == (10, 4, 2)


def test_ensemble_simulate_default_iterations(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    traj = ens.simulate()
    expected = fourbar.get_rotation_period()
    assert traj.shape[1] == expected


def test_ensemble_simulate_no_store(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    traj = ens.simulate(iterations=5, store=False)
    assert traj.shape == (3, 5, 4, 2)
    assert ens.trajectories is None


def test_ensemble_simulate_member_one_off(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    traj = ens.simulate_member(1, iterations=7)
    assert traj.shape == (7, 4, 2)
    # No batch run, so trajectories should still be None
    assert ens.trajectories is None


def test_ensemble_simulate_member_default_iterations(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    traj = ens.simulate_member(0)
    assert traj.shape[0] == fourbar.get_rotation_period()


def test_ensemble_simulate_member_uses_cache_when_available(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=6)
    cached = ens.simulate_member(2)
    np.testing.assert_allclose(cached, ens.trajectories[2])


def test_ensemble_filter_custom_predicate(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    sub = ens.filter(lambda m: m.scores["score"] > 1.5)
    assert isinstance(sub, Ensemble)
    assert sub.n_members == 2
    assert set(sub.scores["score"]) == {3.0, 2.0}


def test_ensemble_filter_by_score_range(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    sub = ens.filter_by_score("score", min_val=1.5, max_val=2.5)
    assert sub.n_members == 1
    np.testing.assert_allclose(sub.scores["score"], [2.0])


def test_ensemble_rank_and_top(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    asc = ens.rank("score", ascending=True)
    np.testing.assert_allclose(asc.scores["score"], [1.0, 2.0, 3.0])

    desc = ens.rank("score", ascending=False)
    np.testing.assert_allclose(desc.scores["score"], [3.0, 2.0, 1.0])

    top2 = ens.top(2, "score", ascending=False)
    assert top2.n_members == 2
    np.testing.assert_allclose(top2.scores["score"], [3.0, 2.0])


def test_ensemble_resolve_score_single_fallback(fourbar, member_dims, member_positions):
    scores = {"only": np.array([5.0, 3.0, 4.0])}
    ens = _make_ensemble(fourbar, member_dims, member_positions, scores)
    # Request by the default name "score" — should fall back to the only one
    ranked = ens.rank("score")
    np.testing.assert_allclose(ranked.scores["only"], [3.0, 4.0, 5.0])


def test_ensemble_resolve_score_raises_when_ambiguous(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    with pytest.raises(KeyError, match="No score named"):
        ens.rank("missing")


def test_ensemble_slice_propagates_trajectories(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=5)
    sub = ens[:2]
    assert sub.trajectories is not None
    assert sub.trajectories.shape == (2, 5, 4, 2)


def test_ensemble_from_agents(fourbar):
    coords = fourbar.get_coords()
    agents = [
        Agent(score=1.0, dimensions=[1.0, 2.5, 2.0], initial_positions=coords),
        Agent(
            score=2.0,
            dimensions=[1.1, 2.5, 2.0],
            initial_positions=[(None, None)] * len(coords),
        ),
    ]
    ens = Ensemble.from_agents(fourbar, agents)
    assert ens.n_members == 2
    np.testing.assert_allclose(ens.scores["score"], [1.0, 2.0])
    # Second agent had all-None positions → zeros
    np.testing.assert_allclose(ens.initial_positions[1], np.zeros((4, 2)))


def test_ensemble_from_agents_truncates_extra_joints(fourbar):
    # Agent provides more joints than the linkage — must not overflow
    coords = [(i * 1.0, i * 1.0) for i in range(10)]
    agents = [
        Agent(score=1.0, dimensions=[1.0, 2.5, 2.0], initial_positions=coords),
    ]
    ens = Ensemble.from_agents(fourbar, agents)
    assert ens.initial_positions.shape == (1, 4, 2)
    np.testing.assert_allclose(ens.initial_positions[0, 3], [3.0, 3.0])


def test_ensemble_from_pareto_front_with_named_objectives(fourbar):
    coords = fourbar.get_coords()
    front = ParetoFront(
        solutions=[
            ParetoSolution(
                scores=(1.0, 2.0),
                dimensions=np.array([1.0, 2.5, 2.0]),
                initial_positions=coords,
            ),
            ParetoSolution(
                scores=(0.5, 3.0),
                dimensions=np.array([1.2, 2.4, 2.1]),
                initial_positions=[(None, None)] * len(coords),
            ),
        ],
        objective_names=("acc", "smooth"),
    )
    ens = Ensemble.from_pareto_front(fourbar, front)
    assert ens.n_members == 2
    assert set(ens.scores.keys()) == {"acc", "smooth"}
    np.testing.assert_allclose(ens.scores["acc"], [1.0, 0.5])
    np.testing.assert_allclose(ens.scores["smooth"], [2.0, 3.0])


def test_ensemble_from_pareto_front_truncates_extra_joints(fourbar):
    extra_coords = [(i * 1.0, i * 1.0) for i in range(10)]
    front = ParetoFront(
        solutions=[
            ParetoSolution(
                scores=(1.0,),
                dimensions=np.array([1.0, 2.5, 2.0]),
                initial_positions=extra_coords,
            ),
        ],
    )
    ens = Ensemble.from_pareto_front(fourbar, front)
    assert ens.initial_positions.shape == (1, 4, 2)
    np.testing.assert_allclose(ens.initial_positions[0, 3], [3.0, 3.0])


def test_ensemble_from_pareto_front_auto_names_and_empty_positions(fourbar):
    front = ParetoFront(
        solutions=[
            ParetoSolution(
                scores=(1.0, 2.0, 3.0),
                dimensions=np.array([1.0, 2.5, 2.0]),
                initial_positions=[],
            ),
        ],
    )
    ens = Ensemble.from_pareto_front(fourbar, front)
    assert set(ens.scores.keys()) == {"obj_0", "obj_1", "obj_2"}
    np.testing.assert_allclose(ens.initial_positions[0], np.zeros((4, 2)))


def test_ensemble_to_agents_roundtrip(
    fourbar, member_dims, member_positions, member_scores,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    agents = ens.to_agents()
    assert len(agents) == 3
    # "score" is the first key → primary score used by to_agent
    scores = [a.score for a in agents]
    assert scores == [3.0, 1.0, 2.0]


def test_ensemble_show_uses_cached_trajectory(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=8)

    captured: dict = {}

    def fake_show_linkage(linkage, loci=None, **kwargs):
        captured["linkage"] = linkage
        captured["loci"] = loci
        captured["kwargs"] = kwargs
        return "ANIM"

    monkeypatch.setattr(
        "pylinkage.visualizer.animated.show_linkage",
        fake_show_linkage,
    )
    result = ens.show(idx=1, title="demo")
    assert result == "ANIM"
    assert captured["linkage"] is fourbar
    assert isinstance(captured["loci"], tuple)
    assert len(captured["loci"]) == 8
    assert captured["kwargs"] == {"title": "demo"}


def test_ensemble_show_imports_visualizer_lazily(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    # Pre-simulate so the code path doesn't hit the non-cache branch,
    # and verify the lazy import of show_linkage is wired correctly.
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=4)

    def fake_show_linkage(linkage, loci=None, **kwargs):
        return ("ok", len(loci))

    monkeypatch.setattr(
        "pylinkage.visualizer.animated.show_linkage",
        fake_show_linkage,
    )
    result = ens.show(idx=0)
    assert result[0] == "ok"
    assert result[1] == 4


def test_ensemble_show_simulates_and_rebuilds_member(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    # Cover the non-cache path by making simulate_member populate the
    # trajectory cache so _member_at() on the second call returns a
    # member with a real trajectory.
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    real_sim = ens.simulate

    def fake_simulate_member(idx, iterations=None, dt=1.0):
        # Populate the cache via a full batch simulation
        real_sim(iterations=iterations or 3, store=True)
        return ens._trajectories[idx]

    monkeypatch.setattr(ens, "simulate_member", fake_simulate_member)

    def fake_show_linkage(linkage, loci=None, **kwargs):
        return "ANIM"

    monkeypatch.setattr(
        "pylinkage.visualizer.animated.show_linkage",
        fake_show_linkage,
    )
    assert ens.show(idx=0, iterations=3) == "ANIM"


def test_ensemble_plot_plotly_simulates_and_rebuilds_member(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    real_sim = ens.simulate

    def fake_simulate_member(idx, iterations=None, dt=1.0):
        real_sim(iterations=iterations or 3, store=True)
        return ens._trajectories[idx]

    monkeypatch.setattr(ens, "simulate_member", fake_simulate_member)
    monkeypatch.setattr(
        "pylinkage.visualizer.plotly_viz.plot_linkage_plotly",
        lambda linkage, loci=None, **kwargs: "FIG",
    )
    assert ens.plot_plotly(idx=0, iterations=3) == "FIG"


def test_ensemble_save_svg_simulates_and_rebuilds_member(
    fourbar, member_dims, member_positions, member_scores, monkeypatch, tmp_path,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    real_sim = ens.simulate
    seen: dict = {}

    def fake_simulate_member(idx, iterations=None, dt=1.0):
        real_sim(iterations=iterations or 3, store=True)
        return ens._trajectories[idx]

    monkeypatch.setattr(ens, "simulate_member", fake_simulate_member)

    def fake_save(linkage, path, loci=None, **kwargs):
        seen["ok"] = True

    monkeypatch.setattr(
        "pylinkage.visualizer.drawsvg_viz.save_linkage_svg",
        fake_save,
    )
    ens.save_svg(str(tmp_path / "out.svg"), idx=0, iterations=3)
    assert seen["ok"] is True


def test_ensemble_plot_plotly_uses_cache(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=6)

    captured: dict = {}

    def fake_plot(linkage, loci=None, **kwargs):
        captured["loci"] = loci
        captured["kwargs"] = kwargs
        return "FIG"

    monkeypatch.setattr(
        "pylinkage.visualizer.plotly_viz.plot_linkage_plotly",
        fake_plot,
    )
    result = ens.plot_plotly(idx=2, title="hi")
    assert result == "FIG"
    assert captured["kwargs"]["title"] == "hi"
    assert len(captured["loci"]) == 6


def test_ensemble_plot_plotly_uses_sim(
    fourbar, member_dims, member_positions, member_scores, monkeypatch,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=5)

    def fake_plot(linkage, loci=None, **kwargs):
        return len(loci)

    monkeypatch.setattr(
        "pylinkage.visualizer.plotly_viz.plot_linkage_plotly",
        fake_plot,
    )
    result = ens.plot_plotly(idx=0)
    assert result == 5


def test_ensemble_save_svg_uses_cache(
    fourbar, member_dims, member_positions, member_scores, monkeypatch, tmp_path,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=4)

    captured: dict = {}

    def fake_save(linkage, path, loci=None, **kwargs):
        captured["path"] = path
        captured["loci"] = loci
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "pylinkage.visualizer.drawsvg_viz.save_linkage_svg",
        fake_save,
    )
    target = str(tmp_path / "out.svg")
    ens.save_svg(target, idx=1, extra="value")
    assert captured["path"] == target
    assert len(captured["loci"]) == 4
    assert captured["kwargs"]["extra"] == "value"


def test_ensemble_save_svg_calls_backend(
    fourbar, member_dims, member_positions, member_scores, monkeypatch, tmp_path,
):
    ens = _make_ensemble(fourbar, member_dims, member_positions, member_scores)
    ens.simulate(iterations=3)
    seen: dict = {}

    def fake_save(linkage, path, loci=None, **kwargs):
        seen["called"] = True

    monkeypatch.setattr(
        "pylinkage.visualizer.drawsvg_viz.save_linkage_svg",
        fake_save,
    )
    ens.save_svg(str(tmp_path / "out.svg"), idx=0)
    assert seen.get("called") is True
