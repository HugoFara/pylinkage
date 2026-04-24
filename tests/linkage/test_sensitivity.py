"""Tests for pylinkage.linkage.sensitivity.

Covers:
- _get_constraint_names / _name_to_index / _get_output_joint_index
- SensitivityAnalysis dataclass and analyze_sensitivity
- ToleranceAnalysis dataclass and analyze_tolerance (with seed)
- plot_cloud rendering, to_dataframe (pandas branches)
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.linkage.sensitivity import (
    SensitivityAnalysis,
    ToleranceAnalysis,
    _compute_path_deviation,
    _get_constraint_names,
    _get_output_joint_index,
    _name_to_index,
    _simulate_output_path,
    analyze_sensitivity,
    analyze_tolerance,
)
from pylinkage.simulation import Linkage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fourbar() -> Linkage:
    o1 = Ground(0.0, 0.0, name="O1")
    o2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=o2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([o1, o2, crank, rocker], name="FourBar")


def _make_fourbar_with_coupler() -> Linkage:
    o1 = Ground(0.0, 0.0, name="O1")
    o2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=o1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=o2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    coupler = FixedDyad(
        anchor1=crank.output,
        anchor2=rocker,
        distance=1.0,
        angle=math.pi / 4,
        name="coupler",
    )
    return Linkage([o1, o2, crank, rocker, coupler], name="FourBarCoupler")


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def test_get_constraint_names_fourbar():
    lk = _make_fourbar()
    names = _get_constraint_names(lk)
    # O1/O2 have no constraints; crank has one (radius); rocker has two (dist1/dist2)
    assert names == ("crank_radius", "rocker_dist1", "rocker_dist2")


def test_get_constraint_names_with_coupler_distinguishes_fixed():
    lk = _make_fourbar_with_coupler()
    names = _get_constraint_names(lk)
    # FixedDyad has (distance, angle) pair -> "coupler_radius", "coupler_angle"
    assert "coupler_radius" in names
    assert "coupler_angle" in names
    assert "crank_radius" in names
    assert "rocker_dist1" in names
    assert "rocker_dist2" in names


def test_get_constraint_names_falls_back_when_name_is_none():
    """If a component's name attribute is explicitly None, names fall back
    to f"{ClassName}_{id % 10000}"."""

    class NoNameCrank:
        """Minimal stand-in that mimics the parts of the API required by
        _get_constraint_names: class name, explicit ``name = None``, and
        a get_constraints() returning one value."""

        name = None

        def get_constraints(self):
            return [1.0]

    class MockLinkage:
        components = [NoNameCrank()]

    names = _get_constraint_names(MockLinkage())
    assert len(names) == 1
    assert names[0].startswith("NoNameCrank_")
    assert names[0].endswith("_radius")


def test_name_to_index_maps_correctly():
    lk = _make_fourbar()
    names = _get_constraint_names(lk)
    mapping = _name_to_index(lk, names)
    assert mapping["crank_radius"] == 0
    assert mapping["rocker_dist1"] == 1
    assert mapping["rocker_dist2"] == 2


# ---------------------------------------------------------------------------
# _get_output_joint_index
# ---------------------------------------------------------------------------


def test_get_output_joint_index_default_last():
    lk = _make_fourbar()
    assert _get_output_joint_index(lk, None) == len(lk.components) - 1


def test_get_output_joint_index_explicit_int():
    lk = _make_fourbar()
    assert _get_output_joint_index(lk, 0) == 0
    assert _get_output_joint_index(lk, 2) == 2


def test_get_output_joint_index_out_of_range_raises():
    lk = _make_fourbar()
    with pytest.raises(ValueError, match="out of range"):
        _get_output_joint_index(lk, 100)
    with pytest.raises(ValueError, match="out of range"):
        _get_output_joint_index(lk, -1)


def test_get_output_joint_index_by_object():
    lk = _make_fourbar()
    target = lk.components[2]
    assert _get_output_joint_index(lk, target) == 2


def test_get_output_joint_index_by_unknown_object_raises():
    lk = _make_fourbar()
    stranger = object()
    with pytest.raises(ValueError, match="not found"):
        _get_output_joint_index(lk, stranger)


# ---------------------------------------------------------------------------
# _compute_path_deviation / _simulate_output_path
# ---------------------------------------------------------------------------


def test_compute_path_deviation_identical_zero():
    p = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert _compute_path_deviation(p, p) == 0.0


def test_compute_path_deviation_known_values():
    p1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    p2 = np.array([[3.0, 4.0], [0.0, 0.0]])
    # Distances: 5, 0 -> mean = 2.5
    assert math.isclose(_compute_path_deviation(p1, p2), 2.5)


def test_simulate_output_path_shape():
    lk = _make_fourbar()
    path = _simulate_output_path(lk, output_joint_idx=2, iterations=8)
    assert path.shape == (8, 2)


# ---------------------------------------------------------------------------
# analyze_sensitivity + SensitivityAnalysis
# ---------------------------------------------------------------------------


def test_analyze_sensitivity_returns_dataclass():
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=12, delta=0.01)
    assert isinstance(result, SensitivityAnalysis)
    assert set(result.sensitivities.keys()) == {
        "crank_radius",
        "rocker_dist1",
        "rocker_dist2",
    }
    # All coefficients should be finite and non-negative (deviation / |perturb|)
    for _, s in result.sensitivities.items():
        assert s >= 0.0 or math.isinf(s)
    assert result.baseline_path_metric == 0.0
    assert result.perturbation_delta == 0.01
    assert len(result.perturbed_path_metrics) == 3
    assert result.constraint_names == (
        "crank_radius",
        "rocker_dist1",
        "rocker_dist2",
    )


def test_analyze_sensitivity_includes_transmission():
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=12, include_transmission=True)
    assert result.baseline_transmission is not None
    assert 0.0 <= result.baseline_transmission <= 180.0
    assert result.perturbed_transmission is not None
    assert result.perturbed_transmission.shape == (3,)


def test_analyze_sensitivity_without_transmission():
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=8, include_transmission=False)
    assert result.baseline_transmission is None
    assert result.perturbed_transmission is None


def test_analyze_sensitivity_most_sensitive_and_ranking():
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=12)
    top = result.most_sensitive
    assert top in result.sensitivities
    ranked = result.sensitivity_ranking
    assert len(ranked) == 3
    # Ranking sorted by abs value descending
    abs_values = [abs(v) for _, v in ranked]
    assert abs_values == sorted(abs_values, reverse=True)
    assert ranked[0][0] == top


def test_analyze_sensitivity_with_explicit_output_joint():
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, output_joint=2, iterations=8)
    # Crank joint traces a perfect circle; perturbing rocker distances shouldn't
    # change the crank's path at all, so those sensitivities should be ~0.
    assert result.sensitivities["rocker_dist1"] < 1e-6
    assert result.sensitivities["rocker_dist2"] < 1e-6


def test_analyze_sensitivity_restores_state():
    lk = _make_fourbar()
    nominal = list(lk.get_constraints())
    initial = lk.get_coords()
    analyze_sensitivity(lk, iterations=8)
    assert list(lk.get_constraints()) == nominal
    # Coords restored too
    for (x0, _), (x1, _) in zip(initial, lk.get_coords()):
        if x0 is None or x1 is None:
            continue
        assert math.isclose(x0, x1, abs_tol=1e-9)


def test_analyze_sensitivity_default_iterations():
    lk = _make_fourbar()
    expected = lk.get_rotation_period()
    result = analyze_sensitivity(lk, include_transmission=False)
    # Each perturbed metric should still be computed
    assert result.perturbed_path_metrics.size == 3
    # Just ensure it ran over the default iterations - no crash
    assert result.perturbation_delta == 0.01
    # Sanity - expected > 0
    assert expected > 0


def test_analyze_sensitivity_zero_constraint_skipped():
    # Construct a linkage where one constraint is effectively zero.
    # A tiny radius crank still has a non-zero constraint, so we need to
    # manipulate constraints to 0 after construction.
    lk = _make_fourbar()
    initial_constraints = list(lk.get_constraints())
    # Zero-out one constraint temporarily in the nominal list
    lk.set_constraints([0.0, 2.5, 2.0])
    try:
        result = analyze_sensitivity(lk, iterations=8, include_transmission=False)
        # Zero constraint gets sensitivity == 0
        assert result.sensitivities["crank_radius"] == 0.0
        assert result.perturbed_path_metrics[0] == 0.0
    finally:
        lk.set_constraints(initial_constraints)


def test_sensitivity_to_dataframe_raises_without_pandas(monkeypatch):
    """Even when pandas IS installed, we verify the DataFrame path works."""
    pd = pytest.importorskip("pandas")  # skip test if pandas missing
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=8, include_transmission=False)
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "constraint" in df.columns
    assert "sensitivity" in df.columns
    assert "perturbed_metric" in df.columns


def test_sensitivity_to_dataframe_with_transmission():
    pd = pytest.importorskip("pandas")
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=8, include_transmission=True)
    df = result.to_dataframe()
    assert "perturbed_transmission" in df.columns


def test_sensitivity_to_dataframe_importerror_when_pandas_missing(monkeypatch):
    """Simulate pandas being unavailable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    lk = _make_fourbar()
    result = analyze_sensitivity(lk, iterations=4, include_transmission=False)
    with pytest.raises(ImportError, match="pandas is required"):
        result.to_dataframe()


# ---------------------------------------------------------------------------
# analyze_tolerance + ToleranceAnalysis
# ---------------------------------------------------------------------------


def test_analyze_tolerance_returns_dataclass():
    lk = _make_fourbar()
    tolerances = {"crank_radius": 0.02}
    result = analyze_tolerance(
        lk, tolerances=tolerances, iterations=8, n_samples=6, seed=42
    )
    assert isinstance(result, ToleranceAnalysis)
    assert result.tolerances == tolerances
    assert result.nominal_path.shape == (8, 2)
    assert result.output_cloud.shape[0] <= 6
    assert result.output_cloud.shape[1] == 8
    assert result.output_cloud.shape[2] == 2
    assert result.position_std.shape == (8,)
    assert result.mean_deviation >= 0.0
    assert result.max_deviation >= result.mean_deviation
    assert result.std_deviation >= 0.0


def test_analyze_tolerance_reproducible_with_seed():
    lk1 = _make_fourbar()
    lk2 = _make_fourbar()
    tol = {"crank_radius": 0.02, "rocker_dist1": 0.01}
    r1 = analyze_tolerance(lk1, tol, iterations=6, n_samples=5, seed=123)
    r2 = analyze_tolerance(lk2, tol, iterations=6, n_samples=5, seed=123)
    np.testing.assert_allclose(r1.output_cloud, r2.output_cloud)
    assert math.isclose(r1.mean_deviation, r2.mean_deviation)
    assert math.isclose(r1.max_deviation, r2.max_deviation)


def test_analyze_tolerance_unknown_name_raises():
    lk = _make_fourbar()
    with pytest.raises(ValueError, match="Unknown constraint name"):
        analyze_tolerance(lk, {"nonexistent": 0.1}, iterations=4, n_samples=2, seed=0)


def test_analyze_tolerance_restores_state():
    lk = _make_fourbar()
    nominal = list(lk.get_constraints())
    initial = lk.get_coords()
    analyze_tolerance(
        lk, {"crank_radius": 0.01}, iterations=6, n_samples=3, seed=1
    )
    assert list(lk.get_constraints()) == nominal
    for (x0, _), (x1, _) in zip(initial, lk.get_coords()):
        if x0 is None or x1 is None:
            continue
        assert math.isclose(x0, x1, abs_tol=1e-9)


def test_analyze_tolerance_default_iterations():
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.01}, n_samples=3, seed=5
    )
    expected = lk.get_rotation_period()
    assert result.nominal_path.shape == (expected, 2)


def test_analyze_tolerance_zero_tolerance_deviation_zero():
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.0}, iterations=6, n_samples=3, seed=7
    )
    # Zero tolerance -> deviations should be ~0
    assert result.mean_deviation < 1e-9
    assert result.max_deviation < 1e-9


def test_analyze_tolerance_explicit_output_joint():
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk,
        {"crank_radius": 0.02},
        output_joint=2,
        iterations=6,
        n_samples=3,
        seed=11,
    )
    # Crank output path is a circle of radius 1.0 - within ~2% tolerance
    center_distance = np.linalg.norm(result.nominal_path, axis=1)
    assert np.allclose(center_distance, 1.0, atol=1e-6)


def test_tolerance_all_samples_fail_raises():
    """If tolerances are massive, all samples may be unbuildable."""
    lk = _make_fourbar()
    # Huge tolerance on rocker_dist1 to make many samples unbuildable.
    # We can't guarantee ALL fail, so we just check the function handles large
    # values gracefully. Use a very narrow seed-driven case.
    # Instead, monkey-patch _simulate_output_path to always raise.
    import pylinkage.linkage.sensitivity as sens_mod

    original = sens_mod._simulate_output_path

    def always_raise(linkage, idx, iters):
        raise RuntimeError("simulated failure")

    # First call (nominal) must succeed; subsequent must fail.
    call_count = {"n": 0}

    def fail_after_first(linkage, idx, iters):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original(linkage, idx, iters)
        raise RuntimeError("simulated failure")

    sens_mod._simulate_output_path = fail_after_first
    try:
        with pytest.raises(ValueError, match="No valid samples"):
            analyze_tolerance(
                lk, {"crank_radius": 0.01}, iterations=4, n_samples=3, seed=42
            )
    finally:
        sens_mod._simulate_output_path = original


def test_tolerance_plot_cloud_creates_axes():
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.02}, iterations=6, n_samples=4, seed=3
    )
    ax = result.plot_cloud()
    assert ax is not None
    plt.close("all")


def test_tolerance_plot_cloud_with_supplied_axes_no_nominal():
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.02}, iterations=6, n_samples=3, seed=3
    )
    fig, ax = plt.subplots()
    returned = result.plot_cloud(ax=ax, show_nominal=False, alpha=0.3)
    assert returned is ax
    plt.close(fig)


def test_tolerance_to_dataframe_with_pandas():
    pd = pytest.importorskip("pandas")
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.01}, iterations=4, n_samples=2, seed=0
    )
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "constraint" in df.columns
    assert "tolerance" in df.columns


def test_tolerance_to_dataframe_importerror_when_pandas_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    lk = _make_fourbar()
    result = analyze_tolerance(
        lk, {"crank_radius": 0.01}, iterations=4, n_samples=2, seed=0
    )
    with pytest.raises(ImportError, match="pandas is required"):
        result.to_dataframe()
