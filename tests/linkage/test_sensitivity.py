"""Tests for sensitivity and tolerance analysis."""

import warnings

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.linkage.sensitivity import (
    SensitivityAnalysis,
    ToleranceAnalysis,
    _get_constraint_names,
    _get_output_joint_index,
    _name_to_index,
    analyze_sensitivity,
    analyze_tolerance,
)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetConstraintNames:
    """Test the constraint name generation helper."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="crank",
            )
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
                name="coupler",
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
                name="Test Four-Bar",
            )
        return linkage

    def test_returns_tuple_of_strings(self, fourbar_linkage):
        """Should return a tuple of string names."""
        names = _get_constraint_names(fourbar_linkage)
        assert isinstance(names, tuple)
        assert all(isinstance(n, str) for n in names)

    def test_correct_count_for_fourbar(self, fourbar_linkage):
        """Four-bar has 1 crank + 2 revolute = 3 constraints."""
        names = _get_constraint_names(fourbar_linkage)
        # Crank: 1 constraint (radius)
        # Revolute: 2 constraints (dist1, dist2)
        assert len(names) == 3

    def test_names_contain_joint_name(self, fourbar_linkage):
        """Names should include the joint's name attribute."""
        names = _get_constraint_names(fourbar_linkage)
        # Check that crank and coupler names are present
        assert any("crank" in n for n in names)
        assert any("coupler" in n for n in names)

    def test_crank_has_radius_suffix(self, fourbar_linkage):
        """Crank constraints should have _radius suffix."""
        names = _get_constraint_names(fourbar_linkage)
        crank_names = [n for n in names if "crank" in n]
        assert len(crank_names) == 1
        assert crank_names[0].endswith("_radius")

    def test_revolute_has_dist_suffixes(self, fourbar_linkage):
        """Revolute constraints should have _dist1 and _dist2 suffixes."""
        names = _get_constraint_names(fourbar_linkage)
        coupler_names = [n for n in names if "coupler" in n]
        assert len(coupler_names) == 2
        assert any(n.endswith("_dist1") for n in coupler_names)
        assert any(n.endswith("_dist2") for n in coupler_names)

    def test_unique_names(self, fourbar_linkage):
        """All constraint names should be unique."""
        names = _get_constraint_names(fourbar_linkage)
        assert len(names) == len(set(names))


class TestNameToIndex:
    """Test the name to index mapping helper."""

    def test_creates_valid_mapping(self):
        """Should create dict from names to indices."""
        names = ("a_radius", "b_dist1", "b_dist2")
        mapping = _name_to_index(None, names)  # type: ignore[arg-type]
        assert mapping == {"a_radius": 0, "b_dist1": 1, "b_dist2": 2}


class TestGetOutputJointIndex:
    """Test output joint resolution helper."""

    @pytest.fixture
    def simple_linkage(self):
        """Create a simple linkage with multiple joints."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_none_returns_last_joint(self, simple_linkage):
        """None should return index of last joint."""
        idx = _get_output_joint_index(simple_linkage, None)
        assert idx == len(simple_linkage.joints) - 1

    def test_integer_index_valid(self, simple_linkage):
        """Valid integer index should be returned as-is."""
        idx = _get_output_joint_index(simple_linkage, 0)
        assert idx == 0

    def test_integer_index_out_of_range_raises(self, simple_linkage):
        """Out of range index should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            _get_output_joint_index(simple_linkage, 100)

    def test_joint_object_resolved(self, simple_linkage):
        """Joint object should be resolved to its index."""
        joint = simple_linkage.joints[0]
        idx = _get_output_joint_index(simple_linkage, joint)
        assert idx == 0

    def test_unknown_joint_raises(self, simple_linkage):
        """Unknown joint object should raise ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            unknown_joint = pl.Crank(0, 0, joint0=(0, 0), angle=0.1, distance=1.0)
        with pytest.raises(ValueError, match="not found"):
            _get_output_joint_index(simple_linkage, unknown_joint)


# =============================================================================
# Sensitivity Analysis Tests
# =============================================================================


class TestSensitivityAnalysis:
    """Test the sensitivity analysis function and dataclass."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="crank",
            )
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
                name="coupler",
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_returns_sensitivity_analysis(self, fourbar_linkage):
        """Should return SensitivityAnalysis dataclass."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert isinstance(result, SensitivityAnalysis)

    def test_sensitivities_is_dict(self, fourbar_linkage):
        """sensitivities should be a dict of name -> float."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert isinstance(result.sensitivities, dict)
        assert all(isinstance(k, str) for k in result.sensitivities.keys())
        assert all(isinstance(v, (int, float)) for v in result.sensitivities.values())

    def test_constraint_names_tuple(self, fourbar_linkage):
        """constraint_names should be a tuple of strings."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert isinstance(result.constraint_names, tuple)
        assert all(isinstance(n, str) for n in result.constraint_names)

    def test_sensitivities_non_negative(self, fourbar_linkage):
        """Sensitivities should be non-negative (deviation can only increase)."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        # Allow for numerical noise
        for sens in result.sensitivities.values():
            assert sens >= -1e-10 or sens == float("inf")

    def test_perturbed_metrics_array(self, fourbar_linkage):
        """perturbed_path_metrics should be a numpy array."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert isinstance(result.perturbed_path_metrics, np.ndarray)
        assert len(result.perturbed_path_metrics) == len(result.constraint_names)

    def test_baseline_path_metric_is_zero(self, fourbar_linkage):
        """Baseline path metric should be 0 (nominal vs nominal)."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert result.baseline_path_metric == 0.0

    def test_perturbation_delta_stored(self, fourbar_linkage):
        """perturbation_delta should match input."""
        result = analyze_sensitivity(fourbar_linkage, delta=0.05, iterations=10)
        assert result.perturbation_delta == 0.05

    def test_most_sensitive_property(self, fourbar_linkage):
        """most_sensitive should return a constraint name."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        assert result.most_sensitive in result.constraint_names

    def test_sensitivity_ranking_sorted(self, fourbar_linkage):
        """sensitivity_ranking should be sorted by absolute sensitivity."""
        result = analyze_sensitivity(fourbar_linkage, iterations=10)
        ranking = result.sensitivity_ranking
        assert isinstance(ranking, list)
        sensitivities = [abs(s) for _, s in ranking]
        assert sensitivities == sorted(sensitivities, reverse=True)

    def test_include_transmission_false(self, fourbar_linkage):
        """include_transmission=False should skip transmission analysis."""
        result = analyze_sensitivity(
            fourbar_linkage,
            include_transmission=False,
            iterations=10,
        )
        assert result.baseline_transmission is None
        assert result.perturbed_transmission is None

    def test_include_transmission_true(self, fourbar_linkage):
        """include_transmission=True should include transmission analysis."""
        result = analyze_sensitivity(
            fourbar_linkage,
            include_transmission=True,
            iterations=10,
        )
        # Four-bar has transmission angle
        assert result.baseline_transmission is not None
        assert result.perturbed_transmission is not None

    def test_state_restored_after_analysis(self, fourbar_linkage):
        """Linkage state should be restored after analysis."""
        initial_coords = fourbar_linkage.get_coords()
        initial_constraints = fourbar_linkage.get_num_constraints()

        analyze_sensitivity(fourbar_linkage, iterations=10)

        final_coords = fourbar_linkage.get_coords()
        final_constraints = fourbar_linkage.get_num_constraints()

        assert initial_coords == final_coords
        assert initial_constraints == final_constraints


class TestSensitivityAnalysisDataclass:
    """Test SensitivityAnalysis dataclass properties."""

    def test_frozen_dataclass(self):
        """Dataclass should be immutable."""
        analysis = SensitivityAnalysis(
            sensitivities={"a": 1.0},
            baseline_path_metric=0.0,
            baseline_transmission=90.0,
            perturbed_path_metrics=np.array([0.1]),
            perturbed_transmission=np.array([91.0]),
            constraint_names=("a",),
            perturbation_delta=0.01,
        )
        with pytest.raises(AttributeError):
            analysis.baseline_path_metric = 1.0  # type: ignore[misc]


class TestSensitivityAnalysisToDataframe:
    """Test to_dataframe method."""

    def test_to_dataframe_returns_dataframe(self):
        """to_dataframe should return a pandas DataFrame."""
        pytest.importorskip("pandas")

        analysis = SensitivityAnalysis(
            sensitivities={"a_radius": 1.0, "b_dist1": 2.0},
            baseline_path_metric=0.0,
            baseline_transmission=90.0,
            perturbed_path_metrics=np.array([0.1, 0.2]),
            perturbed_transmission=np.array([91.0, 92.0]),
            constraint_names=("a_radius", "b_dist1"),
            perturbation_delta=0.01,
        )
        df = analysis.to_dataframe()
        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert "constraint" in df.columns
        assert "sensitivity" in df.columns


class TestLinkageSensitivityConvenienceMethod:
    """Test the Linkage.sensitivity_analysis() convenience method."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            revolute = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_method_returns_sensitivity_analysis(self, fourbar_linkage):
        """Method should return SensitivityAnalysis."""
        result = fourbar_linkage.sensitivity_analysis(iterations=10)
        assert isinstance(result, SensitivityAnalysis)


# =============================================================================
# Tolerance Analysis Tests
# =============================================================================


class TestToleranceAnalysis:
    """Test the tolerance analysis function and dataclass."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0,
                joint0=(0, 0),
                angle=0.1,
                distance=1.0,
                name="crank",
            )
            revolute = pl.Revolute(
                3, 1,
                joint0=crank,
                joint1=(4, 0),
                distance0=3.0,
                distance1=2.0,
                name="coupler",
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_returns_tolerance_analysis(self, fourbar_linkage):
        """Should return ToleranceAnalysis dataclass."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
        )
        assert isinstance(result, ToleranceAnalysis)

    def test_nominal_path_shape(self, fourbar_linkage):
        """nominal_path should have shape (iterations, 2)."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=20,
        )
        assert result.nominal_path.shape == (20, 2)

    def test_output_cloud_shape(self, fourbar_linkage):
        """output_cloud should have shape (n_samples, iterations, 2)."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=15,
            iterations=20,
        )
        # May have fewer samples if some are unbuildable
        assert result.output_cloud.shape[1] == 20
        assert result.output_cloud.shape[2] == 2
        assert result.output_cloud.shape[0] <= 15

    def test_statistics_computed(self, fourbar_linkage):
        """Statistics should be computed."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
        )
        assert isinstance(result.mean_deviation, float)
        assert isinstance(result.max_deviation, float)
        assert isinstance(result.std_deviation, float)

    def test_mean_less_than_max(self, fourbar_linkage):
        """Mean deviation should be <= max deviation."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=20,
            iterations=10,
        )
        assert result.mean_deviation <= result.max_deviation

    def test_position_std_shape(self, fourbar_linkage):
        """position_std should have shape (iterations,)."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=20,
        )
        assert result.position_std.shape == (20,)

    def test_tolerances_stored(self, fourbar_linkage):
        """tolerances dict should be stored in result."""
        tolerances = {"crank_radius": 0.05, "coupler_dist1": 0.1}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
        )
        assert result.tolerances == tolerances

    def test_unknown_tolerance_raises(self, fourbar_linkage):
        """Unknown constraint name should raise ValueError."""
        tolerances = {"nonexistent_constraint": 0.01}
        with pytest.raises(ValueError, match="Unknown constraint"):
            analyze_tolerance(
                fourbar_linkage,
                tolerances=tolerances,
                n_samples=10,
                iterations=10,
            )

    def test_seed_reproducibility(self, fourbar_linkage):
        """Same seed should give same results."""
        tolerances = {"crank_radius": 0.01}
        result1 = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
            seed=42,
        )
        result2 = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
            seed=42,
        )
        np.testing.assert_array_almost_equal(
            result1.output_cloud, result2.output_cloud
        )

    def test_state_restored_after_analysis(self, fourbar_linkage):
        """Linkage state should be restored after analysis."""
        initial_coords = fourbar_linkage.get_coords()
        initial_constraints = fourbar_linkage.get_num_constraints()

        tolerances = {"crank_radius": 0.01}
        analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
        )

        final_coords = fourbar_linkage.get_coords()
        final_constraints = fourbar_linkage.get_num_constraints()

        assert initial_coords == final_coords
        assert initial_constraints == final_constraints


class TestToleranceAnalysisDataclass:
    """Test ToleranceAnalysis dataclass properties."""

    def test_frozen_dataclass(self):
        """Dataclass should be immutable."""
        analysis = ToleranceAnalysis(
            nominal_path=np.array([[0, 0], [1, 1]]),
            output_cloud=np.array([[[0, 0], [1, 1]]]),
            tolerances={"a": 0.1},
            mean_deviation=0.1,
            max_deviation=0.2,
            std_deviation=0.05,
            position_std=np.array([0.1, 0.1]),
        )
        with pytest.raises(AttributeError):
            analysis.mean_deviation = 0.5  # type: ignore[misc]


class TestToleranceAnalysisPlotCloud:
    """Test plot_cloud method."""

    def test_plot_cloud_returns_axes(self):
        """plot_cloud should return matplotlib axes."""
        pytest.importorskip("matplotlib")

        analysis = ToleranceAnalysis(
            nominal_path=np.array([[0, 0], [1, 1], [2, 0]]),
            output_cloud=np.array([[[0.1, 0.1], [1.1, 1.1], [2.1, 0.1]]]),
            tolerances={"a": 0.1},
            mean_deviation=0.1,
            max_deviation=0.2,
            std_deviation=0.05,
            position_std=np.array([0.1, 0.1, 0.1]),
        )
        import matplotlib.pyplot as plt

        ax = analysis.plot_cloud()
        assert ax is not None
        plt.close("all")


class TestLinkageToleranceConvenienceMethod:
    """Test the Linkage.tolerance_analysis() convenience method."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="crank"
            )
            revolute = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0
            )
            linkage = pl.Linkage(
                joints=[crank, revolute],
                order=[crank, revolute],
            )
        return linkage

    def test_method_returns_tolerance_analysis(self, fourbar_linkage):
        """Method should return ToleranceAnalysis."""
        tolerances = {"crank_radius": 0.01}
        result = fourbar_linkage.tolerance_analysis(
            tolerances=tolerances,
            n_samples=10,
            iterations=10,
        )
        assert isinstance(result, ToleranceAnalysis)
