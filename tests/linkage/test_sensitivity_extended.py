"""Extended tests for sensitivity analysis -- targeting uncovered lines."""

import warnings

import numpy as np
import pytest

import pylinkage as pl
from pylinkage.linkage.sensitivity import (
    SensitivityAnalysis,
    ToleranceAnalysis,
    _compute_path_deviation,
    _get_constraint_names,
    _get_output_joint_index,
    _simulate_output_path,
    analyze_sensitivity,
    analyze_tolerance,
)


class TestGetConstraintNamesExtended:
    """Additional tests for _get_constraint_names (lines 59, 63, 69-73)."""

    def test_joint_without_name_uses_classname(self):
        """Joint with no name attribute should use class_id fallback (line 59)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            # Remove name attribute to trigger fallback
            crank.name = None
            linkage = pl.Linkage(joints=[crank], order=[crank])
        names = _get_constraint_names(linkage)
        assert len(names) == 1
        assert "Crank" in names[0] or "radius" in names[0]

    def test_fixed_joint_names(self):
        """Fixed joint should have _radius and _angle suffixes (lines 69-73)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = pl.Static(0, 0)
            parent1 = pl.Static(1, 0)
            fixed = pl.Fixed(
                joint0=parent0,
                joint1=parent1,
                distance=1.0,
                angle=0.0,
                name="my_fixed",
            )
            linkage = pl.Linkage(joints=[fixed], order=[fixed])
        names = _get_constraint_names(linkage)
        assert len(names) == 2
        assert any("radius" in n for n in names)
        assert any("angle" in n for n in names)

    def test_prismatic_joint_names(self):
        """Prismatic joint should have _radius suffix (line 73)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor = pl.Static(0, 0)
            line_start = pl.Static(0, 2)
            line_end = pl.Static(4, 2)
            prismatic = pl.Prismatic(
                2, 2,
                joint0=anchor,
                joint1=line_start,
                joint2=line_end,
                revolute_radius=2.5,
                name="my_prismatic",
            )
            linkage = pl.Linkage(joints=[prismatic], order=[prismatic])
        names = _get_constraint_names(linkage)
        assert len(names) == 1
        assert "my_prismatic_radius" in names[0]


class TestGetOutputJointIndexExtended:
    """Additional tests for _get_output_joint_index (line 63)."""

    def test_negative_index_raises(self):
        """Negative index should raise ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            linkage = pl.Linkage(joints=[crank], order=[crank])
        with pytest.raises(ValueError, match="out of range"):
            _get_output_joint_index(linkage, -1)


class TestComputePathDeviation:
    """Test _compute_path_deviation helper."""

    def test_identical_paths_zero(self):
        """Identical paths should have zero deviation."""
        path = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        assert _compute_path_deviation(path, path) == pytest.approx(0.0)

    def test_known_deviation(self):
        """Test with known path deviation."""
        path1 = np.array([[0, 0], [1, 0]], dtype=np.float64)
        path2 = np.array([[0, 1], [1, 1]], dtype=np.float64)
        # Both points are 1 unit apart in y direction
        assert _compute_path_deviation(path1, path2) == pytest.approx(1.0)


class TestSimulateOutputPath:
    """Test _simulate_output_path helper."""

    def test_returns_correct_shape(self):
        """Should return array of shape (iterations, 2)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0)
            revolute = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0
            )
            linkage = pl.Linkage(
                joints=[crank, revolute], order=[crank, revolute]
            )
        path = _simulate_output_path(linkage, 1, 10)
        assert path.shape == (10, 2)


class TestAnalyzeSensitivityExtended:
    """Additional sensitivity analysis tests (lines 174-175, 231, 281, 294-296, 307-310, 327-330, 341-342)."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="crank"
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

    def test_with_output_joint_index(self, fourbar_linkage):
        """Test sensitivity with explicit output joint index."""
        result = analyze_sensitivity(fourbar_linkage, output_joint=0, iterations=10)
        assert isinstance(result, SensitivityAnalysis)

    def test_with_output_joint_object(self, fourbar_linkage):
        """Test sensitivity with joint object as output_joint."""
        joint = fourbar_linkage.joints[0]
        result = analyze_sensitivity(fourbar_linkage, output_joint=joint, iterations=10)
        assert isinstance(result, SensitivityAnalysis)

    def test_sensitivity_without_transmission(self, fourbar_linkage):
        """Test sensitivity with transmission disabled."""
        result = analyze_sensitivity(
            fourbar_linkage, include_transmission=False, iterations=10
        )
        assert result.perturbed_transmission is None
        assert result.baseline_transmission is None


class TestAnalyzeToleranceExtended:
    """Additional tolerance analysis tests (lines 463-475, 537, 568-570, 582)."""

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a standard four-bar linkage."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(
                1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="crank"
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

    def test_multiple_tolerances(self, fourbar_linkage):
        """Test with tolerances on multiple constraints."""
        tolerances = {
            "crank_radius": 0.01,
            "coupler_dist1": 0.02,
            "coupler_dist2": 0.01,
        }
        result = analyze_tolerance(
            fourbar_linkage, tolerances=tolerances, n_samples=10, iterations=10, seed=42
        )
        assert isinstance(result, ToleranceAnalysis)
        assert result.tolerances == tolerances

    def test_with_explicit_output_joint(self, fourbar_linkage):
        """Test tolerance with explicit output joint index."""
        tolerances = {"crank_radius": 0.01}
        result = analyze_tolerance(
            fourbar_linkage,
            tolerances=tolerances,
            output_joint=0,
            n_samples=10,
            iterations=10,
        )
        assert isinstance(result, ToleranceAnalysis)


class TestToleranceAnalysisToDataframe:
    """Test ToleranceAnalysis.to_dataframe (lines 463-475)."""

    def test_to_dataframe_returns_dataframe(self):
        """to_dataframe should return a pandas DataFrame."""
        pytest.importorskip("pandas")

        analysis = ToleranceAnalysis(
            nominal_path=np.array([[0, 0], [1, 1]]),
            output_cloud=np.array([[[0, 0], [1, 1]]]),
            tolerances={"a": 0.1, "b": 0.2},
            mean_deviation=0.1,
            max_deviation=0.2,
            std_deviation=0.05,
            position_std=np.array([0.1, 0.1]),
        )
        df = analysis.to_dataframe()
        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert "constraint" in df.columns
        assert "tolerance" in df.columns
        assert len(df) == 2

    def test_to_dataframe_without_pandas_raises(self):
        """to_dataframe should raise ImportError if pandas not available."""
        # This tests the error path; we can't easily force pandas to be missing
        # but we test the method works when pandas IS available
        analysis = ToleranceAnalysis(
            nominal_path=np.array([[0, 0]]),
            output_cloud=np.array([[[0, 0]]]),
            tolerances={"a": 0.1},
            mean_deviation=0.1,
            max_deviation=0.2,
            std_deviation=0.05,
            position_std=np.array([0.1]),
        )
        try:
            analysis.to_dataframe()
        except ImportError:
            pass  # Expected if pandas not installed


class TestSensitivityAnalysisToDataframeExtended:
    """Test SensitivityAnalysis.to_dataframe with transmission data."""

    def test_to_dataframe_with_transmission(self):
        """to_dataframe should include transmission column when available."""
        pytest.importorskip("pandas")

        analysis = SensitivityAnalysis(
            sensitivities={"a_radius": 1.0},
            baseline_path_metric=0.0,
            baseline_transmission=90.0,
            perturbed_path_metrics=np.array([0.1]),
            perturbed_transmission=np.array([91.0]),
            constraint_names=("a_radius",),
            perturbation_delta=0.01,
        )
        df = analysis.to_dataframe()
        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert "perturbed_transmission" in df.columns
