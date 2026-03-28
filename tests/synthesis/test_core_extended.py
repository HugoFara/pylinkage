"""Extended tests for synthesis core module -- targeting uncovered lines."""

import unittest

import numpy as np

from pylinkage.synthesis._types import DyadSolution, SynthesisType
from pylinkage.synthesis.core import (
    BurmesterCurves,
    Dyad,
    SynthesisProblem,
    SynthesisResult,
)


class TestSynthesisProblemNumPrecision(unittest.TestCase):
    """Test num_precision_positions for all synthesis types (lines 67-72)."""

    def test_function_type_returns_angle_pairs_count(self):
        """FUNCTION type should count angle_pairs (line 67-68)."""
        problem = SynthesisProblem(
            synthesis_type=SynthesisType.FUNCTION,
            angle_pairs=[(0.0, 0.0), (0.5, 0.6), (1.0, 1.1)],
        )
        self.assertEqual(problem.num_precision_positions, 3)

    def test_path_type_returns_precision_points_count(self):
        """PATH type should count precision_points (line 69-70)."""
        problem = SynthesisProblem(
            synthesis_type=SynthesisType.PATH,
            precision_points=[(1.0, 2.0), (3.0, 4.0)],
        )
        self.assertEqual(problem.num_precision_positions, 2)

    def test_motion_type_returns_poses_count(self):
        """MOTION type should count poses (line 71-72)."""
        from pylinkage.synthesis._types import Pose

        problem = SynthesisProblem(
            synthesis_type=SynthesisType.MOTION,
            poses=[Pose(0, 0, 0), Pose(1, 1, 0.5)],
        )
        self.assertEqual(problem.num_precision_positions, 2)

    def test_empty_function(self):
        """Empty angle_pairs should return 0."""
        problem = SynthesisProblem(
            synthesis_type=SynthesisType.FUNCTION,
        )
        self.assertEqual(problem.num_precision_positions, 0)


class TestSynthesisResultProtocol(unittest.TestCase):
    """Test SynthesisResult __len__, __iter__, __getitem__, __bool__ (lines 99, 103, 107, 111)."""

    def _make_result(self, n_solutions=0):
        """Helper to create a SynthesisResult with mock solutions."""
        # Use None as placeholder since we're testing container protocol
        solutions = [None] * n_solutions  # type: ignore
        return SynthesisResult(
            solutions=solutions,
            raw_solutions=[],
            problem=SynthesisProblem(synthesis_type=SynthesisType.FUNCTION),
        )

    def test_len_zero(self):
        """len() should return 0 for empty results (line 99)."""
        result = self._make_result(0)
        self.assertEqual(len(result), 0)

    def test_len_nonzero(self):
        """len() should return count of solutions."""
        result = self._make_result(3)
        self.assertEqual(len(result), 3)

    def test_iter(self):
        """__iter__ should iterate over solutions (line 103)."""
        result = self._make_result(2)
        items = list(result)
        self.assertEqual(len(items), 2)

    def test_getitem(self):
        """__getitem__ should return solution by index (line 107)."""
        result = self._make_result(3)
        item = result[1]
        self.assertIsNone(item)  # We used None as placeholder

    def test_bool_false(self):
        """__bool__ should be False for empty results (line 111)."""
        result = self._make_result(0)
        self.assertFalse(result)

    def test_bool_true(self):
        """__bool__ should be True for non-empty results (line 111)."""
        result = self._make_result(1)
        self.assertTrue(result)


class TestDyad(unittest.TestCase):
    """Test Dyad dataclass (lines 137, 145, 152)."""

    def test_link_length(self):
        """link_length should return distance between circle and center (line 137)."""
        dyad = Dyad(circle_point=complex(3, 4), center_point=complex(0, 0))
        self.assertAlmostEqual(dyad.link_length, 5.0)

    def test_to_cartesian(self):
        """to_cartesian should convert to (x,y) tuples (line 145)."""
        dyad = Dyad(circle_point=complex(1, 2), center_point=complex(3, 4))
        cp, cenp = dyad.to_cartesian()
        self.assertEqual(cp, (1.0, 2.0))
        self.assertEqual(cenp, (3.0, 4.0))

    def test_to_dyad_solution(self):
        """to_dyad_solution should return DyadSolution (line 152)."""
        dyad = Dyad(circle_point=complex(1, 2), center_point=complex(3, 4))
        sol = dyad.to_dyad_solution()
        self.assertIsInstance(sol, DyadSolution)
        self.assertEqual(sol.circle_point, complex(1, 2))
        self.assertEqual(sol.center_point, complex(3, 4))


class TestBurmesterCurves(unittest.TestCase):
    """Test BurmesterCurves (lines 213, 226-230, 243-244)."""

    def _make_curves(self, n=10, is_discrete=False):
        """Helper to create test BurmesterCurves."""
        return BurmesterCurves(
            circle_curve=np.arange(n, dtype=np.complex128),
            center_curve=np.arange(n, dtype=np.complex128) + 1j,
            parameter=np.linspace(0, 1, n),
            is_discrete=is_discrete,
        )

    def test_get_all_dyads(self):
        """get_all_dyads should return list of Dyads (line 213)."""
        curves = self._make_curves(5)
        dyads = curves.get_all_dyads()
        self.assertEqual(len(dyads), 5)
        self.assertIsInstance(dyads[0], Dyad)

    def test_sample_continuous_downsamples(self):
        """sample() on continuous curve should downsample (lines 226-230)."""
        curves = self._make_curves(20, is_discrete=False)
        sampled = curves.sample(5)
        self.assertEqual(len(sampled), 5)

    def test_sample_discrete_returns_self(self):
        """sample() on discrete points should return self (line 226)."""
        curves = self._make_curves(5, is_discrete=True)
        sampled = curves.sample(3)
        # Should return same object since discrete
        self.assertIs(sampled, curves)

    def test_sample_when_fewer_than_requested(self):
        """sample() when n_samples > len returns self (line 226)."""
        curves = self._make_curves(3, is_discrete=False)
        sampled = curves.sample(10)
        self.assertIs(sampled, curves)

    def test_filter_finite(self):
        """filter_finite should remove NaN/inf (lines 243-244)."""
        curves = BurmesterCurves(
            circle_curve=np.array([1 + 0j, float("nan") + 0j, 3 + 0j], dtype=np.complex128),
            center_curve=np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex128),
            parameter=np.array([0.0, 0.5, 1.0]),
        )
        filtered = curves.filter_finite()
        self.assertEqual(len(filtered), 2)

    def test_len(self):
        """__len__ should return number of points."""
        curves = self._make_curves(7)
        self.assertEqual(len(curves), 7)

    def test_bool_true(self):
        """__bool__ should be True for non-empty curves."""
        curves = self._make_curves(3)
        self.assertTrue(curves)

    def test_bool_false(self):
        """__bool__ should be False for empty curves."""
        curves = BurmesterCurves(
            circle_curve=np.array([], dtype=np.complex128),
            center_curve=np.array([], dtype=np.complex128),
            parameter=np.array([], dtype=np.float64),
        )
        self.assertFalse(curves)

    def test_get_dyad(self):
        """get_dyad should return a single Dyad."""
        curves = self._make_curves(5)
        dyad = curves.get_dyad(2)
        self.assertIsInstance(dyad, Dyad)
        self.assertEqual(dyad.circle_point, curves.circle_curve[2])
        self.assertEqual(dyad.center_point, curves.center_curve[2])


if __name__ == "__main__":
    unittest.main()
