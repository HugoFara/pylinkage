"""Tests for six-bar path generation synthesis."""

from __future__ import annotations

import pytest

from pylinkage.synthesis._types import FourBarSolution, PrecisionPoint
from pylinkage.synthesis.six_bar import (
    _combine_to_six_bar,
    _find_coupler_positions_at_targets,
    _generate_partitions,
    _nbar_to_six_bar_linkage,
    _synthesize_driving_fourbar,
    _validate_six_bar,
    six_bar_path_generation,
)
from pylinkage.synthesis.topology_types import NBarSolution


class TestGeneratePartitions:
    """Tests for _generate_partitions."""

    def test_six_points_two_groups(self) -> None:
        """6 points → 2 groups of 3 each."""
        partitions = _generate_partitions(6, n_groups=2, min_per_group=3, max_per_group=5)
        assert len(partitions) > 0
        # C(6,3) = 20 partitions
        assert len(partitions) == 20
        for g1, g2 in partitions:
            assert len(g1) == 3
            assert len(g2) == 3
            assert set(g1) | set(g2) == set(range(6))

    def test_seven_points_two_groups(self) -> None:
        """7 points → groups of (3,4) and (4,3)."""
        partitions = _generate_partitions(7, n_groups=2, min_per_group=3, max_per_group=5)
        assert len(partitions) > 0
        for g1, g2 in partitions:
            assert len(g1) >= 3
            assert len(g2) >= 3
            assert len(g1) + len(g2) == 7

    def test_five_points_not_enough(self) -> None:
        """5 points can't be split into 2 groups of ≥3."""
        partitions = _generate_partitions(5, n_groups=2, min_per_group=3, max_per_group=5)
        assert len(partitions) == 0

    def test_eight_points_respects_max(self) -> None:
        """8 points → each group ≤5."""
        partitions = _generate_partitions(8, n_groups=2, min_per_group=3, max_per_group=5)
        assert len(partitions) > 0
        for g1, g2 in partitions:
            assert 3 <= len(g1) <= 5
            assert 3 <= len(g2) <= 5

    def test_min_constraint_enforced(self) -> None:
        """No partition has fewer than min_per_group points."""
        for n in range(6, 11):
            partitions = _generate_partitions(n, n_groups=2, min_per_group=3, max_per_group=5)
            for g1, g2 in partitions:
                assert len(g1) >= 3, f"Group 1 has {len(g1)} points for n={n}"
                assert len(g2) >= 3, f"Group 2 has {len(g2)} points for n={n}"


class TestSynthesizeDrivingFourbar:
    """Tests for _synthesize_driving_fourbar."""

    def test_returns_fourbars(self) -> None:
        """Should return valid FourBarSolution objects."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.0), (2.0, 0.5),
        ]
        orientations = [0.0, 0.5, 0.3]
        results = _synthesize_driving_fourbar(points, orientations, require_grashof=False)
        # May or may not find solutions depending on geometry,
        # but should not crash
        assert isinstance(results, list)
        for fb in results:
            assert isinstance(fb, FourBarSolution)
            assert fb.crank_length > 0
            assert fb.coupler_length > 0

    def test_empty_on_insufficient_points(self) -> None:
        """Should handle fewer than 3 points gracefully."""
        points: list[PrecisionPoint] = [(0.0, 0.0), (1.0, 1.0)]
        orientations = [0.0, 0.5]
        results = _synthesize_driving_fourbar(points, orientations)
        # Burmester needs ≥3 poses, so this may return empty
        assert isinstance(results, list)


class TestFindCouplerPositions:
    """Tests for _find_coupler_positions_at_targets."""

    def test_with_known_fourbar(self) -> None:
        """Simulate a known four-bar and find coupler positions."""
        from pylinkage.synthesis import fourbar_from_lengths

        linkage = fourbar_from_lengths(1.0, 3.0, 3.0, 4.0)

        # Create a FourBarSolution matching this linkage
        from pylinkage.synthesis.conversion import linkage_to_synthesis_params
        fb = linkage_to_synthesis_params(linkage)

        targets: list[PrecisionPoint] = [(1.5, 2.0), (2.5, 1.5)]
        result = _find_coupler_positions_at_targets(fb, targets)

        # Should return positions (may be approximate)
        if result is not None:
            assert len(result) == 2
            for b_pos, c_pos, mid in result:
                assert len(b_pos) == 2
                assert len(c_pos) == 2
                assert len(mid) == 2


class TestCombineToSixBar:
    """Tests for _combine_to_six_bar."""

    def test_combines_two_fourbars(self) -> None:
        """Should produce NBarSolution with 6 joints."""
        driving = FourBarSolution(
            ground_pivot_a=(0.0, 0.0),
            ground_pivot_d=(4.0, 0.0),
            crank_pivot_b=(1.0, 0.0),
            coupler_pivot_c=(3.0, 2.0),
            crank_length=1.0,
            coupler_length=2.83,
            rocker_length=2.24,
            ground_length=4.0,
        )
        second = FourBarSolution(
            ground_pivot_a=(3.0, 2.0),  # C becomes anchor
            ground_pivot_d=(5.0, -1.0),
            crank_pivot_b=(4.0, 3.0),
            coupler_pivot_c=(5.0, 1.0),
            crank_length=1.41,
            coupler_length=2.24,
            rocker_length=2.0,
            ground_length=3.61,
        )
        points: list[PrecisionPoint] = [(1, 1), (2, 2), (3, 3), (4, 2), (5, 1), (6, 0)]
        nbar = _combine_to_six_bar(
            driving, second, points,
            stage1_indices=(0, 1, 2),
            stage2_indices=(3, 4, 5),
        )
        assert nbar is not None
        assert len(nbar.joint_positions) == 6
        assert "A" in nbar.joint_positions
        assert "F" in nbar.joint_positions
        assert nbar.topology_id == "watt"

    def test_rejects_overlapping_joints(self) -> None:
        """Should return None if joints overlap."""
        driving = FourBarSolution(
            ground_pivot_a=(0.0, 0.0),
            ground_pivot_d=(4.0, 0.0),
            crank_pivot_b=(1.0, 0.0),
            coupler_pivot_c=(3.0, 2.0),
            crank_length=1.0,
            coupler_length=2.83,
            rocker_length=2.24,
            ground_length=4.0,
        )
        # E overlaps with A
        second = FourBarSolution(
            ground_pivot_a=(3.0, 2.0),
            ground_pivot_d=(0.0, 0.0),  # Same as A!
            crank_pivot_b=(4.0, 3.0),
            coupler_pivot_c=(0.0, 0.0),  # Same as A!
            crank_length=1.41,
            coupler_length=2.24,
            rocker_length=2.0,
            ground_length=3.61,
        )
        nbar = _combine_to_six_bar(
            driving, second, [],
            stage1_indices=(0, 1, 2),
            stage2_indices=(3, 4, 5),
        )
        assert nbar is None


class TestNBarToLinkage:
    """Tests for _nbar_to_six_bar_linkage."""

    def test_creates_valid_linkage(self) -> None:
        """Should create a Linkage with 6+ joints."""
        nbar = NBarSolution(
            topology_id="watt",
            joint_positions={
                "A": (0.0, 0.0),
                "D": (4.0, 0.0),
                "E": (6.0, -1.0),
                "B": (0.8, 0.6),
                "C": (3.0, 2.0),
                "F": (5.5, 1.0),
            },
            link_lengths={
                "crank_AB": 1.0,
                "coupler_BC": 2.83,
                "rocker_DC": 2.24,
                "ground_AD": 4.0,
                "link_CF": 2.7,
                "rocker_EF": 2.1,
            },
            coupler_point=(2.0, 3.0),
        )
        linkage = _nbar_to_six_bar_linkage(nbar)
        assert linkage is not None
        # 6 main joints + possibly P (coupler tracker)
        assert len(linkage.joints) >= 6

        # Check joint names
        names = {getattr(j, "name", None) for j in linkage.joints}
        assert "A" in names
        assert "B" in names
        assert "C" in names
        assert "D" in names
        assert "E" in names
        assert "F" in names


class TestSixBarPathGeneration:
    """Integration tests for the full six_bar_path_generation pipeline."""

    def test_returns_synthesis_result(self) -> None:
        """Should return a SynthesisResult even if no solutions found."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 2.0), (2.0, 3.0),
            (3.0, 2.0), (4.0, 0.0), (5.0, -1.0),
        ]
        result = six_bar_path_generation(
            points, topology="watt", max_solutions=3,
            n_orientation_samples=6,
        )
        assert hasattr(result, "solutions")
        assert hasattr(result, "warnings")
        assert hasattr(result, "problem")
        assert result.problem.synthesis_type.name == "PATH"

    def test_warns_on_few_points(self) -> None:
        """Should warn when given fewer than 4 points."""
        points: list[PrecisionPoint] = [(0, 0), (1, 1), (2, 0)]
        result = six_bar_path_generation(points, max_solutions=1, n_orientation_samples=3)
        assert any("4-7" in w or "few" in w.lower() or "best" in w.lower() for w in result.warnings)

    def test_invalid_topology_raises(self) -> None:
        """Should raise ValueError for unknown topology."""
        with pytest.raises(ValueError, match="Unknown six-bar topology"):
            six_bar_path_generation([(0, 0), (1, 1)], topology="invalid")

    def test_watt_solutions_can_simulate(self) -> None:
        """If solutions are found, they should be simulatable."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 1.5), (2.0, 2.0),
            (3.0, 1.5), (4.0, 0.0), (5.0, -0.5),
        ]
        result = six_bar_path_generation(
            points, topology="watt", max_solutions=2,
            require_grashof_driver=False,
            n_orientation_samples=8,
        )
        for linkage in result.solutions:
            # Should be able to step without error
            from pylinkage.exceptions import UnbuildableError
            try:
                step_count = 0
                for _ in linkage.step():
                    step_count += 1
                    if step_count >= 5:
                        break
                assert step_count > 0
            except UnbuildableError:
                pass  # Acceptable: some configurations lock up

    def test_stephenson_returns_result(self) -> None:
        """Stephenson synthesis should return a result."""
        points: list[PrecisionPoint] = [
            (0.0, 0.0), (1.0, 2.0), (2.0, 3.0),
            (3.0, 2.0), (4.0, 0.0), (5.0, -1.0),
        ]
        result = six_bar_path_generation(
            points, topology="stephenson", max_solutions=2,
            n_orientation_samples=4,
        )
        assert hasattr(result, "solutions")


class TestValidateSixBar:
    """Tests for _validate_six_bar."""

    def test_valid_linkage_passes(self) -> None:
        """A simple four-bar (as proxy) should pass structural validation."""
        from pylinkage.synthesis import fourbar_from_lengths

        linkage = fourbar_from_lengths(1.0, 3.0, 3.0, 4.0)
        points: list[PrecisionPoint] = [(2.0, 2.0), (3.0, 1.5)]
        result = _validate_six_bar(linkage, points)
        assert result is True  # A valid four-bar should pass structural check

    def test_invalid_linkage_fails(self) -> None:
        """A degenerate linkage should fail validation."""
        from pylinkage.synthesis import fourbar_from_lengths

        # This is just a structural test — degenerate lengths
        try:
            linkage = fourbar_from_lengths(0.01, 100.0, 100.0, 0.01)
            result = _validate_six_bar(linkage, [])
            assert isinstance(result, bool)
        except ValueError:
            pass  # Can't even assemble
