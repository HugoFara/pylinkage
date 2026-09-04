"""``orientation_resolution`` replaces ``n_orientation_samples``.

The old name never denoted a number of samples: it was folded into a per-axis
grid resolution floored at 6, so values from 6 to 216 produced an identical
search. The new name states what the parameter is, and makes the cost model
readable -- the grid holds ``resolution ** (n_points - 1)`` candidates.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from pylinkage.synthesis import path_generation

pg_module = importlib.import_module("pylinkage.synthesis.path_generation")

FOUR_POINTS = [(0, 0), (1, 1), (2, 1), (3, 0)]
THREE_POINTS = [(0, 0), (1.5, 1.2), (3, 0)]
FIVE_POINTS = [(0, 0), (1, 1), (2, 1.2), (3, 0.6), (4, 0)]


def signature(result):
    return sorted(
        (
            round(s.crank_length, 6),
            round(s.coupler_length, 6),
            round(s.rocker_length, 6),
            round(s.ground_length, 6),
        )
        for s in result.raw_solutions
    )


class TestLegacyArgument:
    def test_warns(self):
        with pytest.warns(DeprecationWarning, match="n_orientation_samples"):
            path_generation(FOUR_POINTS, n_orientation_samples=36)

    def test_warning_names_the_replacement_and_removal(self):
        with pytest.warns(DeprecationWarning) as record:
            path_generation(FOUR_POINTS, n_orientation_samples=36)
        message = str(record[0].message)
        assert "orientation_resolution" in message
        assert "2.0.0" in message

    def test_still_produces_the_search_it_always_did(self):
        """The old default must give exactly the new default's results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = path_generation(FOUR_POINTS, n_orientation_samples=36)
        assert signature(legacy) == signature(path_generation(FOUR_POINTS))

    @pytest.mark.parametrize(
        ("n_samples", "n_points", "expected"),
        [
            (36, 4, (6, 12)),  # the old default, on four points
            (36, 3, (6, 12)),  # free == 2: sqrt(36) is exactly 6
            (36, 5, (6, 12)),  # free == 4: floored to 6
            (1000, 4, (10, 333)),  # large values did raise the resolution
            (4, 4, (6, 6)),  # small values were floored on both axes
        ],
    )
    def test_translation_reproduces_the_old_formula(self, n_samples, n_points, expected):
        assert pg_module._legacy_orientation_args(n_samples, n_points) == expected

    def test_new_parameter_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            path_generation(FOUR_POINTS, orientation_resolution=6)


class TestResolution:
    def test_default_is_six(self):
        assert pg_module.DEFAULT_ORIENTATION_RESOLUTION == 6

    def test_grid_size_is_resolution_to_the_free_power(self):
        """The documented cost model has to be the real one."""
        for resolution, n_points, expected in ((3, 4, 27), (4, 3, 16), (6, 4, 216)):
            points = FOUR_POINTS[:n_points]
            candidates = list(
                pg_module._generate_orientation_candidates(points, resolution=resolution)
            )
            # base estimate + grid + perturbations + 5 progressive rotations
            grid = expected
            assert grid < len(candidates) <= grid + 1 + 12 + 5

    def test_candidate_count_is_capped(self):
        """Six or more precision points must not run unbounded."""
        points = [(float(i), float(i % 2)) for i in range(7)]
        candidates = list(
            pg_module._generate_orientation_candidates(points, resolution=6)
        )
        # 6**6 is 46656; the guard stops it far short.
        assert len(candidates) <= pg_module.MAX_ORIENTATION_CANDIDATES + 1 + 12 + 5

    def test_raising_resolution_searches_more(self):
        coarse = list(pg_module._generate_orientation_candidates(FOUR_POINTS, resolution=3))
        fine = list(pg_module._generate_orientation_candidates(FOUR_POINTS, resolution=6))
        assert len(fine) > len(coarse)


class TestExponentialCostWarning:
    def test_five_points_warn_about_cost(self):
        """Nine seconds of silence is worse than a warning."""
        with pytest.warns(UserWarning, match="exponentially"):
            path_generation(FIVE_POINTS, max_solutions=1)

    def test_four_points_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            path_generation(FOUR_POINTS)

    def test_three_points_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            path_generation(THREE_POINTS)

    def test_the_warning_is_also_reported_in_the_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = path_generation(FIVE_POINTS, max_solutions=1)
        assert any("exponentially" in w for w in result.warnings)
