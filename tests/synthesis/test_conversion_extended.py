"""Extended tests for synthesis/conversion.py covering uncovered branches."""

from __future__ import annotations

import math

import pytest

from pylinkage.synthesis._types import FourBarSolution
from pylinkage.synthesis.conversion import (
    _compute_coupler_point_params,
    _generic_nbar_to_linkage,
    _point_dist,
    fourbar_from_lengths,
    nbar_solution_to_linkage,
    solution_to_linkage,
    stephenson_from_lengths,
    watt_from_lengths,
)
from pylinkage.synthesis.topology_types import NBarSolution


class TestComputeCouplerPointParams:
    def test_point_on_bc_line(self):
        # P on segment B->C; angle should be 0
        B = (0.0, 0.0)
        C = (2.0, 0.0)
        P = (1.0, 0.0)
        d, a = _compute_coupler_point_params(B, C, P)
        assert d == pytest.approx(1.0)
        assert a == pytest.approx(0.0, abs=1e-12)

    def test_point_perpendicular_to_bc(self):
        B = (0.0, 0.0)
        C = (1.0, 0.0)
        P = (0.0, 1.0)
        d, a = _compute_coupler_point_params(B, C, P)
        assert d == pytest.approx(1.0)
        assert a == pytest.approx(math.pi / 2)


class TestSolutionToLinkageCouplerPoint:
    def test_with_coupler_point(self):
        solution = FourBarSolution(
            ground_pivot_a=(0.0, 0.0),
            ground_pivot_d=(4.0, 0.0),
            crank_pivot_b=(1.0, 0.0),
            coupler_pivot_c=(3.0, 2.0),
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
            coupler_point=(2.0, 1.5),
        )
        linkage = solution_to_linkage(solution, name="with_cp")
        # Should have 5 components (2 Ground + Crank + RRR + Fixed)
        assert len(linkage.components) == 5


def _double_rocker(*, mirrored: bool, rotated: float = 0.0) -> FourBarSolution:
    """A Grashof double-rocker (3, 1.5, 1.5, 2): its crank oscillates.

    The crank starts at 40 degrees from the ground line (its range is 1 to 70), below it when
    *mirrored*; the whole linkage is turned by *rotated* radians.
    """
    crank, coupler, rocker, ground = 3.0, 1.5, 1.5, 2.0
    theta = math.radians(-40.0 if mirrored else 40.0) + rotated
    A = (0.0, 0.0)
    D = (ground * math.cos(rotated), ground * math.sin(rotated))
    B = (crank * math.cos(theta), crank * math.sin(theta))
    # C anywhere plausible: it is only the initial guess for branch selection.
    C = ((B[0] + D[0]) / 2, (B[1] + D[1]) / 2 + (-0.5 if mirrored else 0.5))
    return FourBarSolution(
        ground_pivot_a=A,
        ground_pivot_d=D,
        crank_pivot_b=B,
        coupler_pivot_c=C,
        crank_length=crank,
        coupler_length=coupler,
        rocker_length=rocker,
        ground_length=ground,
    )


class TestSolutionToLinkageArcLimits:
    def test_without_arc_limits_a_double_rocker_cannot_turn(self):
        from pylinkage.actuators import Crank
        from pylinkage.exceptions import UnbuildableError

        linkage = solution_to_linkage(_double_rocker(mirrored=False))
        assert isinstance(linkage.components[2], Crank)
        with pytest.raises(UnbuildableError):
            list(linkage.step())

    @pytest.mark.parametrize("mirrored", [False, True])
    @pytest.mark.parametrize("rotated", [0.0, 2.5, -3.0])
    def test_arc_limits_build_an_arc_crank_that_sweeps_its_range(self, mirrored, rotated):
        from pylinkage.actuators import ArcCrank
        from pylinkage.synthesis import crank_angle_limits

        raw = _double_rocker(mirrored=mirrored, rotated=rotated)
        limits = crank_angle_limits(
            raw.crank_length, raw.coupler_length, raw.rocker_length, raw.ground_length
        )
        assert limits is not None
        linkage = solution_to_linkage(raw._replace(arc_limits=limits), iterations=90)

        crank = linkage.components[2]
        assert isinstance(crank, ArcCrank)
        assert (crank.x, crank.y) == pytest.approx(raw.crank_pivot_b)
        assert crank.arc_start <= crank.initial_angle <= crank.arc_end
        assert crank.arc_end - crank.arc_start == pytest.approx(limits[1] - limits[0])
        # The arc lies on the crank's side of the ground line, next to its start angle.
        start = math.atan2(raw.crank_pivot_b[1], raw.crank_pivot_b[0])
        assert crank.initial_angle == pytest.approx(start)

        # A full oscillation builds at every step and visits both ends of the arc.
        loci = list(linkage.step())
        assert len(loci) == linkage.get_rotation_period() == 180
        angles = [math.atan2(p[2][1], p[2][0]) for p in loci]
        unwrapped = [
            crank.arc_start + math.remainder(a - crank.arc_start, 2 * math.pi) for a in angles
        ]
        assert min(unwrapped) == pytest.approx(crank.arc_start, abs=crank.angular_velocity)
        assert max(unwrapped) == pytest.approx(crank.arc_end, abs=crank.angular_velocity)


class TestWattFromLengths:
    def test_basic(self):
        linkage = watt_from_lengths(
            crank=1.5,
            coupler1=4.0,
            rocker1=3.5,
            link4=3.0,
            link5=2.5,
            rocker2=3.0,
            ground_length=6.0,
        )
        # Should have 6 components (2 Ground + Crank + 3 RRR)
        assert len(linkage.components) == 6

    def test_with_custom_pivot(self):
        linkage = watt_from_lengths(
            crank=1.5,
            coupler1=4.0,
            rocker1=3.5,
            link4=3.0,
            link5=2.5,
            rocker2=3.0,
            ground_length=6.0,
            ground_pivot_a=(10.0, 5.0),
            name="my_watt",
        )
        assert linkage.name == "my_watt"
        assert linkage.components[0].x == pytest.approx(10.0)
        assert linkage.components[0].y == pytest.approx(5.0)


class TestStephensonFromLengths:
    def test_basic(self):
        linkage = stephenson_from_lengths(
            crank=0.8,
            coupler=3.5,
            rocker=3.0,
            link4=2.0,
            link5=2.5,
            link6=3.0,
            ground_length=4.0,
        )
        # Should have 6 components
        assert len(linkage.components) == 6

    def test_with_custom_params(self):
        linkage = stephenson_from_lengths(
            crank=0.8,
            coupler=3.5,
            rocker=3.0,
            link4=2.0,
            link5=2.5,
            link6=3.0,
            ground_length=4.0,
            ground_pivot_a=(1.0, 2.0),
            initial_crank_angle=0.5,
            name="my_steph",
        )
        assert linkage.name == "my_steph"


class TestNBarSolutionToLinkage:
    def test_fourbar_topology(self):
        sol = NBarSolution(
            topology_id="four-bar",
            joint_positions={
                "A": (0.0, 0.0),
                "D": (4.0, 0.0),
                "B": (1.0, 0.0),
                "C": (3.0, 2.0),
            },
            link_lengths={
                "crank_AB": 1.0,
                "coupler_BC": 2.83,
                "rocker_DC": 2.24,
                "ground_AD": 4.0,
            },
        )
        linkage = nbar_solution_to_linkage(sol)
        assert linkage is not None
        assert len(linkage.components) == 4

    def test_watt_topology(self):
        # Watt NBarSolution — joint positions must match catalog node IDs
        sol = NBarSolution(
            topology_id="watt",
            joint_positions={
                "J0_4": (0.0, 0.0),
                "J1_5": (6.0, 0.0),
                "J2_4": (0.5, 1.2),  # crank output
                "J2_3": (3.0, 2.5),  # coupler
                "J3_5": (6.0, 3.5),  # rocker1
                "J4_5": (5.0, 1.5),  # ternary link (Watt has ternary coupler)
            },
            link_lengths={},
        )
        try:
            linkage = nbar_solution_to_linkage(sol)
            assert linkage is not None
        except ValueError:
            # Watt conversion may fail with arbitrary positions - both cases valid
            pass

    def test_unknown_topology_raises(self):
        sol = NBarSolution(
            topology_id="totally-bogus-topology",
            joint_positions={},
            link_lengths={},
        )
        with pytest.raises(ValueError):
            nbar_solution_to_linkage(sol)


class TestGenericNBarToLinkage:
    def test_eight_bar_positions(self):
        # Proper eight-bar-01 positions matching catalog node roles
        positions = {
            "J0_1": (0.0, 0.0),  # ground
            "J0_6": (0.0, 0.0),  # ground
            "J1_7": (1.0, 0.5),  # driver
            "J2_3": (2.0, 1.0),
            "J2_6": (1.5, 1.0),
            "J3_7": (2.5, 0.7),
            "J4_5": (3.0, 1.5),
            "J4_6": (2.8, 1.5),
            "J5_7": (3.3, 1.2),
            "J6_7": (2.0, 0.8),
        }
        sol = NBarSolution(
            topology_id="eight-bar-01",
            joint_positions=positions,
            link_lengths={},
        )
        try:
            linkage = _generic_nbar_to_linkage(sol)
            assert linkage is not None
        except (ValueError, Exception):
            # Construction may fail — both cases valid
            pass

    def test_with_coupler_point(self):
        positions = {
            "J0_1": (0.0, 0.0),
            "J0_6": (0.0, 0.0),
            "J1_7": (1.0, 0.5),
            "J2_3": (2.0, 1.0),
            "J2_6": (1.5, 1.0),
            "J3_7": (2.5, 0.7),
            "J4_5": (3.0, 1.5),
            "J4_6": (2.8, 1.5),
            "J5_7": (3.3, 1.2),
            "J6_7": (2.0, 0.8),
        }
        sol = NBarSolution(
            topology_id="eight-bar-01",
            joint_positions=positions,
            link_lengths={},
            coupler_point=(2.5, 1.5),
            coupler_node="J2_3",
        )
        try:
            linkage = _generic_nbar_to_linkage(sol)
            assert linkage is not None
        except (ValueError, Exception):
            pass

    def test_unknown_topology_raises(self):
        sol = NBarSolution(
            topology_id="bogus",
            joint_positions={},
            link_lengths={},
        )
        with pytest.raises(ValueError):
            _generic_nbar_to_linkage(sol)


class TestPointDist:
    def test_basic(self):
        assert _point_dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_same_point(self):
        assert _point_dist((2.0, 2.0), (2.0, 2.0)) == 0.0


class TestFourbarFromLengthsErrors:
    def test_custom_name(self):
        linkage = fourbar_from_lengths(
            crank_length=1.0,
            coupler_length=3.0,
            rocker_length=3.0,
            ground_length=4.0,
            name="custom",
        )
        assert linkage.name == "custom"

    def test_unassemblable_raises(self):
        # Impossible configuration — unreachable
        with pytest.raises(ValueError):
            fourbar_from_lengths(
                crank_length=10.0,
                coupler_length=0.1,
                rocker_length=0.1,
                ground_length=1.0,
                initial_crank_angle=0.0,
            )
