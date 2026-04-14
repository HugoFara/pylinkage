"""Tests for single-shot transmission/stroke accessors + Mechanism.rebuild.

Covers the small convenience methods that round out Mechanism and
simulation.Linkage parity with the legacy Linkage class.
"""

import pytest

from pylinkage.linkage.transmission import transmission_angle_at_position
from pylinkage.mechanism import fourbar


def _modern_fourbar():
    from pylinkage.actuators import Crank
    from pylinkage.components import Ground
    from pylinkage.dyads import RRRDyad
    from pylinkage.simulation import Linkage

    A = Ground(0.0, 0.0, name="A")
    D = Ground(4.0, 0.0, name="D")
    crank = Crank(anchor=A, radius=1.0, angular_velocity=0.1, name="crank")
    pin = RRRDyad(
        anchor1=crank.output,
        anchor2=D,
        distance1=3.0,
        distance2=3.0,
        name="C",
    )
    return Linkage([A, D, crank, pin], name="modern-fourbar")


# ---------------------------------------------------------------------------
# transmission_angle — single-shot
# ---------------------------------------------------------------------------


class TestTransmissionAngle:
    def test_mechanism_matches_free_function(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert m.transmission_angle() == pytest.approx(
            transmission_angle_at_position(m),
        )

    def test_simulation_linkage_matches_free_function(self) -> None:
        linkage = _modern_fourbar()
        assert linkage.transmission_angle() == pytest.approx(
            transmission_angle_at_position(linkage),
        )

    def test_in_degree_range(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        angle = m.transmission_angle()
        assert 0.0 <= angle <= 180.0


# ---------------------------------------------------------------------------
# stroke_position — single-shot (no prismatic joint → ValueError)
# ---------------------------------------------------------------------------


class TestStrokePosition:
    def test_mechanism_without_prismatic_raises(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        with pytest.raises(ValueError):
            m.stroke_position()

    def test_simulation_linkage_without_prismatic_raises(self) -> None:
        linkage = _modern_fourbar()
        with pytest.raises(ValueError):
            linkage.stroke_position()


# ---------------------------------------------------------------------------
# Mechanism.rebuild
# ---------------------------------------------------------------------------


class TestMechanismRebuild:
    def test_no_args_invalidates_solver_cache(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.compile()
        assert m._solver_data is not None
        m.rebuild()
        assert m._solver_data is None

    def test_applies_initial_positions(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        original = m.get_joint_positions()

        # Displace every joint by (0.25, 0.25)
        shifted = [(x + 0.25, y + 0.25) for (x, y) in original]
        m.rebuild(initial_positions=shifted)

        restored = m.get_joint_positions()
        for (rx, ry), (sx, sy) in zip(restored, shifted, strict=True):
            assert rx == pytest.approx(sx)
            assert ry == pytest.approx(sy)

    def test_rebuild_after_step_recompiles(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m.step_fast(iterations=5)
        first = m._solver_data
        m.rebuild()
        assert m._solver_data is None
        m.step_fast(iterations=5)
        assert m._solver_data is not None
        assert m._solver_data is not first
