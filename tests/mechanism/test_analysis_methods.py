"""Tests for the analyze_* bound methods on Mechanism and simulation.Linkage.

These methods are thin shims over :mod:`pylinkage.linkage.transmission`
and :mod:`pylinkage.linkage.sensitivity`. The tests verify the methods
delegate correctly and produce the same shape of result as the
corresponding free functions.
"""

from pylinkage.linkage.sensitivity import (
    analyze_sensitivity as free_analyze_sensitivity,
)
from pylinkage.linkage.transmission import (
    analyze_transmission as free_analyze_transmission,
)
from pylinkage.mechanism import fourbar


def _modern_fourbar():
    """Build a four-bar via the simulation.Linkage API."""
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
# Mechanism — bound methods
# ---------------------------------------------------------------------------


class TestMechanismAnalyzeTransmission:
    def test_method_matches_free_function(self) -> None:
        # Use independent mechanisms; the driver state isn't restored
        # between calls and otherwise the second invocation continues
        # from a different crank angle.
        m1 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        m2 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        bound = m1.analyze_transmission(iterations=20)
        free = free_analyze_transmission(m2, iterations=20)
        assert bound.min_angle == free.min_angle
        assert bound.max_angle == free.max_angle

    def test_acceptable_range_kwarg_accepted(self) -> None:
        """The kwarg must reach the underlying function without raising."""
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        result = m.analyze_transmission(
            iterations=20,
            acceptable_range=(50.0, 130.0),
        )
        # The current TransmissionAngleAnalysis hard-codes
        # acceptable_range; we just want to confirm the call succeeds and
        # the result is a TransmissionAngleAnalysis instance.
        assert hasattr(result, "min_angle")


class TestMechanismAnalyzeSensitivity:
    """Mechanism's bound shim mirrors the underlying free function.

    ``analyze_sensitivity`` itself probes constraints via
    ``joint.get_constraints()``, an interface that exists on the
    component API but not on Mechanism's joints (constraints live on
    Links). Until that wiring lands, the bound method on Mechanism is
    expected to surface the same ``AttributeError`` as the free function
    would.
    """

    def test_method_delegation_surface(self) -> None:
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        # Both call sites raise the same AttributeError for the same reason.
        import pytest

        with pytest.raises(AttributeError):
            m.analyze_sensitivity(iterations=5, include_transmission=False)
        with pytest.raises(AttributeError):
            free_analyze_sensitivity(m, iterations=5, include_transmission=False)


# ---------------------------------------------------------------------------
# simulation.Linkage — bound methods
# ---------------------------------------------------------------------------


class TestSimulationLinkageAnalyzeTransmission:
    def test_method_matches_free_function(self) -> None:
        l1 = _modern_fourbar()
        l2 = _modern_fourbar()
        bound = l1.analyze_transmission(iterations=20)
        free = free_analyze_transmission(l2, iterations=20)
        assert bound.min_angle == free.min_angle
        assert bound.max_angle == free.max_angle


class TestSimulationLinkageAnalyzeSensitivity:
    def test_returns_per_constraint_sensitivities(self) -> None:
        l1 = _modern_fourbar()
        l2 = _modern_fourbar()
        bound = l1.analyze_sensitivity(iterations=20, include_transmission=False)
        free = free_analyze_sensitivity(
            l2,
            iterations=20,
            include_transmission=False,
        )
        # Same constraint count → same sensitivity coverage
        assert set(bound.sensitivities) == set(free.sensitivities)
        assert bound.constraint_names == free.constraint_names
