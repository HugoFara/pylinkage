"""Tests for the fourbar / slider-crank factory functions."""

import math

import pytest

from pylinkage.exceptions import UnbuildableError
from pylinkage.mechanism import Mechanism, fourbar, slider_crank


class TestFourbar:
    def test_returns_mechanism(self):
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert isinstance(m, Mechanism)

    def test_default_geometry(self):
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        assert len(m.joints) == 4
        assert len(m.links) == 4

    def test_simulates_one_full_cycle(self):
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0,
                    omega=math.tau / 50)
        loci = list(m.step())
        # ~ tau/omega frames (off-by-one allowed at the period boundary)
        assert 48 <= len(loci) <= 51

    def test_ground_pivots_are_fixed(self):
        m = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0)
        loci = list(m.step())
        # Find the ground positions A=(0,0) and D=(ground,0). They must be
        # constant across all frames.
        for joint_idx in range(len(m.joints)):
            xs = {round(frame[joint_idx][0], 6) for frame in loci}
            ys = {round(frame[joint_idx][1], 6) for frame in loci}
            if (xs, ys) == ({0.0}, {0.0}) or (xs, ys) == ({4.0}, {0.0}):
                break
        else:
            pytest.fail("Ground pivots A=(0,0) and D=(4,0) not found")

    def test_branch_selection(self):
        # Both branches must produce buildable mechanisms with mirrored
        # coupler-rocker positions.
        m0 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0, branch=0)
        m1 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0, branch=1)
        f0 = list(m0.step())[0]
        f1 = list(m1.step())[0]
        # Different branches => different y for the coupler-rocker joint
        ys0 = {round(p[1], 6) for p in f0}
        ys1 = {round(p[1], 6) for p in f1}
        assert ys0 != ys1

    def test_unbuildable_raises(self):
        # Ground >> crank+coupler+rocker => cannot close
        with pytest.raises(UnbuildableError):
            fourbar(crank=1.0, coupler=1.0, rocker=1.0, ground=10.0)

    def test_initial_angle_changes_first_frame(self):
        m0 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0,
                     initial_angle=0.0)
        m1 = fourbar(crank=1.0, coupler=3.0, rocker=3.0, ground=4.0,
                     initial_angle=math.pi / 2)
        f0 = list(m0.step())[0]
        f1 = list(m1.step())[0]
        assert f0 != f1


class TestSliderCrank:
    def test_returns_mechanism(self):
        m = slider_crank(crank=1.0, rod=3.0)
        assert isinstance(m, Mechanism)

    def test_simulates(self):
        m = slider_crank(crank=1.0, rod=3.0, omega=math.tau / 60)
        loci = list(m.step())
        assert 58 <= len(loci) <= 61

    def test_piston_stays_on_axis(self):
        m = slider_crank(crank=1.0, rod=3.0)
        loci = list(m.step())
        # Find the moving joint that stays on the x-axis (the piston).
        # The ground O also has y=0 but x=0 always.
        for j in range(len(m.joints)):
            ys = [frame[j][1] for frame in loci if frame[j][1] is not None]
            xs = [frame[j][0] for frame in loci if frame[j][0] is not None]
            if ys and all(abs(y) < 1e-6 for y in ys) and max(xs) - min(xs) > 0.5:
                return
        pytest.fail("No moving joint sliding along the x-axis was found")
