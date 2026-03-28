"""Tests for dyad factory function."""

import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import (
    PPDyad,
    RRPDyad,
    RRRDyad,
    create_dyad,
    get_isomer_geometry,
    get_required_anchors,
    get_required_constraints,
)
from pylinkage.exceptions import UnbuildableError


class TestGetIsomerGeometry:
    """Test get_isomer_geometry function."""

    def test_rrr(self):
        assert get_isomer_geometry("RRR") == "circle_circle"

    def test_circle_line_isomers(self):
        """All circle-line isomers map to circle_line geometry."""
        circle_line_isomers = ["RR_T", "RRT_", "RT_R", "R_T_T", "RT_T_", "R_TT_", "RT__T"]
        for sig in circle_line_isomers:
            assert get_isomer_geometry(sig) == "circle_line", f"Failed for {sig}"

    def test_line_line_isomers(self):
        """All line-line isomers map to line_line geometry."""
        line_line_isomers = ["T_R_T", "T_RT_", "_TRT_"]
        for sig in line_line_isomers:
            assert get_isomer_geometry(sig) == "line_line", f"Failed for {sig}"

    def test_canonical_aliases(self):
        """Canonical aliases work."""
        assert get_isomer_geometry("RRP") == "circle_line"
        assert get_isomer_geometry("RPR") == "circle_line"
        assert get_isomer_geometry("PRR") == "circle_line"
        assert get_isomer_geometry("PP") == "line_line"

    def test_invalid_signature(self):
        with pytest.raises(ValueError, match="Unknown isomer signature"):
            get_isomer_geometry("XYZ")


class TestGetRequiredAnchors:
    """Test get_required_anchors function."""

    def test_rrr_anchors(self):
        anchors = get_required_anchors("RRR")
        assert anchors == ["anchor1", "anchor2"]

    def test_circle_line_anchors(self):
        anchors = get_required_anchors("RT_R")
        assert anchors == ["revolute", "line1", "line2"]

    def test_line_line_anchors(self):
        anchors = get_required_anchors("T_R_T")
        assert anchors == ["line1_p1", "line1_p2", "line2_p1", "line2_p2"]


class TestGetRequiredConstraints:
    """Test get_required_constraints function."""

    def test_rrr_constraints(self):
        constraints = get_required_constraints("RRR")
        assert constraints == ["distance1", "distance2"]

    def test_circle_line_constraints(self):
        constraints = get_required_constraints("RT_R")
        assert constraints == ["distance"]

    def test_line_line_constraints(self):
        constraints = get_required_constraints("T_R_T")
        assert constraints == []  # No distance constraints


class TestCreateDyadRRR:
    """Test create_dyad with RRR signature."""

    def test_create_rrr_basic(self):
        """Create basic RRR dyad."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")

        dyad = create_dyad(
            signature="RRR",
            anchors={"anchor1": O1, "anchor2": O2},
            constraints={"distance1": 1.5, "distance2": 1.5},
            name="test_rrr",
        )

        assert isinstance(dyad, RRRDyad)
        assert dyad.name == "test_rrr"
        assert dyad.distance1 == 1.5
        assert dyad.distance2 == 1.5

    def test_create_rrr_with_crank(self):
        """Create RRR dyad connected to crank output."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")
        crank = Crank(anchor=O1, radius=1.0)

        dyad = create_dyad(
            signature="RRR",
            anchors={"anchor1": crank.output, "anchor2": O2},
            constraints={"distance1": 2.0, "distance2": 1.5},
        )

        assert isinstance(dyad, RRRDyad)
        # Position should be computable
        assert dyad.x is not None
        assert dyad.y is not None

    def test_create_rrr_missing_anchor(self):
        """Missing anchor should raise ValueError."""
        O1 = Ground(0.0, 0.0, name="O1")

        with pytest.raises(ValueError, match="Missing required anchors"):
            create_dyad(
                signature="RRR",
                anchors={"anchor1": O1},  # Missing anchor2
                constraints={"distance1": 1.5, "distance2": 1.5},
            )

    def test_create_rrr_missing_constraint(self):
        """Missing constraint should raise ValueError."""
        O1 = Ground(0.0, 0.0, name="O1")
        O2 = Ground(2.0, 0.0, name="O2")

        with pytest.raises(ValueError, match="Missing required constraints"):
            create_dyad(
                signature="RRR",
                anchors={"anchor1": O1, "anchor2": O2},
                constraints={"distance1": 1.5},  # Missing distance2
            )


class TestCreateDyadCircleLine:
    """Test create_dyad with circle-line (RRP) isomers."""

    def test_create_rrp_basic(self):
        """Create basic RRP dyad."""
        O1 = Ground(0.0, 0.0, name="O1")
        L1 = Ground(0.0, 2.0, name="L1")
        L2 = Ground(3.0, 2.0, name="L2")

        dyad = create_dyad(
            signature="RT_R",  # Using isomer notation
            anchors={"revolute": O1, "line1": L1, "line2": L2},
            constraints={"distance": 2.5},
            name="slider",
        )

        assert isinstance(dyad, RRPDyad)
        assert dyad.name == "slider"
        assert dyad.distance == 2.5

    def test_create_all_circle_line_isomers(self):
        """All circle-line isomers should create RRPDyad."""
        O1 = Ground(0.0, 0.0)
        L1 = Ground(0.0, 2.0)
        L2 = Ground(3.0, 2.0)

        isomers = ["RR_T", "RRT_", "RT_R", "R_T_T", "RT_T_", "R_TT_", "RT__T", "RRP"]
        for sig in isomers:
            dyad = create_dyad(
                signature=sig,
                anchors={"revolute": O1, "line1": L1, "line2": L2},
                constraints={"distance": 2.5},
            )
            assert isinstance(dyad, RRPDyad), f"Failed for {sig}"

    def test_create_rrp_missing_anchor(self):
        """Missing anchor should raise ValueError."""
        O1 = Ground(0.0, 0.0)
        L1 = Ground(0.0, 2.0)

        with pytest.raises(ValueError, match="Missing required anchors"):
            create_dyad(
                signature="RT_R",
                anchors={"revolute": O1, "line1": L1},  # Missing line2
                constraints={"distance": 2.5},
            )


class TestCreateDyadLineLine:
    """Test create_dyad with line-line (PP) isomers."""

    def test_create_pp_basic(self):
        """Create basic PP dyad (line-line intersection)."""
        # Line 1: horizontal at y=1
        L1_A = Ground(0.0, 1.0, name="L1_A")
        L1_B = Ground(3.0, 1.0, name="L1_B")
        # Line 2: vertical at x=1.5
        L2_A = Ground(1.5, 0.0, name="L2_A")
        L2_B = Ground(1.5, 3.0, name="L2_B")

        dyad = create_dyad(
            signature="T_R_T",
            anchors={
                "line1_p1": L1_A,
                "line1_p2": L1_B,
                "line2_p1": L2_A,
                "line2_p2": L2_B,
            },
            name="double_slider",
        )

        assert isinstance(dyad, PPDyad)
        assert dyad.name == "double_slider"
        # Position should be at intersection (1.5, 1.0)
        assert dyad.x is not None
        assert dyad.y is not None
        assert abs(dyad.x - 1.5) < 1e-9
        assert abs(dyad.y - 1.0) < 1e-9

    def test_create_all_line_line_isomers(self):
        """All line-line isomers should create PPDyad."""
        L1_A = Ground(0.0, 1.0)
        L1_B = Ground(3.0, 1.0)
        L2_A = Ground(1.5, 0.0)
        L2_B = Ground(1.5, 3.0)

        isomers = ["T_R_T", "T_RT_", "_TRT_", "PP"]
        for sig in isomers:
            dyad = create_dyad(
                signature=sig,
                anchors={
                    "line1_p1": L1_A,
                    "line1_p2": L1_B,
                    "line2_p1": L2_A,
                    "line2_p2": L2_B,
                },
            )
            assert isinstance(dyad, PPDyad), f"Failed for {sig}"

    def test_create_pp_no_constraints_needed(self):
        """PP dyad should work without constraints."""
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(1.0, -1.0)
        L2_B = Ground(1.0, 1.0)

        # No constraints dict needed
        dyad = create_dyad(
            signature="T_R_T",
            anchors={
                "line1_p1": L1_A,
                "line1_p2": L1_B,
                "line2_p1": L2_A,
                "line2_p2": L2_B,
            },
        )

        assert isinstance(dyad, PPDyad)
        # Intersection at (1, 0)
        assert abs(dyad.x - 1.0) < 1e-9
        assert abs(dyad.y - 0.0) < 1e-9

    def test_create_pp_missing_anchor(self):
        """Missing anchor should raise ValueError."""
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(1.0, -1.0)

        with pytest.raises(ValueError, match="Missing required anchors"):
            create_dyad(
                signature="T_R_T",
                anchors={
                    "line1_p1": L1_A,
                    "line1_p2": L1_B,
                    "line2_p1": L2_A,
                    # Missing line2_p2
                },
            )


class TestPPDyad:
    """Direct tests for PPDyad class."""

    def test_pp_reload(self):
        """Test PPDyad reload updates position correctly."""
        # Moving line 1 anchors
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(1.0, -1.0)
        L2_B = Ground(1.0, 1.0)

        dyad = PPDyad(
            line1_anchor1=L1_A,
            line1_anchor2=L1_B,
            line2_anchor1=L2_A,
            line2_anchor2=L2_B,
        )

        # Initial intersection at (1, 0)
        assert abs(dyad.x - 1.0) < 1e-9
        assert abs(dyad.y - 0.0) < 1e-9

        # Move line 2 to x=2
        L2_A.x = 2.0
        L2_B.x = 2.0

        # Reload
        dyad.reload()

        # New intersection at (2, 0)
        assert abs(dyad.x - 2.0) < 1e-9
        assert abs(dyad.y - 0.0) < 1e-9

    def test_pp_unbuildable_parallel(self):
        """Parallel lines should raise UnbuildableError."""
        # Two parallel horizontal lines
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(0.0, 1.0)
        L2_B = Ground(2.0, 1.0)

        dyad = PPDyad(
            line1_anchor1=L1_A,
            line1_anchor2=L1_B,
            line2_anchor1=L2_A,
            line2_anchor2=L2_B,
        )

        # Initial position should be at centroid (parallel lines)
        # But reload should raise
        with pytest.raises(UnbuildableError):
            dyad.reload()

    def test_pp_constraints_empty(self):
        """PP dyad has no constraints."""
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(1.0, -1.0)
        L2_B = Ground(1.0, 1.0)

        dyad = PPDyad(
            line1_anchor1=L1_A,
            line1_anchor2=L1_B,
            line2_anchor1=L2_A,
            line2_anchor2=L2_B,
        )

        assert dyad.get_constraints() == ()

    def test_pp_anchors_property(self):
        """Test anchors property returns all four."""
        L1_A = Ground(0.0, 0.0)
        L1_B = Ground(2.0, 0.0)
        L2_A = Ground(1.0, -1.0)
        L2_B = Ground(1.0, 1.0)

        dyad = PPDyad(
            line1_anchor1=L1_A,
            line1_anchor2=L1_B,
            line2_anchor1=L2_A,
            line2_anchor2=L2_B,
        )

        anchors = dyad.anchors
        assert len(anchors) == 4
        assert L1_A in anchors
        assert L1_B in anchors
        assert L2_A in anchors
        assert L2_B in anchors


class TestSignatureCaseInsensitivity:
    """Test that signatures are case insensitive."""

    def test_lowercase_rrr(self):
        O1 = Ground(0.0, 0.0)
        O2 = Ground(2.0, 0.0)

        dyad = create_dyad(
            signature="rrr",  # lowercase
            anchors={"anchor1": O1, "anchor2": O2},
            constraints={"distance1": 1.5, "distance2": 1.5},
        )

        assert isinstance(dyad, RRRDyad)

    def test_mixed_case(self):
        O1 = Ground(0.0, 0.0)
        L1 = Ground(0.0, 2.0)
        L2 = Ground(3.0, 2.0)

        dyad = create_dyad(
            signature="Rt_R",  # mixed case
            anchors={"revolute": O1, "line1": L1, "line2": L2},
            constraints={"distance": 2.5},
        )

        assert isinstance(dyad, RRPDyad)
