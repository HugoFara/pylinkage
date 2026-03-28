"""Tests for DXF and STEP export functionality."""

import tempfile
import unittest
from pathlib import Path

import pylinkage as pl
from pylinkage.joints import Crank, Revolute


class FourBarLinkageTestCase(unittest.TestCase):
    """Base test case with a four-bar linkage fixture."""

    def setUp(self):
        """Set up a standard four-bar linkage for testing."""
        self.crank = Crank(
            0, 1,
            joint0=(0, 0),
            angle=0.31,
            distance=1,
            name="Crank"
        )
        self.pin = Revolute(
            3, 2,
            joint0=self.crank,
            joint1=(3, 0),
            distance0=3,
            distance1=1,
            name="Pin"
        )
        self.linkage = pl.Linkage(
            joints=[self.crank, self.pin],
            order=[self.crank, self.pin],
            name="TestFourBar"
        )
        # Pre-compute loci
        self.linkage.rebuild()
        self.loci = tuple(
            tuple(pos) for pos in self.linkage.step(iterations=10, dt=1)
        )


class TestDXFExportImport(unittest.TestCase):
    """Test DXF export module imports."""

    def test_import_from_visualizer(self):
        """Test that DXF export can be imported from visualizer."""
        from pylinkage.visualizer import plot_linkage_dxf, save_linkage_dxf
        self.assertTrue(callable(plot_linkage_dxf))
        self.assertTrue(callable(save_linkage_dxf))

    def test_import_direct(self):
        """Test direct import from dxf_export module."""
        from pylinkage.visualizer.dxf_export import (
            plot_linkage_dxf,
            save_linkage_dxf,
        )
        self.assertTrue(callable(plot_linkage_dxf))
        self.assertTrue(callable(save_linkage_dxf))


class TestSTEPExportImport(unittest.TestCase):
    """Test STEP export module imports."""

    def test_import_from_visualizer(self):
        """Test that STEP export can be imported from visualizer."""
        from pylinkage.visualizer import (
            LinkProfile,
            build_linkage_3d,
            save_linkage_step,
        )
        self.assertTrue(callable(build_linkage_3d))
        self.assertTrue(callable(save_linkage_step))
        # Test that dataclasses can be instantiated
        profile = LinkProfile(width=0.1, thickness=0.02)
        self.assertEqual(profile.width, 0.1)
        self.assertEqual(profile.thickness, 0.02)

    def test_import_direct(self):
        """Test direct import from step_export module."""
        from pylinkage.visualizer.step_export import (
            build_linkage_3d,
            save_linkage_step,
        )
        self.assertTrue(callable(build_linkage_3d))
        self.assertTrue(callable(save_linkage_step))


# Skip tests if ezdxf is not installed
try:
    import ezdxf  # noqa: F401
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


@unittest.skipUnless(HAS_EZDXF, "ezdxf not installed")
class TestDXFExport(FourBarLinkageTestCase):
    """Test DXF export functionality (requires ezdxf)."""

    def test_plot_linkage_dxf_returns_drawing(self):
        """Test that plot_linkage_dxf returns an ezdxf Drawing."""
        from pylinkage.visualizer import plot_linkage_dxf
        doc = plot_linkage_dxf(self.linkage, self.loci)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.__class__.__name__, "Drawing")

    def test_save_linkage_dxf_creates_file(self):
        """Test that save_linkage_dxf creates a file."""
        from pylinkage.visualizer import save_linkage_dxf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_linkage.dxf"
            save_linkage_dxf(self.linkage, path, loci=self.loci)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_dxf_has_layers(self):
        """Test that DXF has the expected layers."""
        from pylinkage.visualizer import plot_linkage_dxf
        doc = plot_linkage_dxf(self.linkage, self.loci)
        layer_names = [layer.dxf.name for layer in doc.layers]
        self.assertIn("LINKS", layer_names)
        self.assertIn("JOINTS", layer_names)
        self.assertIn("GROUND", layer_names)
        self.assertIn("CRANKS", layer_names)

    def test_dxf_custom_link_width(self):
        """Test DXF export with custom link width."""
        from pylinkage.visualizer import plot_linkage_dxf
        doc = plot_linkage_dxf(
            self.linkage, self.loci,
            link_width=0.5,
            joint_radius=0.2
        )
        self.assertIsNotNone(doc)

    def test_dxf_different_frame_index(self):
        """Test DXF export at different frame indices."""
        from pylinkage.visualizer import plot_linkage_dxf
        # Frame 0
        doc0 = plot_linkage_dxf(self.linkage, self.loci, frame_index=0)
        # Frame 5
        doc5 = plot_linkage_dxf(self.linkage, self.loci, frame_index=5)
        self.assertIsNotNone(doc0)
        self.assertIsNotNone(doc5)

    def test_dxf_invalid_frame_raises(self):
        """Test that invalid frame index raises ValueError."""
        from pylinkage.visualizer import plot_linkage_dxf
        with self.assertRaises(ValueError):
            plot_linkage_dxf(self.linkage, self.loci, frame_index=9999)

    def test_dxf_auto_computes_loci(self):
        """Test that DXF export auto-computes loci if not provided."""
        from pylinkage.visualizer import plot_linkage_dxf
        # Reset linkage and don't provide loci
        self.linkage.rebuild()
        doc = plot_linkage_dxf(self.linkage)
        self.assertIsNotNone(doc)


@unittest.skipIf(HAS_EZDXF, "ezdxf is installed, skipping missing dependency test")
class TestDXFMissingDependency(FourBarLinkageTestCase):
    """Test DXF export error handling when ezdxf is missing."""

    def test_missing_ezdxf_raises_import_error(self):
        """Test that missing ezdxf raises ImportError with helpful message."""
        from pylinkage.visualizer import plot_linkage_dxf
        with self.assertRaises(ImportError) as context:
            plot_linkage_dxf(self.linkage, self.loci)
        self.assertIn("ezdxf", str(context.exception))
        self.assertIn("pylinkage[cad]", str(context.exception))


# Skip tests if build123d is not installed
try:
    import build123d  # noqa: F401
    HAS_BUILD123D = True
except ImportError:
    HAS_BUILD123D = False


@unittest.skipUnless(HAS_BUILD123D, "build123d not installed")
class TestSTEPExport(FourBarLinkageTestCase):
    """Test STEP export functionality (requires build123d)."""

    def test_build_linkage_3d_returns_compound(self):
        """Test that build_linkage_3d returns a build123d Compound."""
        from pylinkage.visualizer import build_linkage_3d
        model = build_linkage_3d(self.linkage, self.loci)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, "Compound")

    def test_save_linkage_step_creates_file(self):
        """Test that save_linkage_step creates a file."""
        from pylinkage.visualizer import save_linkage_step
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_linkage.step"
            save_linkage_step(self.linkage, path, loci=self.loci)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_step_auto_scaling(self):
        """Test STEP export with auto-scaled dimensions."""
        from pylinkage.visualizer import build_linkage_3d
        model = build_linkage_3d(self.linkage, self.loci)
        self.assertIsNotNone(model)

    def test_step_custom_profiles(self):
        """Test STEP export with custom profiles."""
        from pylinkage.visualizer import JointProfile, LinkProfile, build_linkage_3d
        link_profile = LinkProfile(width=0.5, thickness=0.1, fillet_radius=0.0)
        joint_profile = JointProfile(radius=0.1, length=0.15)
        model = build_linkage_3d(
            self.linkage, self.loci,
            link_profile=link_profile,
            joint_profile=joint_profile
        )
        self.assertIsNotNone(model)

    def test_step_different_frame_index(self):
        """Test STEP export at different frame indices."""
        from pylinkage.visualizer import build_linkage_3d
        # Frame 0
        model0 = build_linkage_3d(self.linkage, self.loci, frame_index=0)
        # Frame 5
        model5 = build_linkage_3d(self.linkage, self.loci, frame_index=5)
        self.assertIsNotNone(model0)
        self.assertIsNotNone(model5)

    def test_step_without_pins(self):
        """Test STEP export without joint pins."""
        from pylinkage.visualizer import build_linkage_3d
        model = build_linkage_3d(
            self.linkage, self.loci,
            include_pins=False
        )
        self.assertIsNotNone(model)

    def test_step_invalid_frame_raises(self):
        """Test that invalid frame index raises ValueError."""
        from pylinkage.visualizer import build_linkage_3d
        with self.assertRaises(ValueError):
            build_linkage_3d(self.linkage, self.loci, frame_index=9999)

    def test_step_auto_computes_loci(self):
        """Test that STEP export auto-computes loci if not provided."""
        from pylinkage.visualizer import build_linkage_3d
        # Reset linkage and don't provide loci
        self.linkage.rebuild()
        model = build_linkage_3d(self.linkage)
        self.assertIsNotNone(model)


@unittest.skipIf(HAS_BUILD123D, "build123d is installed, skipping missing dependency test")
class TestSTEPMissingDependency(FourBarLinkageTestCase):
    """Test STEP export error handling when build123d is missing."""

    def test_missing_build123d_raises_import_error(self):
        """Test that missing build123d raises ImportError with helpful message."""
        from pylinkage.visualizer import build_linkage_3d
        with self.assertRaises(ImportError) as context:
            build_linkage_3d(self.linkage, self.loci)
        self.assertIn("build123d", str(context.exception))
        self.assertIn("pylinkage[cad]", str(context.exception))


class TestLinkProfileDataclass(unittest.TestCase):
    """Test LinkProfile dataclass."""

    def test_link_profile_defaults(self):
        """Test LinkProfile with defaults."""
        from pylinkage.visualizer import LinkProfile
        profile = LinkProfile(width=0.1, thickness=0.02)
        self.assertEqual(profile.width, 0.1)
        self.assertEqual(profile.thickness, 0.02)
        self.assertEqual(profile.fillet_radius, 0.0)

    def test_link_profile_custom_fillet(self):
        """Test LinkProfile with custom fillet."""
        from pylinkage.visualizer import LinkProfile
        profile = LinkProfile(width=0.1, thickness=0.02, fillet_radius=0.01)
        self.assertEqual(profile.fillet_radius, 0.01)


class TestJointProfileDataclass(unittest.TestCase):
    """Test JointProfile dataclass."""

    def test_joint_profile(self):
        """Test JointProfile instantiation."""
        from pylinkage.visualizer import JointProfile
        profile = JointProfile(radius=0.05, length=0.1)
        self.assertEqual(profile.radius, 0.05)
        self.assertEqual(profile.length, 0.1)


if __name__ == "__main__":
    unittest.main()
