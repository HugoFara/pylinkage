"""Tests for the visualizer module."""

import unittest

import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for CI

import matplotlib.pyplot as plt

import pylinkage as pl
from pylinkage.joints import Crank, Fixed, Prismatic, Revolute, Static
from pylinkage.joints.revolute import Pivot
from pylinkage.visualizer.animated import (
    plot_kinematic_linkage,
    show_linkage,
    update_animated_plot,
)
from pylinkage.visualizer.core import COLOR_SWITCHER, _get_color
from pylinkage.visualizer.static import plot_static_linkage


class TestGetColor(unittest.TestCase):
    """Test the _get_color helper function."""

    def test_static_joint_color(self):
        """Test color for Static joint."""
        joint = Static(0, 0)
        self.assertEqual(_get_color(joint), 'k')

    def test_crank_joint_color(self):
        """Test color for Crank joint."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.1, distance=1)
        self.assertEqual(_get_color(crank), 'g')

    def test_revolute_joint_color(self):
        """Test color for Revolute joint."""
        joint1 = Revolute(0, 0)
        joint2 = Revolute(1, 0)
        revolute = Revolute(
            0.5, 0.5, joint0=joint1, joint1=joint2,
            distance0=1, distance1=1
        )
        self.assertEqual(_get_color(revolute), 'b')

    def test_fixed_joint_color(self):
        """Test color for Fixed joint."""
        joint1 = Revolute(0, 0)
        joint2 = Revolute(1, 0)
        fixed = Fixed(joint0=joint1, joint1=joint2, angle=0, distance=1)
        self.assertEqual(_get_color(fixed), 'r')

    def test_prismatic_joint_color(self):
        """Test color for Prismatic joint."""
        joint0 = Revolute(0, 0)
        joint1 = Revolute(0, 1)
        joint2 = Revolute(1, 1)
        prismatic = Prismatic(0.5, 0.5, joint0=joint0, joint1=joint1, joint2=joint2, revolute_radius=1)
        self.assertEqual(_get_color(prismatic), 'orange')

    def test_color_switcher_has_all_types(self):
        """Test that COLOR_SWITCHER has colors for all joint types."""
        expected_types = {Static, Crank, Fixed, Pivot, Revolute, Prismatic}
        self.assertEqual(set(COLOR_SWITCHER.keys()), expected_types)


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


class TestPlotStaticLinkage(FourBarLinkageTestCase):
    """Test the plot_static_linkage function."""

    def test_plot_static_linkage_basic(self):
        """Test that static plotting doesn't raise."""
        fig, ax = plt.subplots()
        try:
            plot_static_linkage(self.linkage, ax, self.loci)
            self.assertTrue(True)  # If we get here, no exception was raised
        finally:
            plt.close(fig)

    def test_plot_static_linkage_with_legend(self):
        """Test static plotting with legend enabled."""
        fig, ax = plt.subplots()
        try:
            plot_static_linkage(self.linkage, ax, self.loci, show_legend=True)
            self.assertEqual(ax.get_title(), "Static representation")
            self.assertEqual(ax.get_xlabel(), "x")
            self.assertEqual(ax.get_ylabel(), "y")
        finally:
            plt.close(fig)

    def test_plot_static_linkage_with_highlights(self):
        """Test static plotting with locus highlights."""
        fig, ax = plt.subplots()
        highlights = [[self.loci[0][0], self.loci[5][0]]]
        try:
            plot_static_linkage(
                self.linkage, ax, self.loci, locus_highlights=highlights
            )
            self.assertTrue(True)
        finally:
            plt.close(fig)

    def test_plot_static_linkage_sets_equal_aspect(self):
        """Test that plot sets equal aspect ratio."""
        fig, ax = plt.subplots()
        try:
            plot_static_linkage(self.linkage, ax, self.loci)
            # 'equal' aspect returns 1.0 from get_aspect()
            self.assertEqual(ax.get_aspect(), 1.0)
        finally:
            plt.close(fig)


class TestPlotKinematicLinkage(FourBarLinkageTestCase):
    """Test the plot_kinematic_linkage function."""

    def test_plot_kinematic_linkage_returns_animation(self):
        """Test that kinematic plotting returns an animation object."""
        fig, ax = plt.subplots()
        try:
            animation = plot_kinematic_linkage(
                self.linkage, fig, ax, self.loci, frames=5, interval=100
            )
            self.assertIsNotNone(animation)
        finally:
            plt.close(fig)

    def test_plot_kinematic_linkage_sets_title(self):
        """Test that kinematic plot sets proper title."""
        fig, ax = plt.subplots()
        try:
            plot_kinematic_linkage(self.linkage, fig, ax, self.loci, frames=5)
            self.assertEqual(ax.get_title(), "Animation")
        finally:
            plt.close(fig)


class TestUpdateAnimatedPlot(FourBarLinkageTestCase):
    """Test the update_animated_plot function."""

    def test_update_animated_plot_returns_images(self):
        """Test that update_animated_plot returns the images list."""
        fig, ax = plt.subplots()
        try:
            # Create initial images like plot_kinematic_linkage does
            images = []
            for joint in self.linkage.joints:
                for parent in (joint.joint0, joint.joint1):
                    if parent is not None:
                        images.append(ax.plot([], [], c='b')[0])

            result = update_animated_plot(self.linkage, 0, images, self.loci)
            self.assertEqual(result, images)
        finally:
            plt.close(fig)


class TestShowLinkage(FourBarLinkageTestCase):
    """Test the show_linkage function."""

    def test_show_linkage_returns_animation(self):
        """Test that show_linkage returns an animation and doesn't crash."""
        # Use pre-computed loci to avoid long computation
        animation = show_linkage(
            self.linkage,
            save=False,
            loci=self.loci,
            duration=0.1,  # Very short duration for testing
            fps=10
        )
        self.assertIsNotNone(animation)

    def test_show_linkage_with_custom_title(self):
        """Test show_linkage with custom title."""
        animation = show_linkage(
            self.linkage,
            save=False,
            loci=self.loci,
            title="CustomTitle",
            duration=0.1,
            fps=10
        )
        self.assertIsNotNone(animation)

    def test_show_linkage_computes_loci_if_none(self):
        """Test that show_linkage computes loci if not provided."""
        # Reset linkage
        self.linkage.rebuild()
        animation = show_linkage(
            self.linkage,
            save=False,
            points=5,  # Few points for fast test
            iteration_factor=1,
            duration=0.1,
            fps=10
        )
        self.assertIsNotNone(animation)


class TestPrismaticJointVisualization(unittest.TestCase):
    """Test visualization with Prismatic joints."""

    def setUp(self):
        """Set up a linkage with a Prismatic joint."""
        self.joint0 = Revolute(0, 0)
        self.joint1 = Static(0, 2)
        self.joint2 = Static(2, 2)
        self.prismatic = Prismatic(
            1, 1,
            joint0=self.joint0,
            joint1=self.joint1,
            joint2=self.joint2,
            revolute_radius=1.5,
            name="PrismaticJoint"
        )

    def test_prismatic_joint_color_in_plot(self):
        """Test that Prismatic joint gets correct color."""
        self.assertEqual(_get_color(self.prismatic), 'orange')


if __name__ == '__main__':
    unittest.main()
