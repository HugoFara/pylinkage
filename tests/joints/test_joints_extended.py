"""Extended tests for revolute and prismatic joints -- targeting uncovered lines."""

import math
import unittest
import warnings

from pylinkage import UnbuildableError
from pylinkage.exceptions import NotCompletelyDefinedError
from pylinkage.joints import Prismatic, Revolute, Static


class TestRevoluteGetJointAsCircle(unittest.TestCase):
    """Test __get_joint_as_circle__ method (lines 97-105)."""

    def test_get_joint_as_circle_index_0(self):
        """Test __get_joint_as_circle__(0) returns correct circle (lines 97-100)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = Static(1, 2)
            parent1 = Static(3, 4)
            joint = Revolute(
                2, 3,
                joint0=parent0,
                joint1=parent1,
                distance0=5.0,
                distance1=6.0,
            )
        circle = joint.__get_joint_as_circle__(0)
        self.assertEqual(circle, (1, 2, 5.0))

    def test_get_joint_as_circle_index_1(self):
        """Test __get_joint_as_circle__(1) returns correct circle (lines 102-105)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = Static(1, 2)
            parent1 = Static(3, 4)
            joint = Revolute(
                2, 3,
                joint0=parent0,
                joint1=parent1,
                distance0=5.0,
                distance1=6.0,
            )
        circle = joint.__get_joint_as_circle__(1)
        self.assertEqual(circle, (3, 4, 6.0))


class TestRevoluteReloadEdgeCases(unittest.TestCase):
    """Test Revolute.reload edge cases (lines 137, 148-152, 156-157)."""

    def test_reload_missing_distances_warning(self):
        """Test reload with missing distance constraints warns (lines 148-152)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = Static(0, 0)
            parent1 = Static(2, 0)
            joint = Revolute(1, 0, joint0=parent0, joint1=parent1)
            # r0 and r1 are None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            joint.reload()
            # Should warn about missing constraints
            messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("missing distance" in msg.lower() or "constraint" in msg.lower() for msg in messages)
            )

    def test_reload_with_none_xy_initializes(self):
        """Test reload initializes x,y from joint0 when None (lines 156-157)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = Static(5, 3)
            parent1 = Static(8, 3)
            joint = Revolute(
                joint0=parent0,
                joint1=parent1,
                distance0=2.0,
                distance1=2.0,
            )
            joint.x = None
            joint.y = None
        # Should not raise -- initializes from joint0
        joint.reload()
        self.assertIsNotNone(joint.x)
        self.assertIsNotNone(joint.y)

    def test_reload_one_parent_none_coords(self):
        """Test reload with one parent having None coords gives warning (line 137)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            parent0 = Static(0, 0)
            parent1 = Revolute()
            parent1.x = None
            parent1.y = None
            joint = Revolute(
                1, 0,
                joint0=parent0,
                joint1=parent1,
                distance0=1.0,
                distance1=1.0,
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            joint.reload()
            messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("one constraint" in msg.lower() for msg in messages)
            )


class TestPrismaticReloadEdgeCases(unittest.TestCase):
    """Test Prismatic.reload edge cases (lines 96, 98, 100, 102, 106-107)."""

    def test_reload_joint0_none_coords(self):
        """Test reload when joint0 has None coordinates (line 96)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            j0 = Revolute()
            j0.x = None
            j0.y = None
            line_start = Static(0, 2)
            line_end = Static(4, 2)
            prismatic = Prismatic(
                2, 2,
                joint0=j0,
                joint1=line_start,
                joint2=line_end,
                revolute_radius=2.5,
            )
        with self.assertRaises(UnbuildableError):
            prismatic.reload()

    def test_reload_missing_revolute_radius(self):
        """Test reload when revolute_radius is None (line 98)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor = Static(0, 0)
            line_start = Static(0, 2)
            line_end = Static(4, 2)
            prismatic = Prismatic(
                2, 2,
                joint0=anchor,
                joint1=line_start,
                joint2=line_end,
            )
            # revolute_radius is None
        with self.assertRaises(UnbuildableError):
            prismatic.reload()

    def test_reload_joint1_none_coords(self):
        """Test reload when joint1 has None coordinates (line 100)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor = Static(0, 0)
            j1 = Revolute()
            j1.x = None
            j1.y = None
            line_end = Static(4, 2)
            prismatic = Prismatic(
                2, 2,
                joint0=anchor,
                joint1=j1,
                joint2=line_end,
                revolute_radius=2.5,
            )
        with self.assertRaises(UnbuildableError):
            prismatic.reload()

    def test_reload_joint2_none_coords(self):
        """Test reload when joint2 has None coordinates (line 102)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor = Static(0, 0)
            line_start = Static(0, 2)
            j2 = Revolute()
            j2.x = None
            j2.y = None
            prismatic = Prismatic(
                2, 2,
                joint0=anchor,
                joint1=line_start,
                joint2=j2,
                revolute_radius=2.5,
            )
        with self.assertRaises(UnbuildableError):
            prismatic.reload()

    def test_reload_initializes_xy_from_joint0(self):
        """Test that reload initializes x,y from joint0 when None (lines 106-107)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor = Static(0, 0)
            line_start = Static(0, 2)
            line_end = Static(4, 2)
            prismatic = Prismatic(
                joint0=anchor,
                joint1=line_start,
                joint2=line_end,
                revolute_radius=2.5,
            )
            prismatic.x = None
            prismatic.y = None
        # Should not raise, initializes from joint0
        prismatic.reload()
        self.assertIsNotNone(prismatic.x)
        self.assertIsNotNone(prismatic.y)


if __name__ == "__main__":
    unittest.main()
