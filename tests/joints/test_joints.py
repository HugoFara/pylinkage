#!/usr/bin/env python3
"""
Created on Fri Apr 16 17:19:47 2021

@author: HugoFara
"""

import math
import unittest

from pylinkage import UnbuildableError
from pylinkage.exceptions import NotCompletelyDefinedError
from pylinkage.joints import Crank, Fixed, Linear, Prismatic, Revolute, Static
from pylinkage.joints.joint import joint_syntax_parser, _StaticBase
from pylinkage.joints.revolute import Pivot
from pylinkage.joints.static import Static as StaticFromModule


class TestJointSyntaxParser(unittest.TestCase):
    """Test the joint_syntax_parser function."""

    def test_none_input(self):
        """Test that None returns None."""
        self.assertIsNone(joint_syntax_parser(None))

    def test_joint_input(self):
        """Test that a Joint returns the same Joint."""
        joint = Revolute(1, 2)
        result = joint_syntax_parser(joint)
        self.assertIs(result, joint)

    def test_tuple_input(self):
        """Test that a tuple creates a static joint (_StaticBase to avoid deprecation)."""
        result = joint_syntax_parser((3.0, 4.0))
        # joint_syntax_parser uses _StaticBase to avoid deprecation warnings
        self.assertIsInstance(result, _StaticBase)
        self.assertEqual(result.coord(), (3.0, 4.0))


class TestStaticJointFromModule(unittest.TestCase):
    """Test Static joint from static.py module."""

    def test_creation(self):
        """Test Static joint creation from static.py."""
        static = StaticFromModule(1, 2, name="TestStaticModule")
        self.assertEqual(static.coord(), (1, 2))
        self.assertEqual(static.name, "TestStaticModule")

    def test_reload_does_nothing(self):
        """Test that reload doesn't change coordinates."""
        static = StaticFromModule(5, 6)
        static.reload(dt=2.0)
        self.assertEqual(static.coord(), (5, 6))

    def test_get_constraints_empty(self):
        """Test that get_constraints returns empty tuple."""
        static = StaticFromModule(1, 2)
        self.assertEqual(static.get_constraints(), ())

    def test_set_constraints_does_nothing(self):
        """Test that set_constraints doesn't raise or change anything."""
        static = StaticFromModule(1, 2)
        static.set_constraints(100, 200, 300)
        self.assertEqual(static.coord(), (1, 2))

    def test_set_anchor0(self):
        """Test set_anchor0 method."""
        static = StaticFromModule(0, 0)
        other = StaticFromModule(3, 4)
        static.set_anchor0(other)
        self.assertIs(static.joint0, other)

    def test_set_anchor1(self):
        """Test set_anchor1 method."""
        static = StaticFromModule(0, 0)
        other = StaticFromModule(7, 8)
        static.set_anchor1(other)
        self.assertIs(static.joint1, other)

    def test_set_anchor_with_tuple(self):
        """Test setting anchors with tuple coordinates."""
        static = StaticFromModule(0, 0)
        static.set_anchor0((9, 10))
        self.assertEqual(static.joint0.coord(), (9, 10))
        static.set_anchor1((11, 12))
        self.assertEqual(static.joint1.coord(), (11, 12))


class TestStaticJoint(unittest.TestCase):
    """Test Static joint."""

    def test_creation(self):
        """Test Static joint creation."""
        static = Static(1, 2, name="TestStatic")
        self.assertEqual(static.coord(), (1, 2))
        self.assertEqual(static.name, "TestStatic")

    def test_reload_does_nothing(self):
        """Test that reload doesn't change coordinates."""
        static = Static(1, 2)
        static.reload()
        self.assertEqual(static.coord(), (1, 2))

    def test_get_constraints_empty(self):
        """Test that get_constraints returns empty tuple."""
        static = Static(1, 2)
        self.assertEqual(static.get_constraints(), ())

    def test_set_constraints_does_nothing(self):
        """Test that set_constraints doesn't raise."""
        static = Static(1, 2)
        static.set_constraints(10, 20)  # Should not raise
        self.assertEqual(static.coord(), (1, 2))

    def test_set_anchor0(self):
        """Test set_anchor0 method."""
        static = Static(0, 0)
        other = Static(1, 1)
        static.set_anchor0(other)
        self.assertIs(static.joint0, other)

    def test_set_anchor1(self):
        """Test set_anchor1 method."""
        static = Static(0, 0)
        other = Static(1, 1)
        static.set_anchor1(other)
        self.assertIs(static.joint1, other)

    def test_set_anchor_with_tuple(self):
        """Test setting anchors with tuple coordinates."""
        static = Static(0, 0)
        static.set_anchor0((5, 5))
        # joint_syntax_parser uses _StaticBase to avoid deprecation warnings
        self.assertIsInstance(static.joint0, _StaticBase)
        self.assertEqual(static.joint0.coord(), (5, 5))


class TestJointBase(unittest.TestCase):
    """Test Joint base class methods."""

    def test_repr(self):
        """Test __repr__ method."""
        joint = Static(1, 2, name="MyJoint")
        repr_str = repr(joint)
        self.assertIn("Static", repr_str)
        self.assertIn("x=1", repr_str)
        self.assertIn("y=2", repr_str)
        self.assertIn("name=MyJoint", repr_str)

    def test_get_joints(self):
        """Test __get_joints__ method."""
        parent1 = Static(0, 0)
        parent2 = Static(1, 0)
        joint = Revolute(0.5, 0.5, joint0=parent1, joint1=parent2, distance0=1, distance1=1)
        joints = joint.__get_joints__()
        self.assertEqual(joints, (parent1, parent2))

    def test_set_coord_with_tuple(self):
        """Test set_coord with single tuple argument."""
        joint = Static(0, 0)
        joint.set_coord((3, 4))
        self.assertEqual(joint.coord(), (3, 4))

    def test_set_coord_with_two_args(self):
        """Test set_coord with two arguments."""
        joint = Static(0, 0)
        joint.set_coord(5, 6)
        self.assertEqual(joint.coord(), (5, 6))

    def test_set_coord_with_invalid_single_arg(self):
        """Test set_coord raises with invalid single argument."""
        joint = Static(0, 0)
        with self.assertRaises(TypeError):
            joint.set_coord(5)

    def test_default_name(self):
        """Test that default name is set from id."""
        joint = Static(0, 0)
        self.assertTrue(joint.name)  # Name should be set


class TestCrankJoint(unittest.TestCase):
    """Test Crank joint."""

    def test_crank_creation(self):
        """Test Crank joint creation."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1, name="TestCrank")
        self.assertEqual(crank.name, "TestCrank")

    def test_crank_reload(self):
        """Test Crank reload rotates the joint."""
        crank = Crank(1, 0, joint0=(0, 0), angle=math.pi / 2, distance=1)
        crank.reload(dt=1)
        x, y = crank.coord()
        # After pi/2 rotation, should be near (0, 1)
        self.assertAlmostEqual(y, 1, places=5)

    def test_crank_get_constraints(self):
        """Test Crank get_constraints returns distance."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=2)
        constraints = crank.get_constraints()
        self.assertEqual(constraints, (2,))

    def test_crank_set_constraints(self):
        """Test Crank set_constraints updates distance."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        crank.set_constraints(3)
        self.assertEqual(crank.get_constraints(), (3,))

    def test_crank_reload_missing_joint0(self):
        """Test Crank reload returns early when joint0 is missing."""
        crank = Crank(1, 0, angle=0.5, distance=1)
        # Should not raise, just return early without changing coords
        crank.reload()
        self.assertEqual(crank.coord(), (1, 0))

    def test_crank_reload_joint0_none_coords(self):
        """Test Crank reload raises when joint0 has None coordinates."""
        from pylinkage.exceptions import UnderconstrainedError
        j0 = Revolute()
        j0.x = None
        j0.y = None
        crank = Crank(1, 0, joint0=j0, angle=0.5, distance=1)
        with self.assertRaises(UnderconstrainedError):
            crank.reload()

    def test_crank_reload_missing_distance(self):
        """Test Crank reload raises when distance is missing."""
        from pylinkage.exceptions import UnderconstrainedError
        crank = Crank(1, 0, joint0=(0, 0), angle=0.5)
        with self.assertRaises(UnderconstrainedError):
            crank.reload()

    def test_crank_reload_missing_angle(self):
        """Test Crank reload raises when angle is missing."""
        from pylinkage.exceptions import UnderconstrainedError
        crank = Crank(1, 0, joint0=(0, 0), distance=1)
        with self.assertRaises(UnderconstrainedError):
            crank.reload()

    def test_crank_set_anchor0(self):
        """Test Crank set_anchor0 method."""
        crank = Crank(1, 0, angle=0.5, distance=1)
        anchor = Static(2, 2)
        crank.set_anchor0(anchor, distance=3.0)
        self.assertIs(crank.joint0, anchor)
        self.assertEqual(crank.r, 3.0)


class TestPrismaticJoint(unittest.TestCase):
    """Test Prismatic joint."""

    def setUp(self):
        """Set up test fixtures."""
        self.anchor = Static(0, 0)
        self.line_start = Static(0, 2)
        self.line_end = Static(4, 2)

    def test_prismatic_creation(self):
        """Test Prismatic joint creation."""
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            joint1=self.line_start,
            joint2=self.line_end,
            revolute_radius=2,
            name="TestPrismatic"
        )
        self.assertEqual(prismatic.name, "TestPrismatic")
        self.assertEqual(prismatic.revolute_radius, 2)

    def test_prismatic_get_constraints(self):
        """Test Prismatic get_constraints."""
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            joint1=self.line_start,
            joint2=self.line_end,
            revolute_radius=2.5
        )
        self.assertEqual(prismatic.get_constraints(), (2.5,))

    def test_prismatic_set_constraints(self):
        """Test Prismatic set_constraints."""
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            joint1=self.line_start,
            joint2=self.line_end,
            revolute_radius=2
        )
        prismatic.set_constraints(3.5)
        self.assertEqual(prismatic.revolute_radius, 3.5)

    def test_prismatic_reload(self):
        """Test Prismatic reload computes position."""
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            joint1=self.line_start,
            joint2=self.line_end,
            revolute_radius=2.5
        )
        prismatic.reload()
        x, y = prismatic.coord()
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        # Should be on line y=2
        self.assertAlmostEqual(y, 2)

    def test_prismatic_reload_missing_joint0(self):
        """Test Prismatic reload raises when joint0 missing."""
        prismatic = Prismatic(2, 2, revolute_radius=2)
        with self.assertRaises(NotCompletelyDefinedError):
            prismatic.reload()

    def test_prismatic_reload_missing_line_joints(self):
        """Test Prismatic reload raises when line joints missing."""
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            revolute_radius=2
        )
        with self.assertRaises(NotCompletelyDefinedError):
            prismatic.reload()

    def test_prismatic_reload_no_intersection(self):
        """Test Prismatic reload raises when no intersection possible."""
        # Circle too small to reach the line
        prismatic = Prismatic(
            2, 2,
            joint0=self.anchor,
            joint1=self.line_start,
            joint2=self.line_end,
            revolute_radius=0.5  # Too small
        )
        with self.assertRaises(UnbuildableError):
            prismatic.reload()

    def test_linear_alias_deprecated(self):
        """Test that Linear alias emits deprecation warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Linear(
                2, 2,
                joint0=self.anchor,
                joint1=self.line_start,
                joint2=self.line_end,
                revolute_radius=2,
            )
            # Now emits 2 warnings: Linear alias deprecation + Prismatic class deprecation
            self.assertEqual(len(w), 2)
            messages = [str(warning.message) for warning in w]
            self.assertTrue(any("Linear is deprecated" in msg for msg in messages))
            self.assertTrue(any("Prismatic is deprecated" in msg for msg in messages))


class TestPivot(unittest.TestCase):
    """Test Pivot Joint."""

    pivot1 = Revolute(0, 0)

    def test_buildable(self):
        """Upper intersect test."""
        pivot2 = Revolute(1, 0)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((.5, math.sqrt(.75)), pivot3.coord())

    def test_under_intersect(self):
        """Under intersect test."""
        pivot2 = Revolute(1, 0)
        pivot3 = Revolute(
            y=-1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((.5, -math.sqrt(.75)), pivot3.coord())

    def test_limit_intersect(self):
        """Test system almost breaking."""
        pivot2 = Revolute(2, 0)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((1, 0), pivot3.coord())

    def test_no_intersect(self):
        """Test system almost breaking."""
        pivot2 = Revolute(0, 3)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        with self.assertRaises(UnbuildableError):
            pivot3.reload()


class TestRevolute(unittest.TestCase):
    """Test Revolute Joint."""

    pivot1 = Revolute(0, 0)

    def test_buildable(self):
        """Upper intersect test."""
        pivot2 = Revolute(1, 0)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((.5, math.sqrt(.75)), pivot3.coord())

    def test_under_intersect(self):
        """Under intersect test."""
        pivot2 = Revolute(1, 0)
        pivot3 = Revolute(
            y=-1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((.5, -math.sqrt(.75)), pivot3.coord())

    def test_limit_intersect(self):
        """Test system almost breaking."""
        pivot2 = Revolute(2, 0)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        pivot3.reload()
        self.assertEqual((1, 0), pivot3.coord())

    def test_no_intersect(self):
        """Test system almost breaking."""
        pivot2 = Revolute(0, 3)
        pivot3 = Revolute(
            y=1, joint0=self.pivot1, joint1=pivot2,
            distance0=1, distance1=1
        )
        with self.assertRaises(UnbuildableError):
            pivot3.reload()

    def test_get_constraints(self):
        """Test get_constraints returns distances."""
        joint = Revolute(0, 0, distance0=2.5, distance1=3.5)
        self.assertEqual(joint.get_constraints(), (2.5, 3.5))

    def test_set_constraints(self):
        """Test set_constraints updates distances."""
        joint = Revolute(0, 0, distance0=1, distance1=1)
        joint.set_constraints(2.0, 3.0)
        self.assertEqual(joint.get_constraints(), (2.0, 3.0))

    def test_set_constraints_partial(self):
        """Test set_constraints with None keeps existing value."""
        joint = Revolute(0, 0, distance0=1, distance1=2)
        joint.set_constraints(5.0, None)
        self.assertEqual(joint.get_constraints(), (5.0, 2))

    def test_set_anchor0(self):
        """Test set_anchor0 method."""
        joint = Revolute(0, 0)
        anchor = Static(1, 1)
        joint.set_anchor0(anchor, distance=2.0)
        self.assertIs(joint.joint0, anchor)
        self.assertEqual(joint.r0, 2.0)

    def test_set_anchor1(self):
        """Test set_anchor1 method."""
        joint = Revolute(0, 0)
        anchor = Static(1, 1)
        joint.set_anchor1(anchor, distance=3.0)
        self.assertIs(joint.joint1, anchor)
        self.assertEqual(joint.r1, 3.0)

    def test_set_anchor_with_tuple(self):
        """Test set_anchor with tuple creates static joint."""
        joint = Revolute(0, 0)
        joint.set_anchor0((5, 5), distance=1.0)
        # joint_syntax_parser uses _StaticBase to avoid deprecation warnings
        self.assertIsInstance(joint.joint0, _StaticBase)
        self.assertEqual(joint.joint0.coord(), (5, 5))

    def test_circle_method(self):
        """Test circle method returns correct circle."""
        parent1 = Static(0, 0)
        parent2 = Static(2, 0)
        joint = Revolute(1, 1, joint0=parent1, joint1=parent2, distance0=1.5, distance1=2.5)
        self.assertEqual(joint.circle(parent1), (0, 0, 1.5))
        self.assertEqual(joint.circle(parent2), (2, 0, 2.5))

    def test_circle_method_invalid_joint(self):
        """Test circle raises ValueError for unknown joint."""
        parent1 = Static(0, 0)
        parent2 = Static(2, 0)
        other = Static(5, 5)
        joint = Revolute(1, 1, joint0=parent1, joint1=parent2, distance0=1.5, distance1=2.5)
        with self.assertRaises(ValueError):
            joint.circle(other)

    def test_get_joint_as_circle_invalid_index(self):
        """Test __get_joint_as_circle__ raises for invalid index."""
        parent1 = Static(0, 0)
        parent2 = Static(2, 0)
        joint = Revolute(1, 1, joint0=parent1, joint1=parent2, distance0=1.5, distance1=2.5)
        with self.assertRaises(ValueError):
            joint.__get_joint_as_circle__(2)

    def test_reload_no_joint0(self):
        """Test reload with no joint0 does nothing."""
        joint = Revolute(1, 2)
        joint.reload()
        self.assertEqual(joint.coord(), (1, 2))

    def test_reload_one_constraint_warning(self):
        """Test reload with only one valid constraint emits warning."""
        import warnings
        parent = Static(0, 0)
        joint = Revolute(1, 1, joint0=parent, distance0=1.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            joint.reload()
            self.assertEqual(len(w), 1)
            self.assertIn("Only one constraint", str(w[0].message))

    def test_reload_coincident_circles_warning(self):
        """Test reload with coincident circles emits warning."""
        import warnings
        parent1 = Static(0, 0)
        parent2 = Static(0, 0)  # Same position
        joint = Revolute(1, 0, joint0=parent1, joint1=parent2, distance0=1, distance1=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            joint.reload()
            self.assertEqual(len(w), 1)
            self.assertIn("infinite number", str(w[0].message))


class TestFixed(unittest.TestCase):
    """Test Fixed_Joint."""

    pivot1 = Revolute(0, 0)

    def test_pos(self):
        """Test Fixed_Joint positioning."""
        pivot2 = Revolute(1, 0)
        fixed = Fixed(
            joint0=self.pivot1, joint1=pivot2, angle=0, distance=1
        )
        fixed.reload()
        self.assertEqual((1, 0), fixed.coord())

    def test_get_constraints(self):
        """Test get_constraints returns distance and angle."""
        fixed = Fixed(distance=2.5, angle=0.7)
        self.assertEqual(fixed.get_constraints(), (2.5, 0.7))

    def test_set_constraints(self):
        """Test set_constraints updates distance and angle."""
        fixed = Fixed(distance=1, angle=0.5)
        fixed.set_constraints(distance=3.0, angle=1.2)
        self.assertEqual(fixed.get_constraints(), (3.0, 1.2))

    def test_set_constraints_partial(self):
        """Test set_constraints with partial update."""
        fixed = Fixed(distance=1, angle=0.5)
        fixed.set_constraints(distance=4.0)
        self.assertEqual(fixed.r, 4.0)
        self.assertEqual(fixed.angle, 0.5)

    def test_set_anchor0(self):
        """Test set_anchor0 method."""
        fixed = Fixed()
        anchor = Static(5, 5)
        fixed.set_anchor0(anchor, distance=2.0, angle=0.3)
        self.assertIs(fixed.joint0, anchor)
        self.assertEqual(fixed.r, 2.0)
        self.assertEqual(fixed.angle, 0.3)

    def test_set_anchor1(self):
        """Test set_anchor1 method."""
        fixed = Fixed()
        anchor = Static(7, 7)
        fixed.set_anchor1(anchor)
        self.assertIs(fixed.joint1, anchor)

    def test_reload_missing_joint0(self):
        """Test reload raises when joint0 is missing."""
        from pylinkage.exceptions import UnderconstrainedError
        fixed = Fixed(joint1=Static(1, 0), distance=1, angle=0)
        with self.assertRaises(UnderconstrainedError):
            fixed.reload()

    def test_reload_missing_joint1(self):
        """Test reload raises when joint1 is missing."""
        from pylinkage.exceptions import UnderconstrainedError
        fixed = Fixed(joint0=Static(0, 0), distance=1, angle=0)
        with self.assertRaises(UnderconstrainedError):
            fixed.reload()

    def test_reload_missing_constraints(self):
        """Test reload raises when constraints are missing."""
        from pylinkage.exceptions import UnderconstrainedError
        fixed = Fixed(joint0=Static(0, 0), joint1=Static(1, 0))
        with self.assertRaises(UnderconstrainedError):
            fixed.reload()

    def test_reload_joint0_none_coords(self):
        """Test reload raises when joint0 has None coordinates."""
        from pylinkage.exceptions import UnderconstrainedError
        j0 = Revolute()  # Has None coordinates
        j0.x = None
        j0.y = None
        fixed = Fixed(joint0=j0, joint1=Static(1, 0), distance=1, angle=0)
        with self.assertRaises(UnderconstrainedError):
            fixed.reload()

    def test_reload_joint1_none_coords(self):
        """Test reload raises when joint1 has None coordinates."""
        from pylinkage.exceptions import UnderconstrainedError
        j1 = Revolute()
        j1.x = None
        j1.y = None
        fixed = Fixed(joint0=Static(0, 0), joint1=j1, distance=1, angle=0)
        with self.assertRaises(UnderconstrainedError):
            fixed.reload()


class TestPivotDeprecation(unittest.TestCase):
    """Test Pivot class deprecation."""

    def test_pivot_deprecation_warning(self):
        """Test that Pivot emits deprecation warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Pivot(0, 0)
            # Now emits 2 warnings: Pivot alias deprecation + Revolute class deprecation
            self.assertEqual(len(w), 2)
            messages = [str(warning.message) for warning in w]
            self.assertTrue(any("Pivot" in msg and "deprecated" in msg for msg in messages))
            self.assertTrue(any("Revolute" in msg and "deprecated" in msg for msg in messages))

    def test_pivot_inherits_revolute(self):
        """Test that Pivot still works as Revolute."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pivot = Pivot(1, 2, distance0=1, distance1=2)
            self.assertEqual(pivot.coord(), (1, 2))
            self.assertEqual(pivot.get_constraints(), (1, 2))


if __name__ == '__main__':
    unittest.main()
