"""Integration tests for full pylinkage workflows.

These tests verify that the complete workflow from linkage definition
through simulation and optimization works correctly.
"""

import math
import unittest

import pylinkage as pl
from pylinkage.exceptions import HypostaticError, UnbuildableError
from pylinkage.joints import Crank, Fixed, Linear, Revolute, Static
from pylinkage.linkage.analysis import bounding_box, movement_bounding_box
from pylinkage.optimization.utils import (
    generate_bounds,
    kinematic_maximization,
    kinematic_minimization,
)


class TestFourBarLinkageWorkflow(unittest.TestCase):
    """Test complete workflow with a four-bar linkage."""

    def setUp(self):
        """Set up a standard four-bar linkage."""
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
            name="FourBar"
        )

    def test_linkage_definition_and_rebuild(self):
        """Test that linkage can be defined and rebuilt."""
        self.linkage.rebuild()
        coords = self.linkage.get_coords()
        self.assertEqual(len(coords), 2)

    def test_linkage_step_produces_loci(self):
        """Test that stepping through linkage produces valid loci."""
        self.linkage.rebuild()
        loci = list(self.linkage.step(iterations=10))
        self.assertEqual(len(loci), 10)
        # Each step should have coordinates for both joints
        for step in loci:
            self.assertEqual(len(step), 2)

    def test_linkage_constraints_roundtrip(self):
        """Test that constraints can be get/set correctly."""
        self.linkage.rebuild()
        original_constraints = self.linkage.get_num_constraints()

        # Modify constraints
        new_constraints = [c * 1.1 if c else c for c in original_constraints]
        self.linkage.set_num_constraints(new_constraints)

        # Verify they were set
        retrieved = self.linkage.get_num_constraints()
        for orig, new, retrieved_val in zip(original_constraints, new_constraints, retrieved):
            if orig is not None:
                self.assertAlmostEqual(retrieved_val, new)

    def test_linkage_coords_roundtrip(self):
        """Test that coordinates can be get/set correctly."""
        self.linkage.rebuild()
        original_coords = self.linkage.get_coords()

        # Modify coordinates
        new_coords = [(x + 0.1 if x else x, y + 0.1 if y else y) for x, y in original_coords]
        self.linkage.set_coords(new_coords)

        # Verify they were set
        retrieved = self.linkage.get_coords()
        for (_orig_x, _orig_y), (new_x, new_y), (ret_x, ret_y) in zip(
            original_coords, new_coords, retrieved
        ):
            self.assertAlmostEqual(ret_x, new_x)
            self.assertAlmostEqual(ret_y, new_y)

    def test_rotation_period_calculation(self):
        """Test that rotation period is calculated correctly."""
        period = self.linkage.get_rotation_period()
        self.assertIsInstance(period, int)
        self.assertGreater(period, 0)

    def test_set_completely(self):
        """Test set_completely method."""
        constraints = self.linkage.get_num_constraints()
        positions = [(0, 1), (3, 2)]
        self.linkage.set_completely(constraints, positions)
        coords = self.linkage.get_coords()
        for (exp_x, exp_y), (act_x, act_y) in zip(positions, coords):
            self.assertAlmostEqual(exp_x, act_x)
            self.assertAlmostEqual(exp_y, act_y)


class TestLinkageAutoOrder(unittest.TestCase):
    """Test automatic solving order for linkages."""

    def test_auto_order_raises_hypostatic_error(self):
        """Test that automatic order raises error for unsolvable linkage.

        The automatic order algorithm cannot solve some linkages because
        it looks for joints whose parents are already in the solvable list.
        A Crank's joint0 is a Static (tuple converted), but it's not
        part of the joints list, so it won't be added to solvable.
        """
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        pin = Revolute(2, 1, joint0=crank, joint1=(2, 0), distance0=2, distance1=1)
        linkage = pl.Linkage(joints=[crank, pin])  # No order specified

        # Should raise HypostaticError when auto-order fails
        with self.assertRaises(HypostaticError):
            linkage.rebuild()


class TestLinkageAnalysis(unittest.TestCase):
    """Test linkage analysis functions."""

    def test_bounding_box_simple(self):
        """Test bounding box calculation with simple locus."""
        locus = [(0, 0), (1, 1), (2, 0), (1, -1)]
        bb = bounding_box(locus)
        y_min, x_max, y_max, x_min = bb
        self.assertEqual(y_min, -1)
        self.assertEqual(y_max, 1)
        self.assertEqual(x_min, 0)
        self.assertEqual(x_max, 2)

    def test_movement_bounding_box(self):
        """Test movement bounding box with multiple loci."""
        loci = [
            [(0, 0), (1, 1)],
            [(2, -1), (3, 2)],
        ]
        bb = movement_bounding_box(loci)
        y_min, x_max, y_max, x_min = bb
        self.assertEqual(y_min, -1)
        self.assertEqual(y_max, 2)
        self.assertEqual(x_min, 0)
        self.assertEqual(x_max, 3)


class TestOptimizationWorkflow(unittest.TestCase):
    """Test optimization workflow integration."""

    def setUp(self):
        """Set up linkage for optimization tests."""
        self.crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1, name="Crank")
        self.pin = Revolute(
            2, 1, joint0=self.crank, joint1=(2, 0),
            distance0=2, distance1=1, name="Pin"
        )
        self.linkage = pl.Linkage(
            joints=[self.crank, self.pin],
            order=[self.crank, self.pin],
            name="OptTest"
        )
        self.linkage.rebuild()

    def test_generate_bounds(self):
        """Test that bounds are generated correctly."""
        constraints = self.linkage.get_num_constraints()
        # Filter None values
        valid_constraints = [c for c in constraints if c is not None]
        bounds = generate_bounds(valid_constraints)
        self.assertEqual(len(bounds), 2)  # min and max bounds
        self.assertEqual(len(bounds[0]), len(valid_constraints))
        self.assertEqual(len(bounds[1]), len(valid_constraints))

    def test_kinematic_maximization_decorator(self):
        """Test the kinematic_maximization decorator."""
        @kinematic_maximization
        def dummy_fitness(linkage, params, init_pos, loci):
            # Return negative of total locus length
            return -sum(
                math.dist(loci[i][0], loci[i + 1][0])
                for i in range(len(loci) - 1)
            )

        result = dummy_fitness(
            self.linkage,
            self.linkage.get_num_constraints(),
            self.linkage.get_coords()
        )
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, -float('inf'))  # Should not be error penalty

    def test_kinematic_minimization_decorator(self):
        """Test the kinematic_minimization decorator."""
        @kinematic_minimization
        def dummy_fitness(linkage, params, init_pos, loci):
            # Return total locus length
            return sum(
                math.dist(loci[i][0], loci[i + 1][0])
                for i in range(len(loci) - 1)
            )

        result = dummy_fitness(
            self.linkage,
            self.linkage.get_num_constraints(),
            self.linkage.get_coords()
        )
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, float('inf'))  # Should not be error penalty

    def test_unbuildable_linkage_penalty(self):
        """Test that unbuildable configurations return penalty."""
        @kinematic_minimization
        def dummy_fitness(linkage, params, init_pos, loci):
            return 0

        # Create a completely new linkage with separate joints
        # to avoid mutation issues
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        pin = Revolute(
            2, 1, joint0=crank, joint1=(100, 0),  # Very far fixed point
            distance0=0.01, distance1=0.01,  # Very short distances
            name="Pin"
        )
        bad_linkage = pl.Linkage(joints=[crank, pin], order=[crank, pin])
        bad_linkage.rebuild()

        # These constraints will make it unbuildable
        bad_constraints = [1, 0.01, 0.01]  # Normal crank but tiny distances
        result = dummy_fitness(bad_linkage, bad_constraints, None)
        self.assertEqual(result, float('inf'))


class TestLinearJointWorkflow(unittest.TestCase):
    """Test workflow with Linear joints."""

    def test_linear_joint_definition(self):
        """Test that Linear joint can be defined."""
        anchor = Static(0, 0)
        line_start = Static(0, 2)
        line_end = Static(4, 2)
        linear = Linear(
            2, 2,
            joint0=anchor,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2,
            name="Slider"
        )
        # Get constraints
        constraints = linear.get_constraints()
        self.assertEqual(constraints, (2,))

    def test_linear_joint_set_constraints(self):
        """Test setting constraints on Linear joint."""
        anchor = Static(0, 0)
        line_start = Static(0, 2)
        line_end = Static(4, 2)
        linear = Linear(
            2, 2,
            joint0=anchor,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2,
        )
        linear.set_constraints(3.0)
        self.assertEqual(linear.revolute_radius, 3.0)


class TestFixedJointWorkflow(unittest.TestCase):
    """Test workflow with Fixed joints."""

    def test_fixed_joint_angles(self):
        """Test Fixed joint with different angles."""
        anchor1 = Revolute(0, 0)
        anchor2 = Revolute(2, 0)

        for angle in [0, math.pi / 4, math.pi / 2, math.pi]:
            fixed = Fixed(
                joint0=anchor1, joint1=anchor2,
                angle=angle, distance=1
            )
            fixed.reload()
            pos = fixed.coord()
            self.assertIsNotNone(pos[0])
            self.assertIsNotNone(pos[1])


class TestExceptionHandling(unittest.TestCase):
    """Test exception handling in various scenarios."""

    def test_unbuildable_revolute(self):
        """Test that UnbuildableError is raised for impossible configuration."""
        p1 = Revolute(0, 0)
        p2 = Revolute(10, 0)  # Very far apart
        p3 = Revolute(
            5, 5, joint0=p1, joint1=p2,
            distance0=1, distance1=1  # Too short to reach
        )
        with self.assertRaises(UnbuildableError):
            p3.reload()

    def test_crank_rotation(self):
        """Test Crank rotation mechanics."""
        crank = Crank(0, 1, joint0=(0, 0), angle=math.pi / 2, distance=1)
        crank.reload(dt=1)
        pos = crank.coord()
        # After pi/2 rotation from initial position
        self.assertIsNotNone(pos[0])
        self.assertIsNotNone(pos[1])


class TestHyperstaticityCalculation(unittest.TestCase):
    """Test hyperstaticity calculation."""

    def test_hyperstaticity_with_warning(self):
        """Test that hyperstaticity gives result with warning."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        pin = Revolute(2, 1, joint0=crank, joint1=(2, 0), distance0=2, distance1=1)
        linkage = pl.Linkage(
            joints=[crank, pin],
            order=[crank, pin],
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = linkage.hyperstaticity()
            self.assertIsInstance(result, int)
            self.assertEqual(len(w), 1)
            self.assertIn("experimental", str(w[0].message).lower())


if __name__ == '__main__':
    unittest.main()
