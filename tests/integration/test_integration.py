"""Integration tests for full pylinkage workflows.

These tests verify that the complete workflow from linkage definition
through simulation and optimization works correctly.
"""

import math
import unittest

import pylinkage as pl
from pylinkage.exceptions import UnbuildableError
from pylinkage.joints import Crank, Fixed, Prismatic, Revolute, Static
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

    def test_auto_order_success(self):
        """Test that automatic order succeeds when Static joints are in joints list.

        The automatic order algorithm requires parent joints to be in the joints list.
        When the anchor Static joint is explicitly included, auto-order works.
        """
        import warnings
        anchor0 = Static(0, 0, name="anchor0")
        anchor1 = Static(2, 0, name="anchor1")
        crank = Crank(0, 1, joint0=anchor0, angle=0.5, distance=1, name="crank")
        pin = Revolute(2, 1, joint0=crank, joint1=anchor1, distance0=2, distance1=1, name="pin")

        linkage = pl.Linkage(
            joints=[anchor0, anchor1, crank, pin],
            name="auto_order_test"
        )  # No order specified

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            linkage.rebuild()
            # Should warn about experimental feature
            self.assertEqual(len(w), 1)
            self.assertIn("experimental", str(w[0].message).lower())

        # Should be able to simulate
        positions = list(linkage.step(iterations=5))
        self.assertEqual(len(positions), 5)

    def test_auto_order_with_tuple_shortcuts(self):
        """Test that automatic order works with tuple shortcuts for anchors.

        The automatic order algorithm should detect Static joints created
        from tuple shortcuts (e.g., joint0=(0, 0)) even when they are not
        explicitly included in the joints list.
        """
        import warnings
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        pin = Revolute(2, 1, joint0=crank, joint1=(2, 0), distance0=2, distance1=1)
        linkage = pl.Linkage(joints=[crank, pin])  # No order specified

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            linkage.rebuild()
            # Should warn about experimental feature
            self.assertEqual(len(w), 1)
            self.assertIn("experimental", str(w[0].message).lower())

        # Should be able to simulate
        positions = list(linkage.step(iterations=5))
        self.assertEqual(len(positions), 5)


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


class TestPrismaticJointWorkflow(unittest.TestCase):
    """Test workflow with Prismatic joints."""

    def test_prismatic_joint_definition(self):
        """Test that Prismatic joint can be defined."""
        anchor = Static(0, 0)
        line_start = Static(0, 2)
        line_end = Static(4, 2)
        prismatic = Prismatic(
            2, 2,
            joint0=anchor,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2,
            name="Slider"
        )
        # Get constraints
        constraints = prismatic.get_constraints()
        self.assertEqual(constraints, (2,))

    def test_prismatic_joint_set_constraints(self):
        """Test setting constraints on Prismatic joint."""
        anchor = Static(0, 0)
        line_start = Static(0, 2)
        line_end = Static(4, 2)
        prismatic = Prismatic(
            2, 2,
            joint0=anchor,
            joint1=line_start,
            joint2=line_end,
            revolute_radius=2,
        )
        prismatic.set_constraints(3.0)
        self.assertEqual(prismatic.revolute_radius, 3.0)


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


class TestSerializationSupport(unittest.TestCase):
    """Test JSON serialization and deserialization of linkages."""

    def setUp(self):
        """Set up a standard four-bar linkage for testing."""
        self.crank = Crank(
            0, 1,
            joint0=(0, 0),
            angle=0.31,
            distance=1,
            name="crank"
        )
        self.pin = Revolute(
            3, 2,
            joint0=self.crank,
            joint1=(3, 0),
            distance0=3,
            distance1=1,
            name="pin"
        )
        self.linkage = pl.Linkage(
            joints=[self.crank, self.pin],
            order=[self.crank, self.pin],
            name="test_linkage"
        )

    def test_to_dict_basic(self):
        """Test basic serialization to dictionary."""
        data = self.linkage.to_dict()

        self.assertEqual(data["name"], "test_linkage")
        self.assertIn("joints", data)
        self.assertEqual(len(data["joints"]), 2)

    def test_from_dict_roundtrip(self):
        """Test roundtrip serialization through dictionary."""
        data = self.linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)

        self.assertEqual(loaded.name, self.linkage.name)
        self.assertEqual(len(loaded.joints), len(self.linkage.joints))

    def test_to_json_and_from_json(self):
        """Test saving and loading from JSON file."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.linkage.to_json(temp_path)
            loaded = pl.Linkage.from_json(temp_path)

            self.assertEqual(loaded.name, self.linkage.name)
            self.assertEqual(len(loaded.joints), len(self.linkage.joints))

            # Verify joint types are preserved
            self.assertIsInstance(loaded.joints[0], Crank)
            self.assertIsInstance(loaded.joints[1], Revolute)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_serialization_preserves_constraints(self):
        """Test that joint constraints are preserved through serialization."""
        data = self.linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)

        # Check Crank constraints
        original_crank = self.linkage.joints[0]
        loaded_crank = loaded.joints[0]
        assert isinstance(original_crank, Crank) and isinstance(loaded_crank, Crank)
        self.assertEqual(original_crank.r, loaded_crank.r)
        self.assertEqual(original_crank.angle, loaded_crank.angle)

        # Check Revolute constraints
        original_pin = self.linkage.joints[1]
        loaded_pin = loaded.joints[1]
        assert isinstance(original_pin, Revolute) and isinstance(loaded_pin, Revolute)
        self.assertEqual(original_pin.r0, loaded_pin.r0)
        self.assertEqual(original_pin.r1, loaded_pin.r1)

    def test_serialization_preserves_positions(self):
        """Test that joint positions are preserved through serialization."""
        data = self.linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)

        for orig, load in zip(self.linkage.joints, loaded.joints):
            self.assertEqual(orig.x, load.x)
            self.assertEqual(orig.y, load.y)

    def test_serialization_with_static_joints(self):
        """Test serialization with explicit Static joints."""
        anchor0 = Static(0, 0, name="anchor0")
        anchor1 = Static(3, 0, name="anchor1")
        crank = Crank(0, 1, joint0=anchor0, angle=0.5, distance=1, name="crank")
        pin = Revolute(2, 1, joint0=crank, joint1=anchor1, distance0=2, distance1=1, name="pin")
        linkage = pl.Linkage(
            joints=[anchor0, anchor1, crank, pin],
            order=[anchor0, anchor1, crank, pin],
            name="static_test"
        )

        data = linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)

        self.assertEqual(len(loaded.joints), 4)
        self.assertIsInstance(loaded.joints[0], Static)
        self.assertIsInstance(loaded.joints[1], Static)
        self.assertIsInstance(loaded.joints[2], Crank)
        self.assertIsInstance(loaded.joints[3], Revolute)

    def test_loaded_linkage_can_simulate(self):
        """Test that a loaded linkage can be simulated."""
        data = self.linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)
        loaded.rebuild()

        # Should be able to simulate
        positions = list(loaded.step(iterations=5))
        self.assertEqual(len(positions), 5)

    def test_serialization_with_fixed_joint(self):
        """Test serialization with Fixed joint type."""
        anchor0 = Static(0, 0, name="anchor0")
        anchor1 = Static(3, 0, name="anchor1")
        fixed = Fixed(
            1, 1,
            joint0=anchor0,
            joint1=anchor1,
            distance=1.5,
            angle=0.5,
            name="fixed_joint"
        )
        linkage = pl.Linkage(
            joints=[anchor0, anchor1, fixed],
            order=[anchor0, anchor1, fixed],
            name="fixed_test"
        )

        data = linkage.to_dict()
        loaded = pl.Linkage.from_dict(data)

        self.assertIsInstance(loaded.joints[2], Fixed)
        assert isinstance(loaded.joints[2], Fixed)
        self.assertEqual(loaded.joints[2].r, 1.5)
        self.assertEqual(loaded.joints[2].angle, 0.5)


class TestSimulationContextManager(unittest.TestCase):
    """Test the Simulation context manager API."""

    def setUp(self):
        """Set up a standard four-bar linkage for testing."""
        self.crank = Crank(
            0, 1,
            joint0=(0, 0),
            angle=0.31,
            distance=1,
            name="crank"
        )
        self.pin = Revolute(
            3, 2,
            joint0=self.crank,
            joint1=(3, 0),
            distance0=3,
            distance1=1,
            name="pin"
        )
        self.linkage = pl.Linkage(
            joints=[self.crank, self.pin],
            order=[self.crank, self.pin],
            name="test_linkage"
        )

    def test_simulation_basic_iteration(self):
        """Test basic iteration through simulation."""
        steps_collected = []
        coords_collected = []

        with self.linkage.simulation(iterations=5) as sim:
            for step, coords in sim:
                steps_collected.append(step)
                coords_collected.append(coords)

        self.assertEqual(steps_collected, [0, 1, 2, 3, 4])
        self.assertEqual(len(coords_collected), 5)
        # Each coords should have 2 joints
        for coords in coords_collected:
            self.assertEqual(len(coords), 2)

    def test_simulation_restores_state(self):
        """Test that simulation restores initial state after exiting."""
        initial_coords = self.linkage.get_coords()

        with self.linkage.simulation(iterations=10) as sim:
            for _ in sim:
                pass

        final_coords = self.linkage.get_coords()

        # State should be restored
        for (init_x, init_y), (final_x, final_y) in zip(initial_coords, final_coords):
            if init_x is not None and final_x is not None:
                self.assertAlmostEqual(init_x, final_x)
            if init_y is not None and final_y is not None:
                self.assertAlmostEqual(init_y, final_y)

    def test_simulation_restores_state_on_exception(self):
        """Test that state is restored even when exception occurs."""
        initial_coords = self.linkage.get_coords()

        try:
            with self.linkage.simulation(iterations=10) as sim:
                for step, _ in sim:
                    if step == 5:
                        raise ValueError("Test exception")
        except ValueError:
            pass

        final_coords = self.linkage.get_coords()

        # State should still be restored
        for (init_x, init_y), (final_x, final_y) in zip(initial_coords, final_coords):
            if init_x is not None and final_x is not None:
                self.assertAlmostEqual(init_x, final_x)
            if init_y is not None and final_y is not None:
                self.assertAlmostEqual(init_y, final_y)

    def test_simulation_default_iterations(self):
        """Test that default iterations uses rotation period."""
        expected_iterations = self.linkage.get_rotation_period()

        with self.linkage.simulation() as sim:
            self.assertEqual(sim.iterations, expected_iterations)

    def test_simulation_custom_dt(self):
        """Test simulation with custom dt value."""
        steps = []
        with self.linkage.simulation(iterations=3, dt=0.5) as sim:
            for step, _ in sim:
                steps.append(step)

        self.assertEqual(len(steps), 3)

    def test_simulation_linkage_property(self):
        """Test that simulation provides access to linkage."""
        with self.linkage.simulation(iterations=1) as sim:
            self.assertIs(sim.linkage, self.linkage)


class TestIndeterminacyCalculation(unittest.TestCase):
    """Test indeterminacy calculation.

    Uses the Gruebler-Kutzbach criterion for 2D planar mechanisms:
    DOF = 3 * (n - 1) - kinematic_pairs + mobilities

    Where:
    - n = number of bodies (including ground)
    - kinematic_pairs = sum of constraints from joints
    - mobilities = input degrees of freedom (e.g., motors)

    Indeterminacy = -DOF when DOF < 0 (over-constrained)
    """

    def test_indeterminacy_with_warning(self):
        """Test that indeterminacy gives result with warning."""
        crank = Crank(0, 1, joint0=(0, 0), angle=0.5, distance=1)
        pin = Revolute(2, 1, joint0=crank, joint1=(2, 0), distance0=2, distance1=1)
        linkage = pl.Linkage(
            joints=[crank, pin],
            order=[crank, pin],
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = linkage.indeterminacy()
            self.assertIsInstance(result, int)
            self.assertEqual(len(w), 1)
            self.assertIn("experimental", str(w[0].message).lower())

    def test_indeterminacy_four_bar(self):
        """Test indeterminacy for a standard four-bar linkage.

        A four-bar linkage should have DOF = 1 (or indeterminacy = -1 + input).
        With one crank as input, it should be kinematically determinate.
        """
        import warnings
        crank = Crank(0, 1, joint0=(0, 0), angle=0.31, distance=1)
        pin = Revolute(3, 2, joint0=crank, joint1=(3, 0), distance0=3, distance1=1)
        linkage = pl.Linkage(
            joints=[crank, pin],
            order=[crank, pin],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linkage.indeterminacy()

        # Four-bar should be kinematically determinate with one DOF
        self.assertIsInstance(result, int)

    def test_indeterminacy_with_static_only(self):
        """Test indeterminacy when only Static joints present."""
        import warnings
        anchor = Static(0, 0)
        linkage = pl.Linkage(joints=[anchor], order=[anchor])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linkage.indeterminacy()

        # Just a static point, fully constrained
        self.assertIsInstance(result, int)


if __name__ == '__main__':
    unittest.main()
