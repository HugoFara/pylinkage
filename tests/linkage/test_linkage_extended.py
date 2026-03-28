"""Extended tests for linkage/linkage.py covering missing lines.

Targets: __set_solve_order__ (line 72), __find_solving_order__ (line 117),
rebuild with positions (line 140), step without solve_order (line 220),
indeterminacy (line 196), set_num_constraints non-flat (lines 505-506),
get_num_constraints non-flat (line 468), Simulation.iterations property (line 855),
compile (line 245), get_rotation_period (line 296), set_coords (line 358),
Simulation context manager (lines 451).
"""

import warnings

import pytest

import pylinkage as pl
from pylinkage.exceptions import UnderconstrainedError
from pylinkage.linkage.linkage import Simulation


def _make_fourbar():
    """Create a four-bar linkage."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
        rev = pl.Revolute(
            3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0, name="C"
        )
        linkage = pl.Linkage(
            joints=[crank, rev],
            order=[crank, rev],
            name="FourBar",
        )
    return linkage, crank, rev


class TestSetSolveOrder:
    """Test __set_solve_order__."""

    def test_set_solve_order(self):
        """Manually set solve order."""
        linkage, crank, rev = _make_fourbar()
        linkage.__set_solve_order__([rev, crank])
        assert linkage._solve_order == (rev, crank)


class TestFindSolvingOrder:
    """Test __find_solving_order__ automatic detection."""

    def test_auto_order_fourbar(self):
        """Auto-detect solving order for a four-bar."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            rev = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0, name="C"
            )
            linkage = pl.Linkage(joints=[crank, rev], name="FourBar")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            order = linkage.__find_solving_order__()
        # Crank and revolute should be in the order
        assert crank in order
        assert rev in order

    def test_underconstrained_raises(self):
        """Underconstrained linkage raises UnderconstrainedError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Two unconnected revolutes
            rev1 = pl.Revolute(1, 0, joint0=None, joint1=None, distance0=1, distance1=1, name="R1")
            rev2 = pl.Revolute(2, 0, joint0=None, joint1=None, distance0=1, distance1=1, name="R2")
            linkage = pl.Linkage(joints=[rev1, rev2], name="Bad")

        with pytest.raises(UnderconstrainedError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linkage.__find_solving_order__()


class TestRebuild:
    """Test rebuild method."""

    def test_rebuild_with_positions(self):
        """Rebuild sets initial positions."""
        linkage, crank, rev = _make_fourbar()
        new_pos = [(2.0, 0.0), (3.0, 1.0)]
        linkage.rebuild(pos=new_pos)
        coords = linkage.get_coords()
        assert coords[0] == (2.0, 0.0)
        assert coords[1] == (3.0, 1.0)

    def test_rebuild_without_positions(self):
        """Rebuild with no positions just ensures solve order exists."""
        linkage, _, _ = _make_fourbar()
        linkage.rebuild(pos=None)
        assert hasattr(linkage, "_solve_order")


class TestStepWithoutSolveOrder:
    """Test step when solve order hasn't been set."""

    def test_step_auto_finds_order(self):
        """step() should auto-find solving order if not set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            rev = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0, name="C"
            )
            linkage = pl.Linkage(joints=[crank, rev], name="FourBar")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # step should auto-detect solve order
            results = list(linkage.step(iterations=3))
        assert len(results) == 3


class TestIndeterminacy:
    """Test indeterminacy calculation."""

    def test_fourbar_indeterminacy(self):
        """Four-bar linkage should have 0 or specific indeterminacy."""
        linkage, _, _ = _make_fourbar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linkage.indeterminacy()
        assert isinstance(result, int)


class TestGetNumConstraints:
    """Test get_num_constraints with flat=False."""

    def test_non_flat(self):
        """get_num_constraints with flat=False returns tuples."""
        linkage, _, _ = _make_fourbar()
        result = linkage.get_num_constraints(flat=False)
        assert isinstance(result, list)
        # Each joint's constraints should be a tuple
        for item in result:
            assert isinstance(item, tuple)


class TestSetNumConstraints:
    """Test set_num_constraints."""

    def test_non_flat_constraints(self):
        """set_num_constraints with flat=False accepts tuples."""
        linkage, crank, rev = _make_fourbar()
        original = linkage.get_num_constraints(flat=False)
        # Set the same constraints back
        linkage.set_num_constraints(original, flat=False)
        new = linkage.get_num_constraints(flat=False)
        # The constraints should be the same
        for orig, new_ in zip(original, new, strict=False):
            for o, n in zip(orig, new_, strict=False):
                if o is not None and n is not None:
                    assert abs(o - n) < 1e-10


class TestSimulation:
    """Test the Simulation context manager."""

    def test_simulation_context_restores_state(self):
        """Simulation context manager restores initial state."""
        linkage, crank, rev = _make_fourbar()
        initial_coords = linkage.get_coords()

        with linkage.simulation(iterations=5) as sim:
            for _step, _coords in sim:
                pass  # just run through

        restored = linkage.get_coords()
        for init, rest in zip(initial_coords, restored, strict=False):
            assert init[0] == rest[0]
            assert init[1] == rest[1]

    def test_simulation_iterations_property(self):
        """Simulation.iterations returns correct value."""
        linkage, _, _ = _make_fourbar()
        sim = Simulation(linkage, iterations=42)
        assert sim.iterations == 42

    def test_simulation_default_iterations(self):
        """Simulation with None iterations uses linkage period."""
        linkage, _, _ = _make_fourbar()
        sim = Simulation(linkage, iterations=None)
        assert sim.iterations == linkage.get_rotation_period()

    def test_simulation_linkage_property(self):
        """Simulation.linkage returns the linkage."""
        linkage, _, _ = _make_fourbar()
        sim = Simulation(linkage)
        assert sim.linkage is linkage

    def test_simulation_exit_without_enter(self):
        """Calling __exit__ without __enter__ should not crash."""
        linkage, _, _ = _make_fourbar()
        sim = Simulation(linkage)
        # _initial_coords is None, so __exit__ should be a no-op
        sim.__exit__(None, None, None)


class TestGetRotationPeriod:
    """Test get_rotation_period."""

    def test_period_with_angle(self):
        """Period computed from crank angle."""
        linkage, _, _ = _make_fourbar()
        period = linkage.get_rotation_period()
        assert period > 0
        assert isinstance(period, int)

    def test_period_no_cranks(self):
        """Linkage with no cranks returns 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            static = pl.Static(0, 0, name="S")
            linkage = pl.Linkage(joints=[static], order=[static])
        assert linkage.get_rotation_period() == 1


class TestCompile:
    """Test the compile method."""

    def test_compile_sets_solver_data(self):
        """compile() should set _solver_data."""
        linkage, _, _ = _make_fourbar()
        assert linkage._solver_data is None
        linkage.compile()
        assert linkage._solver_data is not None

    def test_compile_without_solve_order(self):
        """compile() should auto-find solving order if not set."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            crank = pl.Crank(1, 0, joint0=(0, 0), angle=0.1, distance=1.0, name="B")
            rev = pl.Revolute(
                3, 1, joint0=crank, joint1=(4, 0), distance0=3.0, distance1=2.0, name="C"
            )
            linkage = pl.Linkage(joints=[crank, rev], name="FourBar")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            linkage.compile()
        assert linkage._solver_data is not None
