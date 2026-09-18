"""Tests for symbolic solver functions."""

import math

import numpy as np
import pytest
import sympy as sp

from pylinkage.symbolic import (
    SymbolicLinkage,
    SymCrank,
    SymRevolute,
    SymStatic,
    check_buildability,
    compute_trajectory_numeric,
    create_trajectory_functions,
    eliminate_theta,
    solve_linkage_symbolically,
    theta,
)


class TestSolveLinkageSymbolically:
    """Tests for solve_linkage_symbolically function."""

    def test_basic_solve(self):
        """Test basic symbolic solving."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])

        trajectories = solve_linkage_symbolically(linkage)

        assert "A" in trajectories
        assert "B" in trajectories

    def test_output_joints_filter(self):
        """Test filtering output joints."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])

        trajectories = solve_linkage_symbolically(linkage, output_joints=["B"])

        assert "B" in trajectories
        assert "A" not in trajectories

    def test_simplify_option(self):
        """Test that simplify option works."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])

        # Both should work
        traj1 = solve_linkage_symbolically(linkage, simplify=True)
        traj2 = solve_linkage_symbolically(linkage, simplify=False)

        assert "B" in traj1
        assert "B" in traj2


def _same_curve(result, expected):
    """True when *result* and *expected* are the same polynomial up to a constant factor."""
    x, y = sp.symbols("x y", real=True)
    return sp.Poly(result, x, y).monic() == sp.Poly(expected, x, y).monic()


class TestEliminateTheta:
    """Tests for eliminate_theta function."""

    def test_circle_parametric(self):
        """x = cos(theta), y = sin(theta) -> x^2 + y^2 - 1."""
        x, y = sp.symbols("x y", real=True)
        result = eliminate_theta(sp.cos(theta), sp.sin(theta), theta)
        assert result is not None
        assert _same_curve(result, x**2 + y**2 - 1)

    def test_ellipse_with_float_coefficients(self):
        """Floats are read as the rationals they stand for: the result is exact."""
        x, y = sp.symbols("x y", real=True)
        result = eliminate_theta(2.0 * sp.cos(theta) + 1.0, 3.0 * sp.sin(theta), theta)
        assert result is not None
        assert _same_curve(result, 9 * (x - 1) ** 2 + 4 * y**2 - 36)

    def test_multiple_angle(self):
        """cos(2 theta) is polynomial in cos and sin: x = 1 - 2 y^2."""
        x, y = sp.symbols("x y", real=True)
        result = eliminate_theta(sp.cos(2 * theta), sp.sin(theta), theta)
        assert result is not None
        assert _same_curve(result, x + 2 * y**2 - 1)

    @pytest.mark.parametrize(
        ("lengths", "expected"),
        [
            ((4, 1, 3, 3), "(x - 4)**2 + y**2 - 9"),
            ((4.5, 1.25, 3.1, 2.9), "(x - 4.5)**2 + y**2 - 2.9**2"),
        ],
    )
    def test_fourbar_rocker_tip_is_a_circle(self, lengths, expected):
        """The rocker tip's parametrization has a square root; its curve is the rocker circle."""
        from pylinkage.symbolic import fourbar_symbolic

        ground, crank, coupler, rocker = lengths
        linkage = fourbar_symbolic(
            ground_length=ground,
            crank_length=crank,
            coupler_length=coupler,
            rocker_length=rocker,
        )
        x_expr, y_expr = solve_linkage_symbolically(linkage)["C"]

        result = eliminate_theta(x_expr, y_expr)

        assert result is not None
        x, y = sp.symbols("x y", real=True)
        assert _same_curve(result, sp.nsimplify(sp.sympify(expected, locals={"x": x, "y": y})))
        # and the numeric trajectory lies on it
        curve = sp.lambdify((x, y), result)
        positions = compute_trajectory_numeric(linkage, {}, np.linspace(0.1, 6.0, 12))["C"]
        for px, py in positions:
            if not math.isnan(px):
                assert curve(px, py) == pytest.approx(0.0, abs=1e-9)

    def test_unknown_function_of_theta_gives_none(self):
        """A parametrization that is not polynomial in cos and sin is refused, not mangled."""
        assert eliminate_theta(sp.exp(theta), sp.sin(theta), theta) is None


class TestComputeTrajectoryNumeric:
    """Tests for compute_trajectory_numeric function."""

    def test_crank_trajectory(self):
        """Test computing numeric trajectory for crank."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        params = {"r": 2.0}
        theta_vals = np.linspace(0, 2 * np.pi, 100)

        trajectories = compute_trajectory_numeric(linkage, params, theta_vals)

        assert "B" in trajectories
        assert trajectories["B"].shape == (100, 2)

        # Check first point (theta=0): should be (2, 0)
        assert trajectories["B"][0, 0] == pytest.approx(2.0)
        assert trajectories["B"][0, 1] == pytest.approx(0.0)

    def test_fourbar_trajectory(self):
        """Test computing trajectory for four-bar."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", branch=1, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        params = {"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0}
        theta_vals = np.linspace(0, 2 * np.pi, 50)

        trajectories = compute_trajectory_numeric(linkage, params, theta_vals)

        assert "C" in trajectories
        assert trajectories["C"].shape == (50, 2)

        # Verify distances are preserved at all points
        for i in range(50):
            c_x, c_y = trajectories["C"][i]
            b_x, b_y = trajectories["B"][i]
            d_x, d_y = trajectories["D"][i]

            dist_bc = math.sqrt((c_x - b_x) ** 2 + (c_y - b_y) ** 2)
            dist_cd = math.sqrt((c_x - d_x) ** 2 + (c_y - d_y) ** 2)

            assert dist_bc == pytest.approx(3.0, abs=1e-6)
            assert dist_cd == pytest.approx(3.0, abs=1e-6)

    def test_output_joints_filter(self):
        """Test filtering output joints in numeric computation."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius=1, name="B")
        linkage = SymbolicLinkage([A, B])

        theta_vals = np.linspace(0, 2 * np.pi, 10)
        trajectories = compute_trajectory_numeric(linkage, {}, theta_vals, output_joints=["B"])

        assert "B" in trajectories
        assert "A" not in trajectories


class TestCreateTrajectoryFunctions:
    """Tests for create_trajectory_functions function."""

    def test_creates_callable_functions(self):
        """Test that created functions are callable."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        funcs = create_trajectory_functions(linkage)

        assert "B" in funcs
        x_func, y_func, params = funcs["B"]

        # Functions should be callable
        assert callable(x_func)
        assert callable(y_func)

    def test_functions_give_correct_values(self):
        """Test that created functions give correct values."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        funcs = create_trajectory_functions(linkage)
        x_func, y_func, params = funcs["B"]

        # At theta=0 with r=2: x=2, y=0
        x_val = x_func(0, 2.0)
        y_val = y_func(0, 2.0)

        assert x_val == pytest.approx(2.0)
        assert y_val == pytest.approx(0.0)


class TestCheckBuildability:
    """Tests for check_buildability function."""

    def test_buildable_fourbar(self):
        """Test that valid four-bar is buildable."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", branch=1, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        params = {"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0}

        buildable, msg = check_buildability(linkage, params)

        assert buildable is True
        assert msg == ""

    def test_unbuildable_fourbar(self):
        """Test that invalid four-bar is detected as unbuildable."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", branch=1, name="C")
        linkage = SymbolicLinkage([A, D, B, C])

        # Distances too short - circles won't intersect
        params = {"r_AB": 1.0, "r_BC": 0.5, "r_CD": 0.5}

        buildable, msg = check_buildability(linkage, params)

        assert buildable is False
        assert "complex" in msg.lower() or "C" in msg
