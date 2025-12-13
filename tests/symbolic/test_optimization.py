"""Tests for symbolic optimization."""

import numpy as np
import pytest
import sympy as sp

from pylinkage.symbolic import (
    OptimizationResult,
    SymbolicLinkage,
    SymbolicOptimizer,
    SymCrank,
    SymRevolute,
    SymStatic,
    generate_symbolic_bounds,
    symbolic_gradient,
    symbolic_hessian,
    theta,
)


class TestSymbolicGradient:
    """Tests for symbolic_gradient function."""

    def test_simple_polynomial(self):
        """Test gradient of simple polynomial."""
        x, y = sp.symbols("x y")
        f = x**2 + y**2

        grad = symbolic_gradient(f, [x, y])

        assert len(grad) == 2
        assert grad[0] == 2 * x
        assert grad[1] == 2 * y

    def test_mixed_terms(self):
        """Test gradient with mixed terms."""
        x, y = sp.symbols("x y")
        f = x * y + x**2

        grad = symbolic_gradient(f, [x, y])

        assert grad[0] == y + 2 * x
        assert grad[1] == x


class TestSymbolicHessian:
    """Tests for symbolic_hessian function."""

    def test_quadratic(self):
        """Test Hessian of quadratic function."""
        x, y = sp.symbols("x y")
        f = x**2 + 2 * y**2 + x * y

        hess = symbolic_hessian(f, [x, y])

        assert hess.shape == (2, 2)
        assert hess[0, 0] == 2  # d^2f/dx^2
        assert hess[1, 1] == 4  # d^2f/dy^2
        assert hess[0, 1] == 1  # d^2f/dxdy
        assert hess[1, 0] == 1  # d^2f/dydx


class TestGenerateSymbolicBounds:
    """Tests for generate_symbolic_bounds function."""

    def test_default_bounds(self):
        """Test default bound generation."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        center = {"r": 5.0}
        bounds = generate_symbolic_bounds(linkage, center)

        assert "r" in bounds
        assert bounds["r"] == (1.0, 25.0)  # 5/5, 5*5

    def test_custom_ratios(self):
        """Test custom min_ratio and max_factor."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        linkage = SymbolicLinkage([A, B])

        center = {"r": 10.0}
        bounds = generate_symbolic_bounds(linkage, center, min_ratio=2, max_factor=3)

        assert bounds["r"] == (5.0, 30.0)  # 10/2, 10*3


class TestSymbolicOptimizer:
    """Tests for SymbolicOptimizer class."""

    @pytest.fixture
    def simple_linkage(self):
        """Create a simple crank linkage."""
        A = SymStatic(0, 0, name="A")
        B = SymCrank(A, radius="r", name="B")
        return SymbolicLinkage([A, B])

    @pytest.fixture
    def fourbar_linkage(self):
        """Create a four-bar linkage."""
        A = SymStatic(0, 0, name="A")
        D = SymStatic(4, 0, name="D")
        B = SymCrank(A, radius="r_AB", name="B")
        C = SymRevolute(B, D, distance0="r_BC", distance1="r_CD", branch=1, name="C")
        return SymbolicLinkage([A, D, B, C])

    def test_initialization(self, simple_linkage):
        """Test optimizer initialization."""

        def objective(trajectories):
            x, y = trajectories["B"]
            return (x - 1) ** 2 + y**2

        optimizer = SymbolicOptimizer(simple_linkage, objective)

        assert optimizer.linkage is simple_linkage
        assert optimizer.objective_expr is not None
        assert len(optimizer.gradient_exprs) == 1  # One parameter: r

    def test_evaluate(self, simple_linkage):
        """Test objective evaluation."""

        def objective(trajectories):
            x, y = trajectories["B"]
            return x**2 + y**2  # Distance from origin squared

        optimizer = SymbolicOptimizer(simple_linkage, objective, theta_samples=50)

        # With r=1, the trajectory is a unit circle, so x^2 + y^2 = 1
        value = optimizer.evaluate({"r": 1.0})
        assert value == pytest.approx(1.0, abs=0.1)

        # With r=2, the trajectory is a circle of radius 2, so x^2 + y^2 = 4
        value = optimizer.evaluate({"r": 2.0})
        assert value == pytest.approx(4.0, abs=0.1)

    def test_gradient(self, simple_linkage):
        """Test gradient evaluation."""

        def objective(trajectories):
            x, y = trajectories["B"]
            return x**2 + y**2  # r^2

        optimizer = SymbolicOptimizer(simple_linkage, objective, theta_samples=50)

        # d(r^2)/dr = 2r
        grad = optimizer.gradient({"r": 1.0})
        assert len(grad) == 1
        assert grad[0] == pytest.approx(2.0, abs=0.2)

        grad = optimizer.gradient({"r": 2.0})
        assert grad[0] == pytest.approx(4.0, abs=0.2)

    def test_optimize(self, simple_linkage):
        """Test optimization finds correct minimum."""

        def objective(trajectories):
            x, y = trajectories["B"]
            # Minimize (r - 2)^2 by using (x^2 + y^2 - 4)^2
            # For a circle of radius r, x^2 + y^2 = r^2
            # So we minimize (r^2 - 4)^2, which has minimum at r=2
            return (x**2 + y**2 - 4) ** 2

        optimizer = SymbolicOptimizer(simple_linkage, objective, theta_samples=50)

        result = optimizer.optimize(
            initial_params={"r": 1.0},
            bounds={"r": (0.5, 5.0)},
            maxiter=100,
        )

        assert isinstance(result, OptimizationResult)
        # Should converge to r close to 2
        assert result.params["r"] == pytest.approx(2.0, abs=0.3)

    def test_optimize_fourbar(self, fourbar_linkage):
        """Test optimization of four-bar linkage."""

        def objective(trajectories):
            x, y = trajectories["C"]
            # Minimize y coordinate (want coupler to be low)
            return y**2

        optimizer = SymbolicOptimizer(fourbar_linkage, objective, theta_samples=30)

        initial = {"r_AB": 1.0, "r_BC": 3.0, "r_CD": 3.0}
        bounds = {"r_AB": (0.5, 2.0), "r_BC": (2.0, 4.0), "r_CD": (2.0, 4.0)}

        result = optimizer.optimize(
            initial_params=initial,
            bounds=bounds,
            maxiter=50,
        )

        assert isinstance(result, OptimizationResult)
        # Just check it ran and returned valid parameters
        for name in initial:
            assert name in result.params
            assert bounds[name][0] <= result.params[name] <= bounds[name][1]

    def test_custom_theta_samples(self, simple_linkage):
        """Test using custom theta samples."""

        def objective(trajectories):
            x, y = trajectories["B"]
            return x**2

        optimizer = SymbolicOptimizer(simple_linkage, objective)

        # Use specific theta values
        theta_vals = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        value = optimizer.evaluate({"r": 1.0}, theta_samples=theta_vals)

        # At these theta values, x = cos(theta) = 1, 0, -1, 0
        # x^2 = 1, 0, 1, 0 -> mean = 0.5
        assert value == pytest.approx(0.5, abs=0.01)
