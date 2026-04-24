"""Tests for the top-level pylinkage package imports and lazy attribute loader."""

from __future__ import annotations

import importlib

import pytest

import pylinkage as pl


class TestEagerImports:
    def test_version(self):
        assert hasattr(pl, "__version__")
        assert isinstance(pl.__version__, str)
        assert len(pl.__version__) > 0

    def test_assur_module(self):
        assert hasattr(pl, "assur")

    def test_dyads_module(self):
        assert hasattr(pl, "dyads")

    def test_mechanism_module(self):
        assert hasattr(pl, "mechanism")

    def test_linkage_class(self):
        assert pl.Linkage is not None

    def test_simulation_class(self):
        assert pl.Simulation is not None

    def test_exceptions(self):
        assert pl.UnbuildableError is not None
        assert pl.UnderconstrainedError is not None
        assert pl.NotCompletelyDefinedError is not None
        assert pl.OptimizationError is not None

    def test_geometry_functions(self):
        assert callable(pl.circle_intersect)
        assert callable(pl.cyl_to_cart)
        assert callable(pl.intersection)
        assert callable(pl.norm)
        assert callable(pl.sqr_dist)

    def test_linkage_helpers(self):
        assert callable(pl.bounding_box)
        assert callable(pl.extract_trajectories)
        assert callable(pl.extract_trajectory)
        assert callable(pl.kinematic_default_test)

    def test_types(self):
        assert pl.JointType is not None
        assert pl.NodeRole is not None
        assert pl.NodeId is not None
        assert pl.EdgeId is not None
        assert pl.HyperedgeId is not None
        assert pl.ComponentId is not None
        assert pl.PortId is not None


class TestLazyImports:
    def test_population_lazy_module(self):
        # Reimport to force a fresh lazy trigger
        mod = importlib.import_module("pylinkage")
        # Access triggers lazy import
        pop_mod = mod.population
        assert pop_mod is not None

    def test_synthesis_lazy_module(self):
        mod = importlib.import_module("pylinkage")
        syn_mod = mod.synthesis
        assert syn_mod is not None

    def test_symbolic_lazy_module(self):
        mod = importlib.import_module("pylinkage")
        sym_mod = mod.symbolic
        assert sym_mod is not None

    def test_particle_swarm_optimization(self):
        assert callable(pl.particle_swarm_optimization)

    def test_trials_and_errors_optimization(self):
        assert callable(pl.trials_and_errors_optimization)

    def test_generate_bounds(self):
        assert callable(pl.generate_bounds)

    def test_kinematic_minimization(self):
        assert callable(pl.kinematic_minimization)

    def test_kinematic_maximization(self):
        assert callable(pl.kinematic_maximization)

    def test_collections(self):
        assert pl.collections is not None

    def test_ensemble(self):
        assert pl.Ensemble is not None

    def test_population(self):
        assert pl.Population is not None

    def test_member(self):
        assert pl.Member is not None

    def test_show_linkage(self):
        assert callable(pl.show_linkage)

    def test_plot_kinematic_linkage(self):
        assert callable(pl.plot_kinematic_linkage)

    def test_plot_static_linkage(self):
        assert callable(pl.plot_static_linkage)

    def test_swarm_tiled_repr(self):
        assert callable(pl.swarm_tiled_repr)

    def test_invalid_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            pl.definitely_not_an_attribute  # noqa: B018
