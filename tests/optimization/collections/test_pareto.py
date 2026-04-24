"""Tests for ParetoSolution and ParetoFront."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pylinkage.optimization.collections.pareto import ParetoFront, ParetoSolution


def make_sol(scores, dims=None, init=None):
    if dims is None:
        dims = np.array([1.0, 2.0])
    if init is None:
        init = ()
    return ParetoSolution(
        scores=tuple(scores),
        dimensions=dims,
        initial_positions=init,
    )


class TestParetoSolution:
    def test_construction_defaults(self):
        sol = ParetoSolution(scores=(1.0, 2.0), dimensions=np.array([0.5]))
        assert sol.scores == (1.0, 2.0)
        assert sol.initial_positions == ()

    def test_construction_with_initial_positions(self):
        ip = ((0.0, 0.0), (1.0, 1.0))
        sol = ParetoSolution(
            scores=(1.0,),
            dimensions=np.array([0.5]),
            initial_positions=ip,
        )
        assert sol.initial_positions == ip

    def test_construction_with_init_positions_kwarg(self):
        ip = ((2.0, 3.0),)
        sol = ParetoSolution(
            scores=(1.0,),
            dimensions=np.array([0.5]),
            init_positions=ip,
        )
        assert sol.initial_positions == ip

    def test_init_positions_backwards_compat_property(self):
        ip = ((9.0, 9.0),)
        sol = ParetoSolution(
            scores=(0.0,),
            dimensions=np.array([0.5]),
            init_positions=ip,
        )
        assert sol.init_positions == ip

    def test_dominates_true(self):
        a = make_sol((1.0, 1.0))
        b = make_sol((2.0, 2.0))
        assert a.dominates(b) is True

    def test_dominates_false_when_worse_in_one(self):
        a = make_sol((1.0, 3.0))
        b = make_sol((2.0, 2.0))
        assert a.dominates(b) is False

    def test_dominates_false_when_equal(self):
        a = make_sol((1.0, 1.0))
        b = make_sol((1.0, 1.0))
        assert a.dominates(b) is False

    def test_dominates_true_with_one_strict_improvement(self):
        a = make_sol((1.0, 2.0))
        b = make_sol((1.0, 3.0))
        assert a.dominates(b) is True


class TestParetoFrontBasic:
    def test_empty_front(self):
        pf = ParetoFront(solutions=[])
        assert len(pf) == 0
        assert pf.n_objectives == 0
        assert list(iter(pf)) == []

    def test_len_and_getitem(self):
        s1 = make_sol((1.0, 2.0))
        s2 = make_sol((2.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2])
        assert len(pf) == 2
        assert pf[0] is s1
        assert pf[1] is s2

    def test_iteration(self):
        s1 = make_sol((1.0, 2.0))
        s2 = make_sol((2.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2])
        lst = list(pf)
        assert lst == [s1, s2]

    def test_n_objectives(self):
        pf = ParetoFront(solutions=[make_sol((1.0, 2.0, 3.0))])
        assert pf.n_objectives == 3

    def test_scores_array_empty(self):
        pf = ParetoFront(solutions=[])
        arr = pf.scores_array()
        assert isinstance(arr, np.ndarray)
        assert arr.size == 0

    def test_scores_array_populated(self):
        pf = ParetoFront(solutions=[make_sol((1.0, 2.0)), make_sol((3.0, 4.0))])
        arr = pf.scores_array()
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 4.0


class TestParetoFrontFilter:
    def test_filter_preserves_when_under_limit(self):
        s1 = make_sol((1.0, 2.0))
        s2 = make_sol((2.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2], objective_names=("f1", "f2"))
        filtered = pf.filter(max_solutions=5)
        assert len(filtered) == 2
        assert filtered.objective_names == ("f1", "f2")

    def test_filter_reduces(self):
        sols = [
            make_sol((1.0, 5.0)),
            make_sol((2.0, 4.0)),
            make_sol((3.0, 3.0)),
            make_sol((4.0, 2.0)),
            make_sol((5.0, 1.0)),
        ]
        pf = ParetoFront(solutions=sols)
        filtered = pf.filter(max_solutions=3)
        assert len(filtered) == 3

    def test_filter_with_constant_objective(self):
        # obj_range is zero for second objective -> skip branch
        sols = [
            make_sol((1.0, 2.0)),
            make_sol((2.0, 2.0)),
            make_sol((3.0, 2.0)),
            make_sol((4.0, 2.0)),
        ]
        pf = ParetoFront(solutions=sols)
        filtered = pf.filter(max_solutions=2)
        assert len(filtered) == 2


class TestParetoFrontBestCompromise:
    def test_best_compromise_default_weights(self):
        s1 = make_sol((1.0, 10.0))
        s2 = make_sol((5.0, 5.0))
        s3 = make_sol((10.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2, s3])
        best = pf.best_compromise()
        assert best is s2  # the balanced one

    def test_best_compromise_custom_weights(self):
        s1 = make_sol((1.0, 10.0))
        s2 = make_sol((10.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2])
        # Weight objective 0 heavily - first solution minimizes that
        best = pf.best_compromise(weights=(10.0, 1.0))
        assert best is s1

    def test_best_compromise_with_constant_range(self):
        s1 = make_sol((1.0, 5.0))
        s2 = make_sol((1.0, 10.0))
        pf = ParetoFront(solutions=[s1, s2])
        # First objective has zero range — should still work
        best = pf.best_compromise()
        assert best is s1

    def test_best_compromise_empty_raises(self):
        pf = ParetoFront(solutions=[])
        with pytest.raises(ValueError):
            pf.best_compromise()


class TestParetoFrontPlot:
    def test_plot_2d(self):
        s1 = make_sol((1.0, 2.0))
        s2 = make_sol((2.0, 1.0))
        pf = ParetoFront(solutions=[s1, s2])
        fig = pf.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_2d_with_names(self):
        pf = ParetoFront(
            solutions=[make_sol((1.0, 2.0)), make_sol((2.0, 1.0))],
            objective_names=("cost", "time"),
        )
        fig = pf.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_3d(self):
        pf = ParetoFront(
            solutions=[
                make_sol((1.0, 2.0, 3.0)),
                make_sol((2.0, 1.0, 3.0)),
                make_sol((3.0, 2.0, 1.0)),
            ],
        )
        fig = pf.plot(objective_indices=(0, 1, 2))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_3d_with_names(self):
        pf = ParetoFront(
            solutions=[
                make_sol((1.0, 2.0, 3.0)),
                make_sol((2.0, 1.0, 3.0)),
                make_sol((3.0, 2.0, 1.0)),
            ],
            objective_names=("cost", "time", "quality"),
        )
        fig = pf.plot(objective_indices=(0, 1, 2))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_on_existing_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        pf = ParetoFront(solutions=[make_sol((1.0, 2.0)), make_sol((2.0, 1.0))])
        fig2 = pf.plot(ax=ax)
        assert fig2 is not None
        plt.close(fig)

    def test_plot_empty_raises(self):
        pf = ParetoFront(solutions=[])
        with pytest.raises(ValueError):
            pf.plot()

    def test_plot_invalid_indices_count_raises(self):
        pf = ParetoFront(solutions=[make_sol((1.0, 2.0))])
        with pytest.raises(ValueError):
            pf.plot(objective_indices=(0,))  # type: ignore[arg-type]

    def test_plot_out_of_range_index(self):
        pf = ParetoFront(solutions=[make_sol((1.0, 2.0))])
        with pytest.raises(ValueError):
            pf.plot(objective_indices=(0, 5))


class TestParetoFrontHypervolume:
    def test_hypervolume_empty(self):
        pf = ParetoFront(solutions=[])
        hv = pf.hypervolume([10.0, 10.0])
        assert hv == 0.0

    def test_hypervolume_nonempty(self):
        # Requires pymoo
        pytest.importorskip("pymoo")
        pf = ParetoFront(
            solutions=[
                make_sol((1.0, 2.0)),
                make_sol((2.0, 1.0)),
            ]
        )
        hv = pf.hypervolume([5.0, 5.0])
        assert hv > 0.0
