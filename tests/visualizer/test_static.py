from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRRDyad
from pylinkage.simulation import Linkage
from pylinkage.visualizer.static import plot_static_linkage


def _fourbar():
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output, anchor2=O2, distance1=2.5, distance2=2.0, name="rocker"
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


def _get_loci(lk, n=10):
    return [tuple(frame) for frame in lk.step(iterations=n)]


class TestPlotStaticLinkage:
    def teardown_method(self):
        plt.close("all")

    def test_basic(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci)

    def test_show_legend(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, show_legend=True)

    def test_show_legend_with_title(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, show_legend=True, title="Custom")

    def test_no_labels(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, show_labels=False)

    def test_no_loci_drawing(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, show_loci=False)

    def test_with_ghosts(self):
        lk = _fourbar()
        loci = _get_loci(lk, n=20)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, n_ghosts=3)

    def test_single_ghost(self):
        lk = _fourbar()
        loci = _get_loci(lk, n=20)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, n_ghosts=1)

    def test_with_title(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, loci, title="Title")

    def test_locus_highlights(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        fig, ax = plt.subplots()
        highlights = [[(1.0, 0.5), (1.2, 0.6)], [(2.0, 0.5)]]
        plot_static_linkage(lk, ax, loci, locus_highlights=highlights)

    def test_empty_loci(self):
        lk = _fourbar()
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, [])

    def test_loci_with_none_coords(self):
        lk = _fourbar()
        loci = _get_loci(lk)
        # Inject None to exercise None skipping paths
        first = loci[0]
        mutated = list(loci)
        mutated[1] = tuple((None, None) if i == len(first) - 1 else first[i] for i in range(len(first)))
        fig, ax = plt.subplots()
        plot_static_linkage(lk, ax, mutated, n_ghosts=2)
