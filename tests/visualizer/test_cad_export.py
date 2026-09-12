"""DXF and STEP export of a component-API linkage.

Skipped without the ``cad`` extra. Both exporters used to walk the legacy
``linkage.joints`` / ``joint0`` attributes and could not draw a
``simulation.Linkage`` at all.
"""

from __future__ import annotations

from collections import Counter

import pytest

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRPDyad, RRRDyad
from pylinkage.simulation import Linkage

ezdxf = pytest.importorskip("ezdxf")


def _fourbar():
    A = Ground(0.0, 0.0, name="A")
    D = Ground(3.0, 0.0, name="D")
    crank = Crank(anchor=A, radius=1.0, angular_velocity=0.31, name="crank")
    rocker = RRRDyad(anchor1=crank.output, anchor2=D, distance1=3.0, distance2=2.0, name="rocker")
    return Linkage([A, D, crank, rocker], name="FourBar")


def _slider_crank():
    O1 = Ground(0.0, 0.0, name="O1")
    L1 = Ground(0.0, -1.0, name="L1")
    L2 = Ground(4.0, -1.0, name="L2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    slider = RRPDyad(
        revolute_anchor=crank.output, line_anchor1=L1, line_anchor2=L2, distance=2.0, name="s"
    )
    return Linkage([O1, L1, L2, crank, slider], name="SliderCrank")


def _links_layer(doc) -> Counter:
    return Counter(e.dxftype() for e in doc.modelspace() if e.dxf.layer == "LINKS")


class TestDxf:
    def test_fourbar_has_three_link_bars(self):
        from pylinkage.visualizer import plot_linkage_dxf

        doc = plot_linkage_dxf(_fourbar())
        assert _links_layer(doc)["LWPOLYLINE"] == 3

    def test_slider_crank_has_rod_and_rail(self):
        from pylinkage.visualizer import plot_linkage_dxf

        doc = plot_linkage_dxf(_slider_crank())
        # crank, connecting rod, rail
        assert _links_layer(doc)["LWPOLYLINE"] == 3

    def test_save_with_frame(self, tmp_path):
        from pylinkage.visualizer import save_linkage_dxf

        linkage = _fourbar()
        loci = list(linkage.step())
        path = tmp_path / "frame.dxf"
        save_linkage_dxf(linkage, path, loci=loci, frame_index=5, link_width=0.2)
        assert path.stat().st_size > 0
        with pytest.raises(ValueError, match="frame_index"):
            save_linkage_dxf(linkage, path, loci=loci, frame_index=len(loci))


class TestStep:
    def test_fourbar_builds_bars_and_pins(self):
        pytest.importorskip("build123d")
        from pylinkage.visualizer import JointProfile, LinkProfile, build_linkage_3d

        model = build_linkage_3d(
            _fourbar(),
            link_profile=LinkProfile(width=0.3, thickness=0.1),
            joint_profile=JointProfile(radius=0.08, length=0.2),
        )
        # 3 bars + 4 pins + 2 ground symbols
        assert len(list(model.solids())) == 9

    def test_save(self, tmp_path):
        pytest.importorskip("build123d")
        from pylinkage.visualizer import save_linkage_step

        path = tmp_path / "fourbar.step"
        save_linkage_step(_fourbar(), path)
        assert path.stat().st_size > 0
