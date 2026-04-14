"""
An inverted four-stroke engine, that converts a rotary motion into a linear one.

See: https://en.wikipedia.org/wiki/Four-stroke_engine for details
"""

import pylinkage as pl
from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import RRPDyad
from pylinkage.simulation import Linkage


def create_stroke_engine_linkage() -> Linkage:
    """Create an inverted stroke engine linkage."""
    anchor = Ground(0.0, 0.0, name="Crank anchor")
    line_a = Ground(0.0, 0.0, name="Line A")
    line_b = Ground(1.0, 0.0, name="Line B")

    crank = Crank(
        anchor=anchor, radius=1.0, angular_velocity=0.1, name="Crank",
    )
    slider = RRPDyad(
        revolute_anchor=crank.output,
        line_anchor1=line_a,
        line_anchor2=line_b,
        distance=1.5,
        name="Slider",
    )

    return Linkage(
        [anchor, line_a, line_b, crank, slider],
        name="Simple four-stroke engine",
    )


def view_linkage() -> None:
    """View a stroke engine linkage in action."""
    stroke_engine = create_stroke_engine_linkage()
    pl.show_linkage(stroke_engine, duration=5)


if __name__ == "__main__":
    view_linkage()
