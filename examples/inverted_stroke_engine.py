"""
An inverted four-stroke engine, that converts a rotary motion into a linear one.

See: https://en.wikipedia.org/wiki/Four-stroke_engine for details
"""
import pylinkage as pl


def create_stroke_engine_linkage():
    """Create an inverted stroke engine linkage."""
    crank = pl.Crank(x=0, y=0, joint0=(0, 0), distance=1, angle=0.1, name="Crank")

    slider = pl.Linear(
        x=2, y=0, joint0=crank, joint1=(0, 0), joint2=(1, 0), revolute_radius=1.5, name="Slider"
    )

    return pl.Linkage(
        joints=(crank, slider),
        order=(crank, slider),
        name="Simple four-stroke engine"
    )


def view_linkage():
    """View a stroke engine linkage in action"""
    stroke_engine = create_stroke_engine_linkage()
    pl.show_linkage(stroke_engine, duration=5)


if __name__ == "__main__":
    view_linkage()
