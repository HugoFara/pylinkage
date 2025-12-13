"""Prebuilt linkage examples for the web editor."""

import math

import pylinkage as pl
from pylinkage.joints import Crank, Fixed, Prismatic, Revolute, Static


def get_example_names() -> list[str]:
    """Return list of available example names."""
    return [
        "Four-Bar Linkage",
        "Chebyshev Linkage",
        "Crank-Slider",
        "Watt Linkage",
        "Pantograph",
    ]


def load_example(name: str) -> pl.Linkage:
    """Load a prebuilt example linkage by name."""
    examples = {
        "Four-Bar Linkage": create_fourbar,
        "Chebyshev Linkage": create_chebyshev,
        "Crank-Slider": create_crank_slider,
        "Watt Linkage": create_watt,
        "Pantograph": create_pantograph,
    }
    factory = examples.get(name)
    if factory is None:
        raise ValueError(f"Unknown example: {name}")
    return factory()


def create_fourbar() -> pl.Linkage:
    """Create a standard four-bar linkage."""
    # Ground anchors
    anchor_a = Static(0, 0, name="A")
    anchor_d = Static(3, 0, name="D")

    # Crank (input link)
    crank_b = Crank(
        0,
        1,
        joint0=anchor_a,
        distance=1,
        angle=2 * math.pi / 20,  # 20 steps per revolution
        name="B",
    )

    # Coupler point
    coupler_c = Revolute(
        3,
        2,
        joint0=crank_b,
        joint1=anchor_d,
        distance0=3,
        distance1=2,
        name="C",
    )

    return pl.Linkage(
        joints=[anchor_a, anchor_d, crank_b, coupler_c],
        order=[anchor_a, anchor_d, crank_b, coupler_c],
        name="Four-Bar Linkage",
    )


def create_chebyshev() -> pl.Linkage:
    """Create a Chebyshev straight-line mechanism (lambda linkage).

    Classic proportions: ground=4, crank=1, coupler=4, rocker=2.
    The midpoint of the coupler traces an approximately straight line.
    """
    # Ground anchors - symmetric about y-axis
    anchor_a = Static(-2, 0, name="A")
    anchor_b = Static(2, 0, name="B")

    # Short crank from anchor A
    crank_c = Crank(
        -1.5,
        0.86,
        joint0=anchor_a,
        distance=1,  # Short crank for full rotation
        angle=2 * math.pi / 30,
        name="C",
    )

    # Rocker at B with longer coupler connection
    pin_d = Revolute(
        1.5,
        1.73,
        joint0=crank_c,
        joint1=anchor_b,
        distance0=4,  # Coupler length
        distance1=2,  # Rocker length
        name="D",
    )

    # Midpoint tracer (creates the approximately straight line)
    midpoint_p = Fixed(
        0,
        1.3,
        joint0=crank_c,
        joint1=pin_d,
        distance=2,  # Half of coupler
        angle=0,
        name="P",
    )

    return pl.Linkage(
        joints=[anchor_a, anchor_b, crank_c, pin_d, midpoint_p],
        order=[anchor_a, anchor_b, crank_c, pin_d, midpoint_p],
        name="Chebyshev Linkage",
    )


def create_crank_slider() -> pl.Linkage:
    """Create a crank-slider mechanism."""
    # Ground anchor
    anchor_a = Static(0, 0, name="A")

    # Crank
    crank_b = Crank(
        1,
        0,
        joint0=anchor_a,
        distance=1,
        angle=2 * math.pi / 24,
        name="B",
    )

    # Slider on horizontal rail (y=0)
    rail_start = Static(-3, 0, name="Rail_Start")
    rail_end = Static(5, 0, name="Rail_End")

    slider_c = Prismatic(
        2,
        0,
        joint0=crank_b,
        joint1=rail_start,
        joint2=rail_end,
        revolute_radius=2,
        name="C",
    )

    return pl.Linkage(
        joints=[anchor_a, rail_start, rail_end, crank_b, slider_c],
        order=[anchor_a, rail_start, rail_end, crank_b, slider_c],
        name="Crank-Slider",
    )


def create_watt() -> pl.Linkage:
    """Create a Watt's parallel motion linkage."""
    # Two ground anchors
    anchor_a = Static(0, 0, name="A")
    anchor_d = Static(4, 0, name="D")

    # Left arm (crank)
    crank_b = Crank(
        0,
        2,
        joint0=anchor_a,
        distance=2,
        angle=2 * math.pi / 30,
        name="B",
    )

    # Right arm (rocker)
    rocker_c = Revolute(
        4,
        2,
        joint0=crank_b,
        joint1=anchor_d,
        distance0=4,
        distance1=2,
        name="C",
    )

    # Midpoint tracer (approximate straight line)
    midpoint_e = Fixed(
        2,
        2,
        joint0=crank_b,
        joint1=rocker_c,
        distance=2,
        angle=0,
        name="E",
    )

    return pl.Linkage(
        joints=[anchor_a, anchor_d, crank_b, rocker_c, midpoint_e],
        order=[anchor_a, anchor_d, crank_b, rocker_c, midpoint_e],
        name="Watt Linkage",
    )


def create_pantograph() -> pl.Linkage:
    """Create a simple pantograph linkage for scaling."""
    # Fixed pivot
    anchor_o = Static(0, 0, name="O")

    # Input point driven by crank
    crank_a = Crank(
        1,
        1,
        joint0=anchor_o,
        distance=math.sqrt(2),
        angle=2 * math.pi / 30,
        name="A",
    )

    # Parallelogram vertices
    point_b = Revolute(
        2,
        2,
        joint0=crank_a,
        joint1=anchor_o,
        distance0=math.sqrt(2),
        distance1=2 * math.sqrt(2),
        name="B",
    )

    # Output point (scaled copy of input motion)
    point_c = Fixed(
        2,
        0,
        joint0=anchor_o,
        joint1=point_b,
        distance=2 * math.sqrt(2),
        angle=0,
        name="C",
    )

    return pl.Linkage(
        joints=[anchor_o, crank_a, point_b, point_c],
        order=[anchor_o, crank_a, point_b, point_c],
        name="Pantograph",
    )
