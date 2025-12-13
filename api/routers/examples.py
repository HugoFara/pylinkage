"""Prebuilt example endpoints."""

from fastapi import APIRouter, HTTPException

import pylinkage as pl
from pylinkage.joints import Crank, Revolute, Static

from ..models.schemas import ExampleInfo, LinkageResponse
from ..services import linkage_service
from ..storage.memory import storage

router = APIRouter(prefix="/examples", tags=["examples"])


def create_four_bar() -> pl.Linkage:
    """Create a classic four-bar linkage."""
    import math

    # Ground pivots
    ground_a = Static(x=0, y=0, name="GroundA")
    ground_d = Static(x=100, y=0, name="GroundD")

    # Crank (input link) - use non-zero initial angle
    crank_b = Crank(
        x=30,
        y=40,
        joint0=ground_a,
        distance=50,
        angle=math.atan2(40, 30),  # Initial angle based on position
        name="CrankB",
    )

    # Coupler-Rocker connection (output)
    rocker_c = Revolute(
        x=80,
        y=60,
        joint0=crank_b,
        joint1=ground_d,
        distance0=60,
        distance1=70,
        name="RockerC",
    )

    return pl.Linkage(
        joints=[ground_a, ground_d, crank_b, rocker_c],
        name="Four-Bar Linkage",
    )


def create_chebyshev() -> pl.Linkage:
    """Create a Chebyshev straight-line mechanism."""
    # Ground frame
    ground_a = Static(x=0, y=0, name="GroundA")
    ground_b = Static(x=100, y=0, name="GroundB")

    # Crank
    crank = Crank(
        x=-25,
        y=43.3,
        joint0=ground_a,
        distance=50,
        angle=2.094,  # 120 degrees
        name="Crank",
    )

    # Coupler point
    coupler = Revolute(
        x=50,
        y=86.6,
        joint0=crank,
        joint1=ground_b,
        distance0=100,
        distance1=100,
        name="Coupler",
    )

    return pl.Linkage(
        joints=[ground_a, ground_b, crank, coupler],
        name="Chebyshev Linkage",
    )


def create_crank_slider() -> pl.Linkage:
    """Create a crank-slider mechanism."""
    import math

    # Ground
    ground = Static(x=0, y=0, name="Ground")

    # Crank - use non-zero initial angle
    crank = Crank(
        x=30,
        y=40,
        joint0=ground,
        distance=50,
        angle=math.atan2(40, 30),  # Initial angle based on position
        name="Crank",
    )

    # Connecting rod end (slider)
    # Using Revolute with ground as second parent for simplicity
    slider = Revolute(
        x=130,
        y=0,
        joint0=crank,
        joint1=ground,
        distance0=100,
        distance1=130,
        name="Slider",
    )

    return pl.Linkage(
        joints=[ground, crank, slider],
        name="Crank-Slider Mechanism",
    )


def create_pantograph() -> pl.Linkage:
    """Create a pantograph linkage."""
    import math

    # Fixed pivot
    pivot = Static(x=0, y=0, name="Pivot")

    # Crank arm - use small non-zero angle to avoid division by zero
    crank = Crank(
        x=50,
        y=0,
        joint0=pivot,
        distance=50,
        angle=0.01,  # Small initial angle
        name="Crank",
    )

    # Parallelogram points
    point_b = Revolute(
        x=100,
        y=0,
        joint0=crank,
        joint1=pivot,
        distance0=50,
        distance1=100,
        name="PointB",
    )

    return pl.Linkage(
        joints=[pivot, crank, point_b],
        name="Pantograph",
    )


# Registry of available examples
EXAMPLES: dict[str, tuple[callable, str]] = {
    "four-bar": (create_four_bar, "Classic four-bar linkage mechanism"),
    "chebyshev": (create_chebyshev, "Chebyshev straight-line mechanism"),
    "crank-slider": (create_crank_slider, "Crank-slider mechanism"),
    "pantograph": (create_pantograph, "Pantograph scaling mechanism"),
}


@router.get("", response_model=list[ExampleInfo])
def list_examples() -> list[ExampleInfo]:
    """List all available example linkages."""
    result = []
    for name, (factory, description) in EXAMPLES.items():
        linkage = factory()
        result.append(
            ExampleInfo(
                name=name,
                description=description,
                joint_count=len(linkage.joints),
            )
        )
    return result


@router.get("/{example_name}")
def get_example(example_name: str) -> dict:
    """Get an example linkage as JSON (pylinkage format)."""
    if example_name not in EXAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"Example '{example_name}' not found. Available: {list(EXAMPLES.keys())}",
        )

    factory, _ = EXAMPLES[example_name]
    linkage = factory()
    return linkage.to_dict()


@router.post("/{example_name}/load", response_model=LinkageResponse)
def load_example(example_name: str) -> LinkageResponse:
    """Load an example into storage and return its ID."""
    if example_name not in EXAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"Example '{example_name}' not found. Available: {list(EXAMPLES.keys())}",
        )

    factory, _ = EXAMPLES[example_name]
    linkage = factory()
    linkage_dict = linkage.to_dict()

    # Validate and store
    built_linkage, is_buildable, error = linkage_service.validate_and_build(linkage_dict)
    linkage_id = storage.create(linkage_dict)
    stored = storage.get(linkage_id)

    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to load example")

    response_data = linkage_service.linkage_to_response_dict(
        linkage_id, stored, built_linkage, is_buildable, error
    )
    return LinkageResponse(**response_data)
