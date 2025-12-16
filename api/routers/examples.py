"""Prebuilt example endpoints using mechanism module."""

import math

from fastapi import APIRouter, HTTPException

from pylinkage.mechanism import MechanismBuilder
from pylinkage.mechanism.serialization import mechanism_to_dict

from ..models.mechanism_schemas import ExampleInfo, MechanismResponse
from ..services import mechanism_service
from ..storage.memory import storage

router = APIRouter(prefix="/examples", tags=["examples"])


def create_four_bar_mechanism() -> dict:
    """Create a classic four-bar linkage using MechanismBuilder.

    Dimensions satisfy Grashof condition: s + l <= p + q
    where s=25 (crank), l=100 (ground), p=60 (rocker), q=70 (coupler)
    25 + 100 = 125 <= 60 + 70 = 130 ✓
    """
    mechanism = (
        MechanismBuilder("Four-Bar Linkage")
        .add_ground_link("ground", ports={"O1": (0, 0), "O2": (100, 0)})
        .add_driver_link(
            "crank",
            length=25,
            motor_port="O1",
            omega=0.1,
            initial_angle=math.pi / 4,  # 45 degrees
        )
        .add_link("coupler", length=70)
        .add_link("rocker", length=60)
        .connect("crank.tip", "coupler.0")
        .connect("coupler.1", "rocker.0")
        .connect("rocker.1", "ground.O2")
        .build()
    )
    return mechanism_to_dict(mechanism)


def create_chebyshev_mechanism() -> dict:
    """Create Chebyshev straight-line mechanism.

    Classical Chebyshev linkage proportions (unit length a):
    - Link 1 (ground): 4a = 100
    - Links 2 & 4 (crank/rocker): 5a = 125
    - Link 3 (coupler): 2a = 50

    The coupler midpoint traces an approximate straight line.
    """
    mechanism = (
        MechanismBuilder("Chebyshev Linkage")
        .add_ground_link("ground", ports={"O1": (0, 0), "O2": (100, 0)})
        .add_driver_link(
            "crank",
            length=125,
            motor_port="O1",
            omega=0.1,
            initial_angle=2.094,  # 120 degrees
        )
        .add_link("coupler", length=50)
        .add_link("rocker", length=125)
        .connect("crank.tip", "coupler.0")
        .connect("coupler.1", "rocker.0")
        .connect("rocker.1", "ground.O2")
        .build()
    )
    return mechanism_to_dict(mechanism)


def create_crank_slider_mechanism() -> dict:
    """Create a crank-slider mechanism."""
    mechanism = (
        MechanismBuilder("Crank-Slider Mechanism")
        .add_ground_link("ground", ports={"O": (0, 0)})
        .add_driver_link(
            "crank",
            length=50,
            motor_port="O",
            omega=0.1,
            initial_angle=math.atan2(40, 30),
        )
        .add_link("connecting_rod", length=100)
        .add_slide_axis("rail", through=(0, 0), direction=(1, 0))
        .connect("crank.tip", "connecting_rod.0")
        .connect_prismatic("connecting_rod.1", "rail")
        .build()
    )
    return mechanism_to_dict(mechanism)


def create_pantograph_mechanism() -> dict:
    """Create a pantograph linkage."""
    mechanism = (
        MechanismBuilder("Pantograph")
        .add_ground_link("ground", ports={"O": (0, 0)})
        .add_driver_link(
            "crank",
            length=50,
            motor_port="O",
            omega=0.1,
            initial_angle=0.01,  # Small initial angle
        )
        .add_link("arm", length=50)
        .connect("crank.tip", "arm.0")
        .connect("arm.1", "ground.O")
        .build()
    )
    return mechanism_to_dict(mechanism)


# Registry of available examples
EXAMPLES: dict[str, tuple[callable, str]] = {
    "four-bar": (create_four_bar_mechanism, "Classic four-bar linkage mechanism"),
    "chebyshev": (create_chebyshev_mechanism, "Chebyshev straight-line mechanism"),
    "crank-slider": (create_crank_slider_mechanism, "Crank-slider mechanism"),
    "pantograph": (create_pantograph_mechanism, "Pantograph scaling mechanism"),
}


@router.get("", response_model=list[ExampleInfo])
def list_examples() -> list[ExampleInfo]:
    """List all available example mechanisms."""
    result = []
    for name, (factory, description) in EXAMPLES.items():
        mech_dict = factory()
        result.append(
            ExampleInfo(
                name=name,
                description=description,
                joint_count=len(mech_dict.get("joints", [])),
                link_count=len(mech_dict.get("links", [])),
            )
        )
    return result


@router.get("/{example_name}")
def get_example(example_name: str) -> dict:
    """Get an example mechanism as JSON (mechanism format)."""
    if example_name not in EXAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"Example '{example_name}' not found. Available: {list(EXAMPLES.keys())}",
        )

    factory, _ = EXAMPLES[example_name]
    return factory()


@router.post("/{example_name}/load", response_model=MechanismResponse)
def load_example(example_name: str) -> MechanismResponse:
    """Load an example into storage and return its ID."""
    if example_name not in EXAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"Example '{example_name}' not found. Available: {list(EXAMPLES.keys())}",
        )

    factory, _ = EXAMPLES[example_name]
    mechanism_dict = factory()

    # Validate and store
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)
    mechanism_id = storage.create(mechanism_dict)
    stored = storage.get(mechanism_id)

    if stored is None:
        raise HTTPException(status_code=500, detail="Failed to load example")

    response_data = mechanism_service.mechanism_to_response_dict(
        mechanism_id, stored, mechanism, is_buildable, error
    )
    return MechanismResponse(**response_data)
