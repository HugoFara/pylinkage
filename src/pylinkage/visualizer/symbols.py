"""
Kinematic symbol definitions for engineering diagrams.

This module provides shared symbol metadata used by visualization backends
to render proper ISO 3952 kinematic diagram symbols.
"""


from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..joints.joint import Joint


class SymbolType(Enum):
    """Types of kinematic symbols."""

    GROUND = auto()  # Fixed support (triangle with hatching)
    REVOLUTE = auto()  # Pin joint (circle with center dot)
    CRANK = auto()  # Motor/input joint (circle with rotation arrow)
    SLIDER = auto()  # Prismatic joint (rectangle on rails)
    FIXED = auto()  # Fixed distance joint
    LINEAR = auto()  # Linear constraint joint


class LinkStyle(Enum):
    """Visual styles for drawing links between joints."""

    BAR = auto()  # Rounded rectangular bar with centerline
    BONE = auto()  # Dog-bone shape (circles at ends, narrow middle)
    LINE = auto()  # Simple thick line


@dataclass(frozen=True)
class SymbolSpec:
    """Specification for a kinematic symbol."""

    symbol_type: SymbolType
    color: str  # Hex color code
    size: float  # Relative size multiplier
    label_offset: tuple[float, float]  # Label position offset (x, y)


# Symbol specifications for each joint type
# Maps joint class names to their visual specifications
SYMBOL_SPECS: dict[str, SymbolSpec] = {
    "Static": SymbolSpec(SymbolType.GROUND, "#333333", 1.0, (0, 1.5)),
    "Crank": SymbolSpec(SymbolType.CRANK, "#2E86AB", 1.2, (0.5, -0.5)),
    "Revolute": SymbolSpec(SymbolType.REVOLUTE, "#E63946", 1.0, (0.5, -0.5)),
    "Pivot": SymbolSpec(SymbolType.REVOLUTE, "#E63946", 0.9, (0.5, -0.5)),
    "Fixed": SymbolSpec(SymbolType.FIXED, "#F4A261", 1.0, (0.5, -0.5)),
    "Linear": SymbolSpec(SymbolType.SLIDER, "#F18F01", 1.1, (0.5, -0.5)),  # Deprecated
    "Prismatic": SymbolSpec(SymbolType.SLIDER, "#F18F01", 1.1, (0.5, -0.5)),
}

# Link colors for consistent styling across backends
LINK_COLORS: list[str] = [
    "#2E86AB",  # Blue
    "#A23B72",  # Magenta
    "#F18F01",  # Orange
    "#4ECDC4",  # Teal
    "#95E1D3",  # Light green
    "#DDA0DD",  # Plum
    "#6B5B95",  # Purple
    "#88B04B",  # Green
]


def get_symbol_spec(joint: "Joint") -> SymbolSpec:
    """Get the symbol specification for a joint.

    Args:
        joint: The joint to get the symbol for.

    Returns:
        SymbolSpec for the joint type, or a default REVOLUTE spec if unknown.
    """
    class_name = type(joint).__name__
    return SYMBOL_SPECS.get(
        class_name,
        SymbolSpec(SymbolType.REVOLUTE, "#666666", 1.0, (0.5, -0.5)),
    )


def get_link_color(index: int) -> str:
    """Get a link color by index, cycling through available colors.

    Args:
        index: The link index.

    Returns:
        Hex color string.
    """
    return LINK_COLORS[index % len(LINK_COLORS)]


def is_ground_joint(joint: "Joint") -> bool:
    """Check if a joint should be rendered as a ground/fixed support.

    Args:
        joint: The joint to check.

    Returns:
        True if the joint is a ground joint (Static with no parents).
    """
    return type(joint).__name__ == "Static" and joint.joint0 is None
