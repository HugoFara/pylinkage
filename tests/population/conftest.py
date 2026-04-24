"""Shared fixtures for pylinkage.population tests."""

from __future__ import annotations

import numpy as np
import pytest

import math

from pylinkage.actuators import Crank
from pylinkage.components import Ground
from pylinkage.dyads import FixedDyad, RRRDyad
from pylinkage.simulation import Linkage


def build_fourbar() -> Linkage:
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(3.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.0, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=2.5,
        distance2=2.0,
        name="rocker",
    )
    return Linkage([O1, O2, crank, rocker], name="FourBar")


def build_fourbar_variant() -> Linkage:
    """A 5-joint variant with a fixed coupler point (different topology)."""
    O1 = Ground(0.0, 0.0, name="O1")
    O2 = Ground(4.0, 0.0, name="O2")
    crank = Crank(anchor=O1, radius=1.2, angular_velocity=0.1, name="crank")
    rocker = RRRDyad(
        anchor1=crank.output,
        anchor2=O2,
        distance1=3.0,
        distance2=2.5,
        name="rocker",
    )
    coupler = FixedDyad(
        anchor1=rocker,
        anchor2=crank.output,
        distance=0.5,
        angle=math.pi / 3,
        name="coupler",
    )
    return Linkage(
        [O1, O2, crank, rocker, coupler], name="FourBarVariant",
    )


@pytest.fixture
def fourbar() -> Linkage:
    return build_fourbar()


@pytest.fixture
def fourbar_variant() -> Linkage:
    return build_fourbar_variant()


@pytest.fixture
def member_dims() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.5, 2.0],
            [0.9, 2.4, 2.1],
            [1.1, 2.6, 1.9],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def member_positions(fourbar: Linkage) -> np.ndarray:
    coords = [
        [x if x is not None else 0.0, y if y is not None else 0.0]
        for (x, y) in fourbar.get_coords()
    ]
    base = np.array(coords, dtype=np.float64)
    return np.stack([base, base, base])


@pytest.fixture
def member_scores() -> dict[str, np.ndarray]:
    return {
        "score": np.array([3.0, 1.0, 2.0], dtype=np.float64),
        "extra": np.array([0.5, 0.7, 0.6], dtype=np.float64),
    }
