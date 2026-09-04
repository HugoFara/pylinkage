"""Deprecated public names still resolve, and say so.

Each name here is deprecated and scheduled for removal in 2.0.0. The point of
these tests is that the compatibility promise holds -- the name keeps working,
and keeps pointing at the same object -- while reading it warns.
"""

from __future__ import annotations

import warnings

import pytest

import pylinkage.components
import pylinkage.dyads
import pylinkage.synthesis
from pylinkage.assur import Dyad as AssurDyad
from pylinkage.components import Component, ConnectedComponent
from pylinkage.synthesis import BurmesterDyad

# (module, deprecated name, object it must resolve to)
ALIASES = [
    (pylinkage.components, "Dyad", Component),
    (pylinkage.components, "ConnectedDyad", ConnectedComponent),
    (pylinkage.dyads, "Dyad", Component),
    (pylinkage.dyads, "ConnectedDyad", ConnectedComponent),
    (pylinkage.synthesis, "Dyad", BurmesterDyad),
]


@pytest.mark.parametrize(("module", "name", "target"), ALIASES)
def test_alias_warns_and_resolves(module, name, target):
    """Reading a deprecated alias warns but returns the replacement."""
    with pytest.warns(DeprecationWarning) as record:
        assert getattr(module, name) is target

    message = str(record[0].message)
    assert f"{module.__name__}.{name}" in message
    assert "2.0.0" in message


@pytest.mark.parametrize(("module", "name", "target"), ALIASES)
def test_alias_message_names_the_replacement(module, name, target):
    """The warning tells the caller what to use instead."""
    with pytest.warns(DeprecationWarning) as record:
        getattr(module, name)

    assert target.__name__ in str(record[0].message)


def test_unknown_attribute_still_raises():
    """The hook does not swallow genuine typos."""
    with pytest.raises(AttributeError, match="no attribute 'Dyed'"):
        pylinkage.dyads.Dyed  # noqa: B018


def test_from_import_still_works():
    """``from ... import Dyad`` keeps working, which is the whole promise."""
    with pytest.warns(DeprecationWarning):
        from pylinkage.dyads import Dyad

    assert Dyad is Component


def test_deprecated_names_stay_in_dunder_all():
    """``import *`` must keep exporting them until they are removed."""
    assert "Dyad" in pylinkage.dyads.__all__
    assert "ConnectedDyad" in pylinkage.dyads.__all__
    assert "Dyad" in pylinkage.components.__all__
    assert "Dyad" in pylinkage.synthesis.__all__


def test_no_warning_for_the_replacements():
    """The names people should move to are silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert pylinkage.components.Component is Component
        assert pylinkage.synthesis.BurmesterDyad is BurmesterDyad
        assert pylinkage.dyads.RRRDyad is not None


def test_the_three_dyads_are_distinct():
    """The collision that motivated the rename is gone.

    ``assur.Dyad`` is an Assur group, ``synthesis.BurmesterDyad`` a Burmester
    construct, and the ``dyads``/``components`` aliases were never dyads at
    all. Only one bare ``Dyad`` class remains.
    """
    assert AssurDyad is not BurmesterDyad
    assert AssurDyad.__name__ == "Dyad"
    assert BurmesterDyad.__name__ == "BurmesterDyad"
    assert not hasattr(Component, "circle_point")
