"""Legacy joint definitions (Assur groups).

.. deprecated:: 0.8.0
    This module uses misleading terminology. What are called "joints" here
    are actually Assur groups (combinations of joints and links).

    Migration guide:
    - ``Static`` -> ``dyads.Ground``
    - ``Crank`` -> ``dyads.Crank``
    - ``Revolute`` -> ``dyads.RRRDyad``
    - ``Prismatic`` -> ``dyads.RRPDyad``
    - ``Fixed`` -> ``dyads.FixedDyad``

    For Assur group decomposition, use ``pylinkage.assur`` module directly.

    Example migration::

        # Old (deprecated):
        from pylinkage.joints import Static, Crank, Revolute
        A = Static(x=0, y=0, name="A")
        B = Crank(x=1, y=0, joint0=A, distance=1, angle=0.1)

        # New (preferred):
        from pylinkage.dyads import Ground, Crank, RRRDyad, Linkage
        A = Ground(0, 0, name="A")
        B = Crank(anchor=A, radius=1.0, angular_velocity=0.1)
"""

import warnings

__all__ = ["Crank", "Fixed", "Prismatic", "Revolute", "Static"]

# Lazy imports: emit deprecation warning only when accessing package-level names.
# Internal submodule imports (from .joint, .crank, etc.) bypass this entirely.
_LAZY_MAP = {
    "Crank": (".crank", "Crank"),
    "Fixed": (".fixed", "Fixed"),
    "Prismatic": (".prismatic", "Prismatic"),
    "Revolute": (".revolute", "Revolute"),
    "Static": (".joint", "Static"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_MAP:
        warnings.warn(
            "The pylinkage.joints module is deprecated. "
            "Use pylinkage.dyads for the new Assur group API. "
            "See module docstring for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        module_path, attr_name = _LAZY_MAP[name]
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
