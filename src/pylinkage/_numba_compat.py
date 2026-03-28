"""Numba compatibility layer.

Provides a no-op ``@njit`` decorator when numba is not installed,
so geometry and solver modules work (slower) with pure Python/NumPy.
"""

from __future__ import annotations

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op decorator mimicking :func:`numba.njit`."""
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def wrapper(func):  # type: ignore[no-untyped-def]
            return func

        return wrapper
