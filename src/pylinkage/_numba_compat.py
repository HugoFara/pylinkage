"""Numba compatibility layer.

Provides a no-op ``@njit`` decorator when numba is not installed,
so geometry and solver modules work (slower) with pure Python/NumPy.
"""

from __future__ import annotations

from typing import Any

__all__ = ["njit", "HAS_NUMBA"]

try:
    from numba import njit as njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:  # noqa: E501
        """No-op decorator mimicking :func:`numba.njit`."""
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def wrapper(func: Any) -> Any:
            return func

        return wrapper
