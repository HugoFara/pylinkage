"""Root conftest for pytest — disables numba JIT so coverage.py can track JIT'd code."""

import os

# Disable numba JIT so coverage can instrument @njit functions in geometry/,
# solver/, and cam/_numba_core.py. Must be set BEFORE any numba import.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
