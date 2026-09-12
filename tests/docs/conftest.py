"""Run the documentation's code: example scripts and tutorial snippets.

These tests execute what the docs show, so a signature that changes takes
the page that teaches it down with it. They are slow (a few minutes) and
need every optional backend, so they carry the ``docs`` marker and CI runs
them in their own job::

    uv run --extra full --extra cad --extra analysis pytest -m docs

Locally, ``pytest tests/docs`` runs them as well; missing extras show up
as the failure of the page that needs them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TIMEOUT = 900  # seconds per script; the PSO demos take about half a minute


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(str(item.fspath)).parent == Path(__file__).parent:
            item.add_marker(pytest.mark.docs)


def run_script(path: Path, cwd: Path) -> None:
    """Execute *path* headlessly; fail with its output if it does not exit 0.

    Every pylinkage ``DeprecationWarning`` is an error: the docs must teach
    the current name, not one that is on its way out.
    """
    env = dict(os.environ)
    env.update(
        {
            "MPLBACKEND": "Agg",  # plt.show() is then a no-op
            "PLOTLY_RENDERER": "json",  # fig.show() prints instead of opening a browser
            "PYTHONWARNINGS": "error::DeprecationWarning:pylinkage",
            "NUMBA_DISABLE_JIT": "0",  # the root conftest disables it for coverage
        }
    )
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-40:])
        pytest.fail(f"{path.relative_to(REPO)} exited {result.returncode}:\n{tail}")
