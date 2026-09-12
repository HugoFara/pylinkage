"""Every tutorial page executes, code block after code block.

The ``.. code-block:: python`` blocks of a page are concatenated in order
into one script, so a page reads as a session: later blocks may use names
defined by earlier ones, and every block must be real, runnable code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import REPO, run_script

TUTORIALS = sorted((REPO / "docs" / "source" / "tutorials").glob("*.rst"))

# ".. code-block:: python", optional ":option:" lines, a blank line, then
# every following line that is indented or empty.
_BLOCK = re.compile(
    r"\.\. code-block:: python\n(?:[ \t]+:[^\n]*\n)*\n((?:(?:[ \t]+[^\n]*)?\n)+)"
)


def python_blocks(text: str) -> list[str]:
    """Return the page's python code blocks, dedented."""
    blocks = []
    for match in _BLOCK.finditer(text):
        lines = match.group(1).rstrip("\n").split("\n")
        indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        blocks.append("\n".join(line[indent:] for line in lines) + "\n")
    return blocks


@pytest.mark.parametrize("page", TUTORIALS, ids=[p.stem for p in TUTORIALS])
def test_tutorial_runs(page: Path, tmp_path: Path) -> None:
    blocks = python_blocks(page.read_text())
    if not blocks:
        pytest.skip("no python code blocks")
    script = tmp_path / f"{page.stem}.py"
    script.write_text("\n\n# ---- next block ----\n".join(blocks))
    run_script(script, cwd=tmp_path)
