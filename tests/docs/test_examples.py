"""Every script under docs/examples/ runs to completion."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import REPO, run_script

EXAMPLES = sorted((REPO / "docs" / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(script: Path, tmp_path: Path) -> None:
    run_script(script, cwd=tmp_path)
