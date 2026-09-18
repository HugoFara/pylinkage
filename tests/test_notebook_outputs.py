"""The committed tutorial notebooks must not carry outputs from a private machine.

A warning printed while a notebook was executed embeds the path of the file
that raised it, and a path under a home directory then ships with the
notebook to GitHub and the rendered docs.
"""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "docs" / "notebooks").glob("*.ipynb"))

# A home directory on Linux, macOS or Windows.
LOCAL_PATH = re.compile(r"(/home/|/Users/|[A-Za-z]:\\+Users\\)")


def _output_texts(cell: dict) -> list[str]:
    texts = []
    for output in cell.get("outputs", ()):
        if "text" in output:
            texts.append("".join(output["text"]))
        for mime, payload in output.get("data", {}).items():
            if mime.startswith("text/"):
                texts.append("".join(payload))
        if output.get("output_type") == "error":
            texts.append("\n".join(output.get("traceback", ())))
    return texts


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_outputs_do_not_name_a_local_path(notebook: Path) -> None:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    leaks = [
        f"cell {index}: {line}"
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        for text in _output_texts(cell)
        for line in text.splitlines()
        if LOCAL_PATH.search(line)
    ]
    assert not leaks, "\n".join(leaks)
