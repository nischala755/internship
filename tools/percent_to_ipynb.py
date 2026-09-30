"""Convert percent-format Python files used in this project to notebooks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


CELL = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def convert(source: Path, destination: Path) -> None:
    cells = []
    kind = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if kind is None:
            lines = []
            return
        if kind == "markdown":
            content = []
            for line in lines:
                if line.startswith("# "):
                    content.append(line[2:])
                elif line.rstrip() == "#":
                    content.append("")
                elif line.startswith("#"):
                    content.append(line[1:].lstrip())
                else:
                    content.append(line)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in content]})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [x + "\n" for x in lines],
                }
            )
        lines = []

    for line in source.read_text(encoding="utf-8").splitlines():
        match = CELL.match(line)
        if match:
            flush()
            kind = "markdown" if match.group(1) else "code"
        else:
            lines.append(line)
    flush()

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": destination.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: percent_to_ipynb.py SOURCE.py DESTINATION.ipynb")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
