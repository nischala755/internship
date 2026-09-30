"""Convert a Jupytext-style percent Python script into a Colab notebook."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat


CELL_MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def _source(lines: list[str], markdown: bool) -> str:
    if markdown:
        cleaned = []
        for line in lines:
            if line.startswith("# "):
                cleaned.append(line[2:])
            elif line == "#":
                cleaned.append("")
            else:
                cleaned.append(line)
        lines = cleaned
    return "\n".join(lines).strip("\n")


def convert_percent_script(source_path: Path, notebook_path: Path) -> Path:
    cells = []
    current: list[str] = []
    current_type: str | None = None

    def flush() -> None:
        nonlocal current
        if current_type is None:
            return
        source = _source(current, current_type == "markdown")
        factory = nbformat.v4.new_markdown_cell if current_type == "markdown" else nbformat.v4.new_code_cell
        cells.append(factory(source=source))
        current = []

    for line in Path(source_path).read_text(encoding="utf-8").splitlines():
        marker = CELL_MARKER.fullmatch(line)
        if marker:
            flush()
            current_type = "markdown" if marker.group(1) else "code"
        elif current_type is not None:
            current.append(line)
        elif line.strip():
            raise ValueError(f"Content appears before the first percent cell marker: {line!r}")
    flush()

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "colab": {"provenance": []},
        },
    )
    notebook_path = Path(notebook_path)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, notebook_path)
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert_percent_script(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
