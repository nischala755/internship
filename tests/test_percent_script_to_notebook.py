import tempfile
import unittest
from pathlib import Path

import nbformat

from tools.percent_script_to_notebook import convert_percent_script


class PercentScriptConverterTests(unittest.TestCase):
    def test_preserves_cell_types_source_and_python_metadata(self):
        source = (
            "# %% [markdown]\n"
            "# # Heading\n"
            "# explanatory text\n"
            "\n"
            "# %%\n"
            "x = 1\n"
            "\n"
            "# %%\n"
            "print(x)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.py"
            dst = Path(tmp) / "output.ipynb"
            src.write_text(source, encoding="utf-8")

            convert_percent_script(src, dst)

            notebook = nbformat.read(dst, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual([cell.cell_type for cell in notebook.cells], ["markdown", "code", "code"])
            self.assertEqual(notebook.cells[0].source, "# Heading\nexplanatory text")
            self.assertEqual(notebook.cells[1].source, "x = 1")
            self.assertEqual(notebook.cells[2].source, "print(x)")
            self.assertEqual(notebook.metadata.kernelspec.language, "python")
            self.assertEqual(notebook.metadata.language_info.name, "python")


if __name__ == "__main__":
    unittest.main()
