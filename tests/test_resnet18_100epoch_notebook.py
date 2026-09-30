import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "POLISAR_ResNet18_100Epoch_Rigorous.ipynb"


class ResNet18NotebookTests(unittest.TestCase):
    def test_notebook_has_reproducible_leakage_safe_protocol(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(cell.source for cell in notebook.cells)

        required_fragments = (
            "epochs: int = 100",
            "expected_train: int = 1314",
            "expected_valid: int = 162",
            "expected_test: int = 162",
            "force_remount=True",
            "resolve_dataset_root",
            "ImageSets/images/train",
            "Available entries under MyDrive",
            "np.log1p",
            "validate_npy_array",
            'with_suffix(destination.suffix + ".copying")',
            "Corrupt persistent cache detected",
            "assert_no_split_overlap",
            "compute_train_stats",
            "weights=None",
            "nn.Conv2d(4, 64",
            "nn.Linear(model.fc.in_features, len(CLASS_NAMES))",
            "AdamW",
            "CosineAnnealingLR",
            "label_smoothing=cfg.label_smoothing",
            "best_validation_macro_f1.pt",
            "last_checkpoint.pt",
            "classification_report",
            "confusion_matrix",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)

    def test_test_split_is_not_used_during_model_selection(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        training_section = source.split("## Final one-time test evaluation", 1)[0]
        self.assertNotIn('run_epoch(test_loader', training_section)
        self.assertIn('run_epoch(test_loader', source)


if __name__ == "__main__":
    unittest.main()
