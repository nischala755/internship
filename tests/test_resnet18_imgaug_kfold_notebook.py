import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "POLISAR_ResNet18_ImgAug_Shuffled_5Fold.ipynb"


class ResNet18ImgAugKFoldNotebookTests(unittest.TestCase):
    def test_notebook_contains_requested_controlled_experiment(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(cell.source for cell in notebook.cells)
        required = (
            "imgaug==0.4.0",
            "import imgaug as ia",
            "from imgaug import augmenters as iaa",
            "epochs: int = 100",
            "n_splits: int = 5",
            'REGIMES = ("no_augmentation", "imgaug")',
            "StratifiedGroupKFold(",
            "shuffle=True",
            "random_state=cfg.seed",
            'groups=development_manifest["azimuth"]',
            "compute_fold_train_stats",
            "assert train_groups.isdisjoint(validation_groups)",
            "best_validation_macro_f1.pt",
            "last_checkpoint.pt",
        'train_generator.set_state(saved["train_generator_state"].cpu())',
            "ensemble_probabilities",
            "cross_validation_summary.csv",
            "augmentation_comparison.png",
        )
        for fragment in required:
            self.assertIn(fragment, source)

    def test_official_test_is_reserved_for_final_ensembles(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        training, final = source.split("## Final ensemble evaluation on the untouched official test set", 1)
        self.assertNotIn("evaluate_ensemble_on_test(", training)
        self.assertIn("evaluate_ensemble_on_test(", final)


if __name__ == "__main__":
    unittest.main()
