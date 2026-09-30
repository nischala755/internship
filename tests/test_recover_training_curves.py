import unittest

from tools.recover_training_curves import (
    parse_epoch_loss,
    parse_nonyolo_epochs,
    parse_resnet_history,
    parse_yolo_epochs,
    f1_from_pr,
)
from tools.plot_obb_loss_from_results_csv import read_loss_csv
from io import StringIO


class RecoveryTests(unittest.TestCase):
    def test_last_batch_is_epoch_loss(self):
        log = (
            "fold_1 epoch 1 train [0001/0010] loss=2.0000 acc=0.125\n"
            "fold_1 epoch 1 train [0010/0010] loss=1.2500 acc=0.500\n"
            "fold_1 epoch 1 valid [0002/0002] loss=1.1000 acc=0.750\n"
            "fold_1 epoch 2 train [0010/0010] loss=0.8000 acc=0.875\n"
            "fold_1 epoch 2 valid [0002/0002] loss=0.9000 acc=0.875\n"
        )
        rows = parse_epoch_loss(log, "fold_1")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"epoch": 1, "train_loss": 1.25, "valid_loss": 1.1})

    def test_yolo_metrics_do_not_become_validation_loss(self):
        log = (
            "Starting training for 2 epochs...\n"
            "1/2 0.5G 3.02 3.488 2.373 0.2186 34 256: 100%\n"
            "all 162 1613 0.267 0.372 0.247 0.14\n"
            "2/2 0.5G 2.455 2.007 1.884 0.1471 18 256: 100%\n"
            "all 162 1613 0.725 0.455 0.477 0.256\n"
        )
        rows = parse_yolo_epochs(log)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["train_box_loss"], 3.02)
        self.assertEqual(rows[1]["valid_precision"], 0.725)
        self.assertNotIn("valid_loss", rows[0])
        self.assertAlmostEqual(rows[0]["valid_f1_harmonic"], f1_from_pr(.267, .372))

    def test_ultralytics_csv_train_and_validation_loss(self):
        csv_text = StringIO(
            "epoch, train/box_loss, val/box_loss,train/cls_loss,val/cls_loss\n"
            "0,3.1,2.9,2.2,2.1\n1,2.5,2.4,1.8,1.7\n"
        )
        rows = read_loss_csv(csv_text)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[1]["train/box_loss"], 2.5)
        self.assertEqual(rows[1]["val/box_loss"], 2.4)

    def test_nonyolo_epoch_record_has_no_invented_validation_loss(self):
        log = "{'method': 'CenterNet_OBB', 'epoch': 1, 'loss': 0.9, 'precision': 0.1, 'recall': 0.2, 'f1': 0.133}\n"
        rows = parse_nonyolo_epochs(log)
        self.assertEqual(rows["CenterNet_OBB"][0]["train_loss"], .9)
        self.assertNotIn("valid_loss", rows["CenterNet_OBB"][0])

    def test_resnet_history_uses_recorded_accuracy_and_f1(self):
        log = '''
{
  "epoch": 1,
  "train_loss": 0.5,
  "train_accuracy": 0.79,
  "train_f1_macro": 0.78,
  "valid_loss": 7.1,
  "valid_accuracy": 0.32,
  "valid_f1_macro": 0.23,
  "seconds": 10.0
}
'''
        rows = parse_resnet_history(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["train_accuracy"], 0.79)
        self.assertEqual(rows[0]["valid_f1_macro"], 0.23)


if __name__ == "__main__":
    unittest.main()
