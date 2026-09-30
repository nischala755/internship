import unittest
from pathlib import Path

from tools.build_polarization_obb_composite import read_dota_annotations


DATASET = Path(__file__).resolve().parents[2] / "Pol_ISARSpaceTarget1.0"


class AnnotationIntegrityTests(unittest.TestCase):
    def test_capella_boxes_match_source_annotation(self):
        path = DATASET / "ImageSets" / "labelTxt" / "train" / "Capella_el_40_az_0.txt"
        annotations = read_dota_annotations(path)
        self.assertEqual(len(annotations), 10)
        self.assertEqual(
            {item["class_name"] for item in annotations},
            {"parabolic_antenna", "solar_array", "plate_fitting", "polygon_body", "rod_fitting"},
        )
        self.assertTrue(all(item["points"].shape == (4, 2) for item in annotations))

    def test_selected_frames_cover_eight_component_classes(self):
        names = ("Capella_el_40_az_0", "HISEA1_el_45_az_39", "Worldview4_el_40_az_0")
        classes = set()
        for name in names:
            path = DATASET / "ImageSets" / "labelTxt" / "train" / f"{name}.txt"
            classes.update(item["class_name"] for item in read_dota_annotations(path))
        self.assertEqual(len(classes), 8)
        self.assertNotIn("plate_body", classes)


if __name__ == "__main__":
    unittest.main()
