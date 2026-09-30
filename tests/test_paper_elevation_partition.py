import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.paper_elevation_partition import (
    FrameRecord,
    audit_partitions,
    build_yolo_partition,
    convert_dota_to_yolo_obb,
    discover_frames,
    load_single_vv,
    parse_frame_name,
    partition_records,
    read_dota_rows,
)


SATELLITES = ("Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4")


def make_record(base: str, split: str) -> FrameRecord:
    satellite, elevation, azimuth = parse_frame_name(base)
    return FrameRecord(
        base=base,
        official_split=split,
        satellite=satellite,
        elevation=elevation,
        azimuth=azimuth,
        label_path=Path(f"/{split}/{base}.txt"),
        dat_dir=Path(f"/{split}"),
    )


class FilenameParsingTests(unittest.TestCase):
    def test_parse_frame_name(self):
        self.assertEqual(parse_frame_name("Capella_el_40_az_0"), ("Capella", 40, 0))
        self.assertEqual(parse_frame_name("Worldview4_el_50_az_90"), ("Worldview4", 50, 90))
        self.assertEqual(parse_frame_name("HISEA1_el_45_az_7"), ("HISEA1", 45, 7))

    def test_rejects_duplicate_suffix_and_malformed_name(self):
        for name in ("Capella_el_40_az_0 (1)", "Capella_40_0", "Capella_el_41_az_0"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_frame_name(name)


class DiscoveryTests(unittest.TestCase):
    def test_discovers_each_official_label_once_and_ignores_drive_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, base in (
                ("train", "Capella_el_40_az_0"),
                ("valid", "HISEA1_el_45_az_1"),
                ("test", "Worldview4_el_50_az_2"),
            ):
                label_dir = root / "ImageSets" / "labelTxt" / split
                dat_dir = root / "ImageSets" / "images" / split
                label_dir.mkdir(parents=True)
                dat_dir.mkdir(parents=True)
                (label_dir / f"{base}.txt").write_text("", encoding="utf-8")
            duplicate = root / "ImageSets" / "labelTxt" / "train" / "Capella_el_40_az_0 (1).txt"
            duplicate.write_text("", encoding="utf-8")

            rows = discover_frames(root)

            self.assertEqual([row.base for row in rows], [
                "Capella_el_40_az_0", "HISEA1_el_45_az_1", "Worldview4_el_50_az_2"
            ])
            self.assertEqual({row.official_split for row in rows}, {"train", "valid", "test"})


class PartitionAuditTests(unittest.TestCase):
    def make_small_records(self):
        records = []
        split_az = {"train": 0, "valid": 1, "test": 2}
        for split, azimuth in split_az.items():
            for elevation in (40, 45, 50):
                for satellite in SATELLITES:
                    records.append(make_record(f"{satellite}_el_{elevation}_az_{azimuth}", split))
        return records

    def test_partition_elevation_purity_and_dall_union(self):
        records = self.make_small_records()
        partitions = partition_records(records)
        summary = audit_partitions(partitions, strict_counts=False)

        self.assertEqual(set(partitions), {"D40", "D45", "D50", "Dall"})
        self.assertTrue(all(row.elevation == 40 for row in partitions["D40"]["train"]))
        self.assertEqual(len(partitions["Dall"]["test"]), 18)
        self.assertEqual(summary["total_unique_frames"], 54)

    def test_overlap_between_official_splits_fails(self):
        records = self.make_small_records()
        records.append(make_record("Capella_el_40_az_0", "test"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            partition_records(records)

    def test_modified_dall_union_fails(self):
        partitions = partition_records(self.make_small_records())
        partitions["Dall"]["test"] = partitions["Dall"]["test"][:-1]
        with self.assertRaisesRegex(ValueError, "Dall"):
            audit_partitions(partitions, strict_counts=False)

    def test_strict_counts_fail_for_synthetic_dataset(self):
        partitions = partition_records(self.make_small_records())
        with self.assertRaisesRegex(ValueError, "expected"):
            audit_partitions(partitions, strict_counts=True)


class DotaConversionTests(unittest.TestCase):
    def test_valid_row_is_normalized_and_difficult_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            dst = Path(tmp) / "dst.txt"
            src.write_text(
                "0 0 256 0 256 256 0 256 body 0\n"
                "10 10 20 10 20 20 10 20 panel 1\n",
                encoding="utf-8",
            )
            count = convert_dota_to_yolo_obb(src, dst, {"body": 0, "panel": 1})
            self.assertEqual(count, 1)
            self.assertEqual(dst.read_text(encoding="utf-8"), "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.000000 1.000000\n")

    def test_empty_or_difficult_only_annotation_writes_empty_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            for content in ("", "0 0 1 0 1 1 0 1 body 1\n"):
                src = Path(tmp) / "src.txt"
                dst = Path(tmp) / "dst.txt"
                src.write_text(content, encoding="utf-8")
                self.assertEqual(convert_dota_to_yolo_obb(src, dst, {"body": 0}), 0)
                self.assertEqual(dst.read_text(encoding="utf-8"), "")

    def test_unknown_class_and_malformed_rows_fail(self):
        bad_rows = (
            "0 0 1 0 1 1 0 1 unknown 0\n",
            "0 0 1 0 1 1 0 body 0\n",
            "0 0 nan 0 1 1 0 1 body 0\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, content in enumerate(bad_rows):
                src = Path(tmp) / f"bad{index}.txt"
                dst = Path(tmp) / f"bad{index}.out"
                src.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    convert_dota_to_yolo_obb(src, dst, {"body": 0})

    def test_out_of_bounds_finite_coordinates_are_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            dst = Path(tmp) / "dst.txt"
            src.write_text("-10 0 300 0 300 256 -10 256 body 0\n", encoding="utf-8")
            convert_dota_to_yolo_obb(src, dst, {"body": 0})
            values = [float(value) for value in dst.read_text().split()[1:]]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))


class SingleVvTests(unittest.TestCase):
    def make_record(self, root: Path, base: str = "Capella_el_40_az_0") -> FrameRecord:
        label = root / f"{base}.txt"
        label.write_text("", encoding="utf-8")
        return FrameRecord(base, "train", "Capella", 40, 0, label, root)

    def test_tiff_is_preferred_over_dat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self.make_record(root)
            tiff_root = root / "tiff"
            tiff_root.mkdir()
            Image.fromarray(np.full((2, 2), 77, dtype=np.uint8)).save(tiff_root / f"{record.base}_VV.tiff")
            np.zeros(8, dtype=np.float32).tofile(root / f"{record.base}_VV.dat")
            image = load_single_vv(record, tiff_root, dat_shape=(2, 2))
            self.assertEqual(image.shape, (2, 2, 3))
            self.assertTrue(np.all(image == 77))

    def test_dat_fallback_builds_three_channel_uint8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self.make_record(root)
            raw = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=np.float32)
            raw.tofile(root / f"{record.base}_VV.dat")
            image = load_single_vv(record, root / "missing", dat_shape=(2, 2))
            self.assertEqual(image.shape, (2, 2, 3))
            self.assertEqual(image.dtype, np.uint8)
            np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])

    def test_missing_source_and_bad_dat_size_fail_with_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = self.make_record(root)
            with self.assertRaisesRegex(FileNotFoundError, record.base):
                load_single_vv(record, root / "missing", dat_shape=(2, 2))
            np.zeros(6, dtype=np.float32).tofile(root / f"{record.base}_VV.dat")
            with self.assertRaisesRegex(ValueError, record.base):
                load_single_vv(record, root / "missing", dat_shape=(2, 2))


class DatasetBuilderTests(unittest.TestCase):
    def test_streamed_build_writes_exact_counts_yaml_and_completion_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiff_root = root / "tiff"
            tiff_root.mkdir()
            split_map = {}
            for split, azimuth in (("train", 0), ("valid", 1), ("test", 2)):
                source = root / split
                source.mkdir()
                base = f"Capella_el_40_az_{azimuth}"
                label = source / f"{base}.txt"
                label.write_text("0 0 2 0 2 2 0 2 body 0\n", encoding="utf-8")
                record = FrameRecord(base, split, "Capella", 40, azimuth, label, source)
                Image.fromarray(np.full((2, 2), azimuth + 1, dtype=np.uint8)).save(tiff_root / f"{base}_VV.tiff")
                split_map[split] = [record]

            yaml_path = build_yolo_partition("D40", split_map, root / "out", tiff_root, {"body": 0}, dat_shape=(2, 2))
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self.assertEqual(data["train"], "images/train")
            self.assertEqual(data["val"], "images/valid")
            self.assertEqual(data["test"], "images/test")
            self.assertEqual(data["names"], {0: "body"})
            dataset_dir = yaml_path.parent
            self.assertEqual(len(list(dataset_dir.glob("images/*/*.png"))), 3)
            self.assertEqual(len(list(dataset_dir.glob("labels/*/*.txt"))), 3)
            self.assertTrue((dataset_dir / "build_complete.json").exists())

    def test_missing_source_never_writes_completion_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train"
            source.mkdir()
            record = FrameRecord("Capella_el_40_az_0", "train", "Capella", 40, 0, source / "missing.txt", source)
            splits = {"train": [record], "valid": [], "test": []}
            with self.assertRaises(FileNotFoundError):
                build_yolo_partition("D40", splits, root / "out", root / "tiff", {"body": 0}, dat_shape=(2, 2))
            self.assertFalse((root / "out" / "D40" / "build_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
