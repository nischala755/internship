"""Paper-aligned elevation partitions for Pol-ISARSpaceTarget-1.0."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image


OFFICIAL_SPLITS = ("train", "valid", "test")
ELEVATIONS = (40, 45, 50)
SATELLITES = ("Capella", "HISEA1", "QPSSAR", "Radarsat2", "StarLink", "Worldview4")
FRAME_RE = re.compile(r"^(?P<satellite>.+)_el_(?P<elevation>\d+)_az_(?P<azimuth>\d+)$")
DUPLICATE_RE = re.compile(r" \(\d+\)$")


@dataclass(frozen=True)
class FrameRecord:
    base: str
    official_split: str
    satellite: str
    elevation: int
    azimuth: int
    label_path: Path
    dat_dir: Path

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["label_path"] = str(self.label_path)
        row["dat_dir"] = str(self.dat_dir)
        return row


def parse_frame_name(name: str) -> tuple[str, int, int]:
    if DUPLICATE_RE.search(name):
        raise ValueError(f"Drive duplicate-suffixed frame is not allowed: {name}")
    match = FRAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Malformed Pol-ISAR frame name: {name}")
    satellite = match.group("satellite")
    elevation = int(match.group("elevation"))
    azimuth = int(match.group("azimuth"))
    if satellite not in SATELLITES:
        raise ValueError(f"Unknown satellite in frame name: {name}")
    if elevation not in ELEVATIONS:
        raise ValueError(f"Unsupported elevation in frame name: {name}")
    if not 0 <= azimuth <= 360:
        raise ValueError(f"Invalid azimuth in frame name: {name}")
    return satellite, elevation, azimuth


def discover_frames(dataset_root: Path) -> list[FrameRecord]:
    root = Path(dataset_root)
    records: list[FrameRecord] = []
    for split in OFFICIAL_SPLITS:
        label_dir = root / "ImageSets" / "labelTxt" / split
        dat_dir = root / "ImageSets" / "images" / split
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")
        if not dat_dir.exists():
            raise FileNotFoundError(f"Missing complex-image directory: {dat_dir}")
        for label_path in sorted(label_dir.glob("*.txt")):
            base = label_path.stem
            if DUPLICATE_RE.search(base):
                continue
            satellite, elevation, azimuth = parse_frame_name(base)
            records.append(FrameRecord(
                base=base,
                official_split=split,
                satellite=satellite,
                elevation=elevation,
                azimuth=azimuth,
                label_path=label_path,
                dat_dir=dat_dir,
            ))
    return records


def _sort_records(records: Iterable[FrameRecord]) -> list[FrameRecord]:
    return sorted(records, key=lambda row: (row.satellite, row.elevation, row.azimuth, row.base))


def partition_records(records: Sequence[FrameRecord]) -> dict[str, dict[str, list[FrameRecord]]]:
    owners: dict[str, str] = {}
    for record in records:
        previous = owners.get(record.base)
        if previous is not None:
            raise ValueError(
                f"Official split overlap for {record.base}: present in both {previous} and {record.official_split}"
            )
        if record.official_split not in OFFICIAL_SPLITS:
            raise ValueError(f"Unknown official split for {record.base}: {record.official_split}")
        owners[record.base] = record.official_split

    partitions: dict[str, dict[str, list[FrameRecord]]] = {}
    for elevation in ELEVATIONS:
        name = f"D{elevation}"
        partitions[name] = {
            split: _sort_records(
                row for row in records
                if row.official_split == split and row.elevation == elevation
            )
            for split in OFFICIAL_SPLITS
        }
    partitions["Dall"] = {
        split: _sort_records(row for row in records if row.official_split == split)
        for split in OFFICIAL_SPLITS
    }
    return partitions


def _bases(rows: Sequence[FrameRecord]) -> set[str]:
    return {row.base for row in rows}


def audit_partitions(
    partitions: Mapping[str, Mapping[str, Sequence[FrameRecord]]],
    *,
    strict_counts: bool = True,
) -> dict[str, object]:
    expected_names = {"D40", "D45", "D50", "Dall"}
    if set(partitions) != expected_names:
        raise ValueError(f"Partition names differ from expected {sorted(expected_names)}")

    for name, split_map in partitions.items():
        if set(split_map) != set(OFFICIAL_SPLITS):
            raise ValueError(f"{name} must contain train, valid and test")
        split_bases = {split: _bases(split_map[split]) for split in OFFICIAL_SPLITS}
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = split_bases[left] & split_bases[right]
            if overlap:
                raise ValueError(f"{name} split overlap between {left} and {right}: {sorted(overlap)[:3]}")
        if name != "Dall":
            elevation = int(name[1:])
            if any(row.elevation != elevation for split in OFFICIAL_SPLITS for row in split_map[split]):
                raise ValueError(f"{name} contains a frame from a different elevation")

    for split in OFFICIAL_SPLITS:
        expected_union = set().union(*(_bases(partitions[f"D{elevation}"][split]) for elevation in ELEVATIONS))
        actual = _bases(partitions["Dall"][split])
        if actual != expected_union:
            raise ValueError(f"Dall {split} is not the exact union of D40, D45 and D50")

    if strict_counts:
        expected_totals = {
            "D40": {"train": 438, "valid": 54, "test": 54},
            "D45": {"train": 438, "valid": 54, "test": 54},
            "D50": {"train": 438, "valid": 54, "test": 54},
            "Dall": {"train": 1314, "valid": 162, "test": 162},
        }
        per_satellite = {"train": 73, "valid": 9, "test": 9}
        for name, expected in expected_totals.items():
            for split, expected_count in expected.items():
                actual_count = len(partitions[name][split])
                if actual_count != expected_count:
                    raise ValueError(f"{name} {split}: expected {expected_count} frames, found {actual_count}")
        for elevation in ELEVATIONS:
            name = f"D{elevation}"
            for split, expected_count in per_satellite.items():
                for satellite in SATELLITES:
                    actual = sum(row.satellite == satellite for row in partitions[name][split])
                    if actual != expected_count:
                        raise ValueError(
                            f"{name} {split} {satellite}: expected {expected_count} frames, found {actual}"
                        )

    all_rows = [row for split in OFFICIAL_SPLITS for row in partitions["Dall"][split]]
    return {
        "total_unique_frames": len(_bases(all_rows)),
        "counts": {
            name: {split: len(split_map[split]) for split in OFFICIAL_SPLITS}
            for name, split_map in partitions.items()
        },
        "satellites": sorted({row.satellite for row in all_rows}),
        "elevations": sorted({row.elevation for row in all_rows}),
    }


def write_partition_manifests(
    records: Sequence[FrameRecord],
    partitions: Mapping[str, Mapping[str, Sequence[FrameRecord]]],
    output_dir: Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["base", "official_split", "satellite", "elevation", "azimuth", "label_path", "dat_dir"]

    def write_csv(path: Path, rows: Sequence[FrameRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in rows)

    write_csv(output / "master_frames.csv", _sort_records(records))
    for name, split_map in partitions.items():
        for split in OFFICIAL_SPLITS:
            write_csv(output / f"{name}_{split}.csv", list(split_map[split]))
    (output / "partition_summary.json").write_text(
        json.dumps(audit_partitions(partitions, strict_counts=False), indent=2), encoding="utf-8"
    )


def read_dota_rows(path: Path) -> list[tuple[list[float], str, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing DOTA annotation: {path}")
    rows: list[tuple[list[float], str, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"{path}:{line_number}: expected 10 DOTA fields, found {len(parts)}")
        try:
            coordinates = [float(value) for value in parts[:8]]
            difficult_value = float(parts[9])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric DOTA value") from exc
        if not all(math.isfinite(value) for value in coordinates + [difficult_value]):
            raise ValueError(f"{path}:{line_number}: non-finite DOTA value")
        difficult = int(difficult_value)
        if difficult_value != difficult or difficult not in (0, 1):
            raise ValueError(f"{path}:{line_number}: difficult must be 0 or 1")
        rows.append((coordinates, parts[8], difficult))
    return rows


def convert_dota_to_yolo_obb(
    src: Path,
    dst: Path,
    class_to_id: Mapping[str, int],
    width: int = 256,
    height: int = 256,
) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    lines: list[str] = []
    for coordinates, class_name, difficult in read_dota_rows(src):
        if class_name not in class_to_id:
            raise ValueError(f"{src}: unknown component class {class_name!r}")
        if difficult == 1:
            continue
        normalized = [
            min(1.0, max(0.0, value / (width if index % 2 == 0 else height)))
            for index, value in enumerate(coordinates)
        ]
        lines.append(" ".join([str(class_to_id[class_name]), *[f"{value:.6f}" for value in normalized]]))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _robust_uint8(magnitude: np.ndarray) -> np.ndarray:
    values = np.log1p(np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0))
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1e-6
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def load_single_vv(
    record: FrameRecord,
    tiff_root: Path,
    *,
    dat_shape: tuple[int, int] = (256, 256),
) -> np.ndarray:
    if DUPLICATE_RE.search(record.base):
        raise ValueError(f"Drive duplicate-suffixed frame is not allowed: {record.base}")
    tiff_path = Path(tiff_root) / f"{record.base}_VV.tiff"
    if tiff_path.exists():
        gray = np.asarray(Image.open(tiff_path).convert("L"), dtype=np.uint8)
    else:
        dat_path = record.dat_dir / f"{record.base}_VV.dat"
        if not dat_path.exists():
            raise FileNotFoundError(
                f"Missing VV source for {record.base} in {record.official_split}: "
                f"checked {tiff_path} and {dat_path}"
            )
        raw = np.fromfile(dat_path, dtype=np.float32)
        expected = 2 * dat_shape[0] * dat_shape[1]
        if raw.size != expected:
            raise ValueError(f"{record.base}: expected {expected} float32 DAT values, found {raw.size}")
        complex_image = raw[0::2].reshape(dat_shape) + 1j * raw[1::2].reshape(dat_shape)
        gray = _robust_uint8(np.abs(complex_image))
    return np.repeat(gray[:, :, None], 3, axis=2)


def write_yolo_yaml(dataset_dir: Path, class_to_id: Mapping[str, int]) -> Path:
    ids = sorted(class_to_id.values())
    if ids != list(range(len(ids))):
        raise ValueError("Class IDs must be contiguous and start at zero")
    names = {class_id: name for name, class_id in sorted(class_to_id.items(), key=lambda item: item[1])}
    data = {
        "path": str(Path(dataset_dir)),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": names,
    }
    yaml_path = Path(dataset_dir) / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def _build_request(
    partition_name: str,
    split_map: Mapping[str, Sequence[FrameRecord]],
    class_to_id: Mapping[str, int],
) -> dict[str, object]:
    return {
        "build_version": 1,
        "partition": partition_name,
        "class_to_id": dict(sorted(class_to_id.items())),
        "records": {
            split: [row.base for row in _sort_records(split_map[split])]
            for split in OFFICIAL_SPLITS
        },
        "counts": {split: len(split_map[split]) for split in OFFICIAL_SPLITS},
    }


def build_yolo_partition(
    partition_name: str,
    split_map: Mapping[str, Sequence[FrameRecord]],
    output_root: Path,
    tiff_root: Path,
    class_to_id: Mapping[str, int],
    *,
    dat_shape: tuple[int, int] = (256, 256),
    force_rebuild: bool = False,
) -> Path:
    if set(split_map) != set(OFFICIAL_SPLITS):
        raise ValueError(f"{partition_name} must contain train, valid and test")
    dataset_dir = Path(output_root) / partition_name
    completion_path = dataset_dir / "build_complete.json"
    request = _build_request(partition_name, split_map, class_to_id)
    if not force_rebuild and completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        if existing == request and (dataset_dir / "data.yaml").exists():
            return dataset_dir / "data.yaml"

    # Validate every source before replacing a previously usable build.
    for split in OFFICIAL_SPLITS:
        for record in split_map[split]:
            if record.official_split != split:
                raise ValueError(f"{record.base}: record split {record.official_split} does not match {split}")
            read_dota_rows(record.label_path)
            tiff_path = Path(tiff_root) / f"{record.base}_VV.tiff"
            dat_path = record.dat_dir / f"{record.base}_VV.dat"
            if not tiff_path.exists() and not dat_path.exists():
                raise FileNotFoundError(f"Missing VV source for {record.base} in {split}")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in OFFICIAL_SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for record in _sort_records(split_map[split]):
            image = load_single_vv(record, tiff_root, dat_shape=dat_shape)
            Image.fromarray(image).save(dataset_dir / "images" / split / f"{record.base}.png")
            convert_dota_to_yolo_obb(
                record.label_path,
                dataset_dir / "labels" / split / f"{record.base}.txt",
                class_to_id,
                width=image.shape[1],
                height=image.shape[0],
            )
    yaml_path = write_yolo_yaml(dataset_dir, class_to_id)
    completion_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    return yaml_path
