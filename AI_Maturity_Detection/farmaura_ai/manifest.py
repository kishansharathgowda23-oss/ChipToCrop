from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"image_path", "label", "group_id", "split"}
VALID_SPLITS = {"train", "validation", "test"}


class ManifestError(ValueError):
    pass


def read_and_validate_manifest(csv_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    """Require group-safe splits; a physical sample cannot cross train/val/test."""
    if not csv_path.is_file():
        raise ManifestError(f"Split manifest not found: {csv_path}")

    root = dataset_root.resolve()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"Missing manifest columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ManifestError("Split manifest has no rows.")

    group_splits: dict[str, set[str]] = defaultdict(set)
    split_labels: dict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        for key in REQUIRED_COLUMNS:
            row[key] = (row.get(key) or "").strip()
        if not row["image_path"] or not row["label"] or not row["group_id"]:
            raise ManifestError(f"Row {line_number}: image_path, label and group_id are required.")
        if row["split"] not in VALID_SPLITS:
            raise ManifestError(f"Row {line_number}: split must be one of {sorted(VALID_SPLITS)}.")

        image_path = (root / row["image_path"]).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"Row {line_number}: image path escapes the dataset folder.") from exc
        if not image_path.is_file():
            raise ManifestError(f"Row {line_number}: image file is missing: {row['image_path']}")

        group_splits[row["group_id"]].add(row["split"])
        split_labels[row["split"]].add(row["label"])

    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked_groups:
        raise ManifestError(f"Group leakage across splits: {leaked_groups[:10]}")
    missing_splits = VALID_SPLITS - {row["split"] for row in rows}
    if missing_splits:
        raise ManifestError(f"Manifest must contain train, validation and test rows; missing {sorted(missing_splits)}.")

    train_labels = split_labels["train"]
    if not train_labels:
        raise ManifestError("Training split has no labels.")
    return rows
