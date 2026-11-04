from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from farmaura_ai.catalog import DATASETS, DATASET_ROOT, dataset_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz"}


def audit_one(dataset_id: str, hash_files: bool = False) -> dict:
    spec = next(item for item in DATASETS if item.dataset_id == dataset_id)
    root = dataset_path(dataset_id)
    images: list[Path] = []
    archives: list[Path] = []
    license_files: list[str] = []
    directory_counts: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    total_bytes = 0
    all_files: list[Path] = []

    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            all_files.append(path)
            total_bytes += path.stat().st_size
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
                relative = path.relative_to(root)
                if relative.parts:
                    directory_counts[relative.parts[0]] += 1
                if hash_files:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    hash_counts[digest.hexdigest()] += 1
            elif path.suffix.lower() in ARCHIVE_EXTENSIONS:
                archives.append(path)
            if path.name.lower() in {"license", "license.txt", "license.md", "readme.md", "readme.txt"}:
                license_files.append(str(path))

    duplicate_groups = sum(1 for count in hash_counts.values() if count > 1)
    partials = [str(p) for p in all_files if p.name.lower().endswith((".partial", ".incomplete", ".kaggle-partial"))]
    archive_rows = []
    verification_path = root / "source_archive_verification.json"
    verification = None
    if verification_path.is_file():
        try:
            verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            verification = None
    for path in archives:
        is_partial = any(marker in path.name.lower() for marker in (".partial", ".incomplete", ".kaggle-partial"))
        is_partial = is_partial or Path(str(path) + ".kaggle-partial").exists()
        integrity_verified = (
            isinstance(verification, dict)
            and verification.get("archive") == path.name
            and verification.get("zip_crc_test") == "passed"
        )
        archive_rows.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "status": "partial_or_in_progress" if is_partial else
                      "integrity_verified" if integrity_verified else "archive_present_unverified",
        })
    if not root.is_dir():
        download_status = "folder_missing"
    elif partials:
        download_status = "download_in_progress_or_partial"
    elif not all_files:
        download_status = "empty_folder"
    elif images and any(row["status"] == "integrity_verified" for row in archive_rows):
        recorded_license = str((verification or {}).get("license_status", "")).casefold()
        license_pending = "verify" in recorded_license or "pending" in recorded_license or not recorded_license
        download_status = (
            "extracted_content_integrity_verified_license_review_pending"
            if license_pending else "extracted_content_integrity_verified"
        )
    elif archives:
        download_status = "archive_present_unverified"
    else:
        download_status = "files_present_unverified"

    return {
        "dataset_id": dataset_id,
        "name": spec.name,
        "task": spec.task,
        "path": str(root),
        "source_url": spec.source_url,
        "license_expected": spec.license,
        "license_note": spec.license_note,
        "license_status_recorded": (verification or {}).get("license_status"),
        "folder_exists": root.is_dir(),
        "download_status": download_status,
        "total_file_bytes": total_bytes,
        "file_count": len(all_files),
        "image_count": len(images),
        "archive_files": archive_rows,
        "archive_verification_record": str(verification_path) if verification_path.is_file() else None,
        "license_and_readme_files": license_files,
        "top_level_image_counts": dict(directory_counts),
        "exact_duplicate_hash_groups": duplicate_groups if hash_files else None,
        "partial_download_files": partials,
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "training_ready": False,
        "training_ready_reason": "Review labels, license, duplicate risk, and group-safe splits before training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory the six FARMAURA dataset folders.")
    parser.add_argument("--dataset", choices=[item.dataset_id for item in DATASETS], help="Audit one dataset.")
    parser.add_argument("--hash-duplicates", action="store_true", help="Hash all images to find exact duplicates; can take time.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    ids = [args.dataset] if args.dataset else [item.dataset_id for item in DATASETS]
    reports = [audit_one(dataset_id, args.hash_duplicates) for dataset_id in ids]
    result = {
        "data_root": str(DATASET_ROOT),
        "datasets": reports,
        "training_gate": "No model training until all chosen sources have verified files, labels and group-safe splits.",
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
