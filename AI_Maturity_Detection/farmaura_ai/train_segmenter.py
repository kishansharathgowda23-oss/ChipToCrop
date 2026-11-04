from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from farmaura_ai.catalog import dataset_path
from farmaura_ai.vision_models import build_deeplabv3, build_unet


def read_mask_manifest(csv_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    required = {"image_path", "mask_path", "group_id", "split"}
    splits = {"train", "validation", "test"}
    root = dataset_root.resolve()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Mask manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Mask manifest has no rows.")
    group_splits: dict[str, set[str]] = defaultdict(set)
    present_splits = set()
    for line, row in enumerate(rows, 2):
        row = {key: (row.get(key) or "").strip() for key in required}
        if not all(row[key] for key in required):
            raise ValueError(f"Row {line}: image_path, mask_path, group_id and split are required.")
        if row["split"] not in splits:
            raise ValueError(f"Row {line}: split must be one of {sorted(splits)}.")
        for field in ("image_path", "mask_path"):
            path = (root / row[field]).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Row {line}: {field} escapes the dataset folder.") from exc
            if not path.is_file():
                raise ValueError(f"Row {line}: file not found: {row[field]}")
        group_splits[row["group_id"]].add(row["split"])
        present_splits.add(row["split"])
        rows[line - 2] = row
    leaked = [group for group, values in group_splits.items() if len(values) > 1]
    if leaked:
        raise ValueError(f"Sample groups cross splits: {leaked[:10]}")
    if present_splits != splits:
        raise ValueError(f"Manifest must contain train, validation and test rows; found {sorted(present_splits)}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-Net or DeepLabV3 on reviewed, class-index mask files."
    )
    parser.add_argument("--dataset", default="smartharvest", choices=("smartharvest",))
    parser.add_argument("--manifest", required=True, type=Path,
                        help="CSV columns: image_path,mask_path,group_id,split.")
    parser.add_argument("--model", choices=("unet", "deeplabv3"), default="unet")
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--output", type=Path, default=Path("artifacts/segmentation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.num_classes, args.image_size, args.epochs, args.batch_size) < 1:
        raise SystemExit("num-classes, image-size, epochs and batch-size must be positive.")

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    root = dataset_path(args.dataset)
    rows = read_mask_manifest(args.manifest, root)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class MaskDataset(Dataset):
        def __init__(self, split: str, training: bool = False) -> None:
            self.rows = [row for row in rows if row["split"] == split]
            if not self.rows:
                raise ValueError(f"No {split} rows in mask manifest.")
            self.training = training

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(root / row["image_path"]) as source:
                image = source.convert("RGB").resize(
                    (args.image_size, args.image_size), Image.Resampling.BILINEAR
                )
            with Image.open(root / row["mask_path"]) as source:
                mask = source.convert("L").resize(
                    (args.image_size, args.image_size), Image.Resampling.NEAREST
                )
            if self.training and random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            target = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
            if int(target.max()) >= args.num_classes:
                raise ValueError(f"Mask class {int(target.max())} is outside 0..{args.num_classes - 1}.")
            return image_transform(image), target

    datasets = {
        "train": MaskDataset("train", True),
        "validation": MaskDataset("validation"),
        "test": MaskDataset("test"),
    }
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.workers, pin_memory=device.type == "cuda")
        for split, ds in datasets.items()
    }
    model = (build_unet(args.num_classes) if args.model == "unet"
             else build_deeplabv3(args.num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / f"smartharvest_{args.model}_best.pt"
    best_miou = -1.0
    history = []

    def logits_for(batch):
        output = model(batch)
        return output["out"] if isinstance(output, dict) else output

    def evaluate(loader):
        model.eval()
        confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
        with torch.inference_mode():
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                predictions = logits_for(images).argmax(dim=1)
                encoded = targets.reshape(-1) * args.num_classes + predictions.reshape(-1)
                confusion += torch.bincount(encoded, minlength=args.num_classes ** 2).reshape(
                    args.num_classes, args.num_classes
                ).cpu()
        true_counts = confusion.sum(dim=1).float()
        pred_counts = confusion.sum(dim=0).float()
        intersections = confusion.diag().float()
        unions = true_counts + pred_counts - intersections
        present = unions > 0
        ious = intersections[present] / unions[present]
        dices = (2 * intersections[present]) / (true_counts[present] + pred_counts[present]).clamp_min(1)
        return {
            "mean_iou": float(ious.mean().item()) if len(ious) else 0.0,
            "mean_dice": float(dices.mean().item()) if len(dices) else 0.0,
            "per_class_iou": {
                str(index): float(intersections[index] / unions[index]) if unions[index] > 0 else None
                for index in range(args.num_classes)
            },
            "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        }

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits_for(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        validation = evaluate(loaders["validation"])
        epoch_report = {"epoch": epoch, "train_loss": total_loss / seen, "validation": validation}
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        if validation["mean_iou"] > best_miou:
            best_miou = validation["mean_iou"]
            torch.save({"state_dict": model.state_dict(), "model_name": args.model,
                        "num_classes": args.num_classes, "dataset_id": args.dataset,
                        "model_version": "0.1.0", "image_size": args.image_size,
                        "normalization": "imagenet", "seed": args.seed}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_started = time.perf_counter()
    test_metrics = evaluate(loaders["test"])
    report = {
        "dataset_id": args.dataset, "task": "segmentation", "model_name": args.model,
        "model_version": "0.1.0", "num_classes": args.num_classes, "device": str(device),
        "train_rows": len(datasets["train"]), "validation_rows": len(datasets["validation"]),
        "test_rows": len(datasets["test"]), "best_validation_mean_iou": best_miou,
        "test": test_metrics, "test_evaluation_seconds": time.perf_counter() - test_started,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_bytes": checkpoint_path.stat().st_size,
        "history": history,
    }
    report_path = args.output / "smartharvest_segmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved segmentation model: {checkpoint_path.resolve()}")
    print(f"Saved evaluation: {report_path.resolve()}")


if __name__ == "__main__":
    main()
