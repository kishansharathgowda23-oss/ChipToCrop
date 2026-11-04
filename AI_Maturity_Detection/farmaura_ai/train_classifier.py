from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from farmaura_ai.catalog import DATASET_BY_ID, dataset_path
from farmaura_ai.manifest import read_and_validate_manifest
from farmaura_ai.models import CLASSIFIER_MODELS, PRIMARY_MODEL, build_classifier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: list[int], y_pred: list[int], class_names: list[str], confidences: list[float]) -> dict:
    n = len(y_true)
    k = len(class_names)
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    per_class = {}
    f1s = []
    recalls = []
    precisions = []
    for i, name in enumerate(class_names):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    ece = 0.0
    for bin_id in range(10):
        lower, upper = bin_id / 10, (bin_id + 1) / 10
        selected = [idx for idx, confidence in enumerate(confidences)
                    if lower <= confidence < upper or (bin_id == 9 and confidence == 1.0)]
        if selected:
            accuracy = sum(y_true[idx] == y_pred[idx] for idx in selected) / len(selected)
            mean_conf = sum(confidences[idx] for idx in selected) / len(selected)
            ece += len(selected) / n * abs(accuracy - mean_conf)
    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / n,
        "macro_precision": sum(precisions) / k,
        "macro_recall": sum(recalls) / k,
        "macro_f1": sum(f1s) / k,
        "expected_calibration_error_10_bins": ece,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": matrix,
    }


def _evaluate(model, loader, device, class_names):
    import torch
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    latencies = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / images.shape[0]
            probabilities = logits.softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            truths.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            latencies.extend([elapsed] * images.shape[0])
    result = _metrics(truths, predictions, class_names, confidences)
    result["mean_inference_latency_ms_per_image"] = 1000 * (sum(latencies) / len(latencies))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one FARMAURA image classifier from a reviewed split manifest.")
    parser.add_argument("--dataset", required=True, choices=[key for key, spec in DATASET_BY_ID.items() if spec.classifier_ready])
    parser.add_argument("--manifest", required=True, type=Path, help="CSV columns: image_path,label,group_id,split")
    parser.add_argument("--model", choices=CLASSIFIER_MODELS, default=PRIMARY_MODEL)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=0, help="0 is the safest default on Windows.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-pretrained", action="store_true", help="Do not fetch ImageNet weights.")
    args = parser.parse_args()

    spec = DATASET_BY_ID[args.dataset]
    if not spec.classifier_ready:
        raise SystemExit(f"{args.dataset} is not a classification dataset; use a task-specific detector/segmenter.")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    _seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    root = dataset_path(args.dataset)
    rows = read_and_validate_manifest(args.manifest, root)
    class_names = sorted({row["label"] for row in rows})
    if len(class_names) < 2:
        raise SystemExit("At least two label classes are required.")
    train_classes = {row["label"] for row in rows if row["split"] == "train"}
    missing_train_classes = set(class_names) - train_classes
    if missing_train_classes:
        raise SystemExit(f"Every class needs training examples; missing from train: {sorted(missing_train_classes)}")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class ManifestDataset(Dataset):
        def __init__(self, split: str, transform):
            self.rows = [row for row in rows if row["split"] == split]
            if not self.rows:
                raise SystemExit(f"Manifest has no {split} rows.")
            self.transform = transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(root / row["image_path"]).convert("RGB")
            return self.transform(image), class_to_id[row["label"]]

    train_data = ManifestDataset("train", train_transform)
    val_data = ManifestDataset("validation", eval_transform)
    test_data = ManifestDataset("test", eval_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == "cuda")

    model = build_classifier(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / f"{args.dataset}_{args.model}_best.pt"
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.size(0)
            seen += labels.size(0)
        validation = _evaluate(model, val_loader, device, class_names)
        row = {"epoch": epoch, "train_loss": loss_total / seen, "validation": validation}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                        "dataset_id": args.dataset, "task": spec.task, "model_name": args.model,
                        "model_version": "0.1.0", "input_size": 224,
                        "normalization": "imagenet", "seed": args.seed}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = _evaluate(model, test_loader, device, class_names)
    model_bytes = checkpoint_path.stat().st_size
    report = {
        "dataset_id": args.dataset, "task": spec.task, "model_name": args.model,
        "model_version": "0.1.0", "class_names": class_names, "device": str(device),
        "train_rows": len(train_data), "validation_rows": len(val_data), "test_rows": len(test_data),
        "best_validation_macro_f1": best_f1, "test": test_metrics,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_bytes": model_bytes,
        "history": history,
    }
    report_path = args.output / f"{args.dataset}_{args.model}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved model: {checkpoint_path.resolve()}")
    print(f"Saved evaluation: {report_path.resolve()}")


if __name__ == "__main__":
    main()
