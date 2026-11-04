from __future__ import annotations

import argparse
from pathlib import Path

from farmaura_ai.vision_models import build_yolov8n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n after converting reviewed annotations to group-safe YOLO format."
    )
    parser.add_argument("--data", required=True, type=Path, help="Reviewed YOLO data.yaml.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("artifacts/detection"))
    parser.add_argument("--name", default="yolov8n_farmaura")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"YOLO data.yaml not found: {args.data}")
    if min(args.epochs, args.imgsz, args.batch_size) < 1:
        raise SystemExit("epochs, imgsz and batch-size must be positive.")

    model = build_yolov8n()
    device = None if args.device == "auto" else args.device
    model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch_size, device=device, project=str(args.project.resolve()),
        name=args.name, patience=10, save=True, plots=True,
    )


if __name__ == "__main__":
    main()
