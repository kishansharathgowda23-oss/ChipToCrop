from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from farmaura_ai.catalog import DATASET_BY_ID, dataset_path
from farmaura_ai.models import CLASSIFIER_MODELS


DEFAULT_DATASETS = [key for key, spec in DATASET_BY_ID.items() if spec.classifier_ready]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five-model classifier benchmark on every classification dataset."
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", choices=CLASSIFIER_MODELS, default=CLASSIFIER_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--plan-only", action="store_true", help="List jobs without training.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")

    jobs = []
    for dataset_id in args.datasets:
        spec = DATASET_BY_ID[dataset_id]
        jobs.extend({
            "dataset_id": dataset_id,
            "task": spec.task,
            "model": model_name,
            "manifest": str((dataset_path(dataset_id) / "splits.csv").resolve()),
        } for model_name in args.models)

    if args.plan_only:
        print(json.dumps({
            "job_count": len(jobs),
            "datasets": args.datasets,
            "models": args.models,
            "jobs": jobs,
            "training_started": False,
        }, indent=2))
        return

    completed = []
    for job in jobs:
        manifest = Path(job["manifest"])
        if not manifest.is_file():
            completed.append({**job, "status": "skipped", "reason": "reviewed splits.csv is missing"})
            print(f"SKIP {job['dataset_id']} / {job['model']}: {manifest} is missing.")
            continue
        command = [
            sys.executable, "-m", "farmaura_ai.train_classifier",
            "--dataset", job["dataset_id"], "--manifest", str(manifest),
            "--model", job["model"], "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--workers", str(args.workers),
            "--device", args.device, "--output", str(args.output),
        ]
        if args.no_pretrained:
            command.append("--no-pretrained")
        print(f"RUN {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        completed.append({**job, "status": "completed" if result.returncode == 0 else "failed",
                          "return_code": result.returncode})

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_count": len(jobs),
        "completed": completed,
        "training_started": True,
    }
    report_path = args.output / "classifier_benchmark_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved benchmark summary: {report_path.resolve()}")
    if any(item["status"] == "failed" for item in completed):
        raise SystemExit(1)
    if any(item["status"] == "skipped" for item in completed):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
