from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.getenv("FARMAURA_DATA_ROOT", str(PROJECT_ROOT / "DATASETS"))).expanduser().resolve()


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    name: str
    folder: str
    source: str
    task: str
    license: str
    license_note: str
    source_url: str
    classifier_ready: bool
    label_note: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "fruits360", "Fruits-360", "01_Fruits360", "Kaggle: moltean/fruits",
        "fruit_identity", "CC BY-SA 4.0", "Kaggle CLI reported CC-BY-SA-4.0; preserve the included license.",
        "https://www.kaggle.com/datasets/moltean/fruits", True,
        "Identity classification only. Do not treat these classes as ripeness labels.",
    ),
    DatasetSpec(
        "smartharvest", "SmartHarvest", "02_SmartHarvest", "Hugging Face: TheCoffeeAddict/SmartHarvest",
        "detection_segmentation_ripeness", "VERIFY", "The planning card cited MIT; verify the current card and repository files before use.",
        "https://huggingface.co/datasets/TheCoffeeAddict/SmartHarvest", False,
        "Natural garden imagery with detection/segmentation annotations; inspect the downloaded schema and splits.",
    ),
    DatasetSpec(
        "fresh_stale", "Fresh and Stale Images of Fruits and Vegetables", "03_Fresh_Stale",
        "Kaggle: raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables",
        "freshness", "CC0-1.0", "Kaggle CLI reported CC0-1.0; freshness remains separate from ripeness and maturity.",
        "https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables", True,
        "Keep the original fresh/stale labels; never map them to unripe/ripe.",
    ),
    DatasetSpec(
        "rawripe", "RawRipe", "04_RawRipe", "Official project page Google Drive archive",
        "raw_ripe", "CC BY 4.0 (source claim; verify archive terms)", "The archive contains no license/readme file; the project repository's license statement must be checked against the original image archive.",
        "https://sites.google.com/site/koteswarraojerripothula/research-group/fruit-maturity-recognition", True,
        "Two source labels, raw and ripe. Do not infer a four-stage maturity scale.",
    ),
    DatasetSpec(
        "strawberry_avocado", "Strawberry & Avocado Ripeness Detection", "05_Strawberry_Avocado",
        "Hugging Face: Project-AgML/strawberry_avocado_ripeness_detection",
        "object_detection", "CC BY 4.0", "Downloaded Hugging Face README declares CC BY 4.0 and lists 758 images, 924 boxes and 8 categories; preserve the card and annotations.",
        "https://huggingface.co/datasets/Project-AgML/strawberry_avocado_ripeness_detection", False,
        "Detection data for strawberry and avocado; not an ImageFolder classification source.",
    ),
    DatasetSpec(
        "alphonso_mango", "Alphonso Mango Ripening Stage Classification", "06_Alphonso_Mango",
        "Mendeley Data: tyghd6gxw2, version 1",
        "maturity", "CC BY 4.0", "Mendeley lists 750 images, 250 per stage, in a 5.93 GiB Data.rar archive.",
        "https://data.mendeley.com/datasets/tyghd6gxw2/1", True,
        "Keep unripe, partially ripe, and ripe as the three source classes.",
    ),
)

DATASET_BY_ID = {item.dataset_id: item for item in DATASETS}


def dataset_path(dataset_id: str) -> Path:
    spec = DATASET_BY_ID[dataset_id]
    return DATASET_ROOT / spec.folder


def dataset_summary() -> list[dict[str, str | bool]]:
    return [
        {
            "dataset_id": item.dataset_id,
            "name": item.name,
            "folder": str(dataset_path(item.dataset_id)),
            "task": item.task,
            "license": item.license,
            "classifier_ready": item.classifier_ready,
            "label_note": item.label_note,
            "source_url": item.source_url,
        }
        for item in DATASETS
    ]
