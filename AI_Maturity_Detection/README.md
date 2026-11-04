# FARMAURA AI/ML

This is the standalone AI/ML component for FARMAURA 2.0. It owns image quality checks, fruit identification, maturity classification, model evaluation and the model-to-backend response contract. Firebase/database work, inspection persistence, growth-memory history, orchard dashboards, harvest batch creation and QR lookup belong to the backend developer. Connect this module after that developer pushes the backend repository.

The member role sheets assign Member 4 differently in different versions. The current owner instruction is followed here: AI/ML work only, with later backend integration.

## Implemented AI/ML features

- Heuristic image gate for low resolution, blur, low light and overexposure. It does not claim to detect occlusion.
- Multi-view inspection that drops rejected frames and averages class probabilities over the usable views from one scan.
- Fruit identity plus a configured crop-specific maturity model, with a low-confidence `expert_check` fallback.
- Consistent 224-pixel inference preprocessing and ImageNet normalization.
- Group-safe manifest validation, train-only augmentation, test-set metrics, per-class recall, confusion matrix, calibration error, latency and model file size.
- Stable response fields in `integration/model_contract.json`, including separate identity and maturity confidences.

Multi-view consensus covers one inspection only. It does not infer growth over time. Fresh/stale, raw/ripe, maturity, identity, detection and segmentation keep separate labels and model tasks.

## Planned model pool

| Stage | Models | Dataset use |
|---|---|---|
| First classifier | MobileNetV3-Large | Fruit identity baseline, then mango maturity using the verified three Mendeley source labels |
| Classifier comparisons | EfficientNet-B0, ResNet50; optional DenseNet121, ConvNeXt-Tiny | Compare on the same reviewed split and task |
| Separate auxiliary classifiers | MobileNetV3-Large or selected benchmark model | Fresh/stale and RawRipe raw/ripe stay distinct from mango maturity |
| Detection | YOLOv8n | Only after SmartHarvest and Strawberry/Avocado annotation schemas and splits are audited |
| Segmentation | U-Net and DeepLabV3 | Optional; only if the polygon/mask annotations are usable |
| Structured data | Logistic Regression, Random Forest, optional XGBoost/LightGBM | Later, with real orchard records and reviewed outcome labels |
| Longitudinal research | GRU/LSTM | Later, after enough dated observations exist for the same tagged samples |

Classifier architectures and training/evaluation are implemented. YOLOv8n, U-Net, DeepLabV3, structured-model and GRU/LSTM factories are available as optional building blocks; task-specific training is gated until the downloaded annotation/record schemas are audited. No days-to-harvest or shelf-life predictor is enabled without verified outcome data.

The complete six-source experiment mapping is in `integration/model_experiment_matrix.json`. The five classifier architectures run against each of the four classification datasets (20 experiments). SmartHarvest feeds YOLOv8n plus optional U-Net/DeepLabV3 segmentation; Strawberry/Avocado feeds YOLOv8n. Each dataset stays on its matching task labels.

## Dataset location and audit

Default data root is the repository's `DATASETS` folder. Override it with `FARMAURA_DATA_ROOT` to point to a local dataset directory. Each source has its own folder. Preserve source cards, license notices, original labels and archive files. The catalogue is in `farmaura_ai/catalog.py`.

Run the lightweight inventory from the VS Code integrated terminal, opened at this folder:

```powershell
python -m farmaura_ai.dataset_audit --output artifacts\dataset_audit.json
```

The inventory does not certify a dataset's license, labels, splits or training readiness. Finish all six downloads, check extracted contents and sizes, verify current licenses, review labels, and prepare leakage-safe manifests before training.

## VS Code setup

Use Python 3.11 or newer. PyTorch's Windows install guide currently lists Python 3.10–3.14; select a PyTorch build matching the machine's hardware in the official installer selector. [PyTorch Windows install guide](https://pytorch.org/get-started/locally/).

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

Optional detector and structured-model packages are in `requirements-ml-optional.txt`. Install only when starting that work package:

```powershell
python -m pip install -r requirements-ml-optional.txt
```

## Train a reviewed classifier

Create a CSV manifest per task with these columns:

```text
image_path,label,group_id,split
```

`image_path` is relative to that dataset folder. `group_id` is the physical fruit/sample or capture burst. A group must appear in only one split. Include train, validation and test, and include every class in the train split. Split before augmentation; validation and test images are never augmented.

Example for the initial three-class Alphonso maturity model, after its download and review:

```powershell
python -m farmaura_ai.train_classifier `
  --dataset alphonso_mango `
  --manifest ..\DATASETS\06_Alphonso_Mango\splits.csv `
  --model mobilenet_v3_large `
  --epochs 20 `
  --output artifacts
```

Compare EfficientNet-B0 and ResNet50 on the same manifest. Choose by macro F1, per-class recall, confusion matrix, calibration, latency and model size. Do not interpret the Mendeley three-class labels as four maturity stages; do not merge RawRipe or Fresh/Stale labels into them.

After all four classification datasets have reviewed `splits.csv` manifests, plan or launch the full grid:

```powershell
python -m farmaura_ai.benchmark_classifiers --plan-only
python -m farmaura_ai.benchmark_classifiers --epochs 20 --device auto
```

The second command launches 20 dataset/model runs and writes `artifacts\\classifier_benchmark_summary.json`. It skips a dataset/model job when that dataset's reviewed `splits.csv` is missing.

For detection, first convert the audited SmartHarvest or Strawberry/Avocado annotations into YOLO format, split by physical sample/capture group, and create `data.yaml`. Then run `python -m farmaura_ai.train_detector --data path\\to\\data.yaml`. For segmentation, rasterize reviewed SmartHarvest polygons as indexed grayscale masks, then create a CSV with `image_path,mask_path,group_id,split` and run `python -m farmaura_ai.train_segmenter --manifest path\\to\\masks.csv --model unet --num-classes N`. Mask values must be class indices from `0` to `N-1`.

## Multi-view inference

Load one fruit-identity checkpoint and a dictionary of maturity checkpoints keyed by crop. `fruit_to_maturity_model` can map an identity label to a crop key when source labels differ. The pipeline accepts one or more frames from the same scan:

```python
from pathlib import Path
from farmaura_ai.inspection import FruitInspectionPipeline
from farmaura_ai.predictor import ImageClassifier

identity = ImageClassifier(Path("artifacts/fruit_identity_mobilenet_v3_large_best.pt"))
mango_maturity = ImageClassifier(Path("artifacts/alphonso_mango_mobilenet_v3_large_best.pt"))
pipeline = FruitInspectionPipeline(identity, {"mango": mango_maturity})
result = pipeline.predict_views([Path("view_1.jpg"), Path("view_2.jpg")])
```

The backend can map `result` to its response after the backend push. It should store model version and confidence separately from any verified harvest, origin or batch fields.
