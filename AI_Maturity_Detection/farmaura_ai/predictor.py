from __future__ import annotations

from pathlib import Path

from farmaura_ai.catalog import DATASET_BY_ID
from farmaura_ai.contracts import ModelResponse, build_response
from farmaura_ai.models import build_classifier


class ImageClassifier:
    """Load a trained classifier and return a backend-neutral response object."""

    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.dataset_id = checkpoint["dataset_id"]
        self.spec = DATASET_BY_ID[self.dataset_id]
        self.class_names = checkpoint["class_names"]
        self.model_name = checkpoint["model_name"]
        self.model_version = checkpoint["model_version"]
        self.model = build_classifier(self.model_name, len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def predict_probabilities(self, image_path: Path) -> dict[str, float]:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        return {name: float(probabilities[index].item())
                for index, name in enumerate(self.class_names)}

    def predict(self, image_path: Path) -> ModelResponse:
        probabilities = self.predict_probabilities(image_path)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        task = self.spec.task
        fruit_type = label if task == "fruit_identity" else None
        return build_response(
            task=task,
            predicted_label=label,
            confidence=confidence,
            model_name=self.model_name,
            model_version=self.model_version,
            fruit_type=fruit_type,
            fruit_confidence=confidence if task == "fruit_identity" else None,
            image_quality="not_evaluated",
        )
