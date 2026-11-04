from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Sequence

from farmaura_ai.contracts import ModelResponse
from farmaura_ai.image_quality import assess_image
from farmaura_ai.predictor import ImageClassifier


CONFIDENCE_THRESHOLD = 0.60


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


class FruitInspectionPipeline:
    """AI-only multi-view inspection. It does not persist records or infer history."""

    def __init__(self, fruit_identifier: ImageClassifier,
                 maturity_models: dict[str, ImageClassifier],
                 fruit_to_maturity_model: dict[str, str] | None = None) -> None:
        self.fruit_identifier = fruit_identifier
        self.maturity_models = {_normalise(key): model for key, model in maturity_models.items()}
        self.fruit_to_maturity_model = {
            _normalise(fruit): _normalise(model)
            for fruit, model in (fruit_to_maturity_model or {}).items()
        }

    def _empty_result(self, rejected: int, details: list[dict]) -> ModelResponse:
        return {
            "fruit_type": None, "variety": None, "fruit_confidence": None,
            "maturity_label": None, "freshness_label": None, "raw_ripe_label": None,
            "confidence": 0.0, "task": "fruit_maturity_inspection",
            "model_name": "FARMAURA multi-view inspection",
            "model_version": "0.1.0", "image_quality": "retake",
            "advisory": "retake_image",
            "limitations": ["No frame passed the image-quality heuristics.",
                            "Quality checks do not detect occlusion or certify field suitability."],
            "views_analyzed": 0, "views_rejected": rejected, "quality_details": details,
        }

    def predict_views(self, image_paths: Sequence[Path]) -> ModelResponse:
        if not image_paths:
            raise ValueError("Provide at least one captured frame.")
        details = [{"image": path.name, **assess_image(path)} for path in image_paths]
        usable = [path for path, quality in zip(image_paths, details) if quality["usable"]]
        rejected = len(image_paths) - len(usable)
        if not usable:
            return self._empty_result(rejected, details)

        fruit_votes = [self.fruit_identifier.predict_probabilities(path) for path in usable]
        fruit_classes = fruit_votes[0].keys()
        mean_fruit_scores = {label: mean(vote[label] for vote in fruit_votes) for label in fruit_classes}
        fruit_type, fruit_confidence = max(mean_fruit_scores.items(), key=lambda item: item[1])

        selected_key = self.fruit_to_maturity_model.get(_normalise(fruit_type), _normalise(fruit_type))
        maturity_model = self.maturity_models.get(selected_key)
        if maturity_model is None:
            return {
                "fruit_type": fruit_type, "variety": None,
                "fruit_confidence": round(fruit_confidence, 6), "maturity_label": None,
                "freshness_label": None, "raw_ripe_label": None, "confidence": 0.0,
                "task": "fruit_maturity_inspection", "model_name": self.fruit_identifier.model_name,
                "model_version": self.fruit_identifier.model_version,
                "image_quality": "mixed" if rejected else "acceptable",
                "advisory": "expert_check",
                "limitations": ["No maturity model is configured for this predicted fruit.",
                                "Fruit identity and maturity are separate model outputs."],
                "views_analyzed": len(usable), "views_rejected": rejected,
                "quality_details": details,
            }

        stage_votes = [maturity_model.predict_probabilities(path) for path in usable]
        stage_classes = stage_votes[0].keys()
        mean_stage_scores = {label: mean(vote[label] for vote in stage_votes) for label in stage_classes}
        maturity_label, confidence = max(mean_stage_scores.items(), key=lambda item: item[1])
        advisory = "expert_check" if min(confidence, fruit_confidence) < CONFIDENCE_THRESHOLD \
            else "monitor_or_harvest_review"
        model_name = f"{self.fruit_identifier.model_name} + {maturity_model.model_name}"
        return {
            "fruit_type": fruit_type, "variety": None,
            "fruit_confidence": round(fruit_confidence, 6),
            "maturity_label": maturity_label, "freshness_label": None, "raw_ripe_label": None,
            "confidence": round(confidence, 6), "task": "fruit_maturity_inspection",
            "model_name": model_name,
            "model_version": f"{self.fruit_identifier.model_version}+{maturity_model.model_version}",
            "image_quality": "mixed" if rejected else "acceptable",
            "advisory": advisory,
            "limitations": ["visual estimate; verify in field",
                            "image quality checks are heuristic and do not assess occlusion",
                            "multi-view consensus covers this scan only; it does not infer growth over time"],
            "views_analyzed": len(usable), "views_rejected": rejected,
            "quality_details": details,
        }
