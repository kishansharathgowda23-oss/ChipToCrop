from __future__ import annotations

from typing import Any, Literal, TypedDict


TaskName = Literal[
    "fruit_maturity_inspection",
    "fruit_identity",
    "maturity",
    "raw_ripe",
    "freshness",
    "object_detection",
    "segmentation",
]


class ModelResponse(TypedDict):
    fruit_type: str | None
    variety: str | None
    fruit_confidence: float | None
    maturity_label: str | None
    freshness_label: str | None
    raw_ripe_label: str | None
    confidence: float
    task: str
    model_name: str
    model_version: str
    image_quality: str
    advisory: str
    limitations: list[str]
    views_analyzed: int
    views_rejected: int
    quality_details: list[dict[str, Any]]


def build_response(
    *,
    task: TaskName,
    predicted_label: str,
    confidence: float,
    model_name: str,
    model_version: str,
    fruit_type: str | None = None,
    variety: str | None = None,
    fruit_confidence: float | None = None,
    image_quality: str = "not_evaluated",
) -> ModelResponse:
    """Build the stable AI-to-backend contract without merging label meanings."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    response: ModelResponse = {
        "fruit_type": fruit_type,
        "variety": variety,
        "fruit_confidence": round(float(fruit_confidence), 6) if fruit_confidence is not None else None,
        "maturity_label": None,
        "freshness_label": None,
        "raw_ripe_label": None,
        "confidence": round(float(confidence), 6),
        "task": task,
        "model_name": model_name,
        "model_version": model_version,
        "image_quality": image_quality,
        "advisory": "expert_check" if confidence < 0.60 else "monitor_or_harvest_review",
        "limitations": ["visual estimate; verify in field"],
        "views_analyzed": 1,
        "views_rejected": 0,
        "quality_details": [],
    }

    if task == "freshness":
        response["freshness_label"] = predicted_label
    elif task == "raw_ripe":
        response["raw_ripe_label"] = predicted_label
    elif task in ("maturity", "fruit_maturity_inspection"):
        response["maturity_label"] = predicted_label
    elif task == "fruit_identity":
        response["fruit_type"] = predicted_label
    elif task in ("object_detection", "segmentation"):
        response["limitations"].append("detection/segmentation output requires its task-specific annotation contract")
    return response
