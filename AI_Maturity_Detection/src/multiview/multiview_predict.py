import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from image_quality import check_image_quality


# ==========================================
# FARMAURA MULTI-VIEW CONFIGURATION
# ==========================================

MODEL_PATH = BASE_DIR / "models" / "farmaura_best.keras"

CLASS_NAMES = [
    "partially_ripe",
    "ripe",
    "unripe"
]

VIEW_NAMES = [
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom"
]

IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.65
AGREEMENT_THRESHOLD = 0.80

MODEL_VERSION = "2.0"

INSPECTION_DIR = BASE_DIR / "results" / "inspections"


# ==========================================
# LOAD MODEL
# ==========================================

def load_mango_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return load_model(MODEL_PATH)


# ==========================================
# FIND IMAGE
# ==========================================

def find_view_image(folder, view):

    for extension in [
        ".jpg", ".jpeg", ".png",
        ".JPG", ".JPEG", ".PNG"
    ]:

        image_path = folder / f"{view}{extension}"

        if image_path.exists():
            return image_path

    return None


# ==========================================
# SINGLE IMAGE PREDICTION
# ==========================================

def predict_single_image(model, image_path):

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = img_to_array(image)
    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    scores = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(scores))

    return {
        "maturity_stage": CLASS_NAMES[predicted_index],

        "scores": {
            CLASS_NAMES[i]: round(float(scores[i]), 6)
            for i in range(len(CLASS_NAMES))
        },

        "model_score": round(
            float(scores[predicted_index]), 6
        )
    }


# ==========================================
# SAVE INSPECTION
# ==========================================

def save_inspection(result):

    INSPECTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    inspection_id = result["inspection_id"]

    output_path = (
        INSPECTION_DIR /
        f"{inspection_id}.json"
    )

    with open(output_path, "w") as file:

        json.dump(
            result,
            file,
            indent=4
        )

    return str(output_path.relative_to(BASE_DIR))


# ==========================================
# MULTI-VIEW PREDICTION
# ==========================================

def predict_multiple_views(image_directory):

    image_directory = Path(image_directory)

    if not image_directory.is_dir():

        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    inspection_id = str(uuid.uuid4())

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    model = load_mango_model()

    view_results = []
    failed_views = []

    # --------------------------------------
    # PROCESS ALL SIX VIEWS
    # --------------------------------------

    for view in VIEW_NAMES:

        image_path = find_view_image(
            image_directory,
            view
        )

        if image_path is None:

            failed_views.append(view)

            view_results.append({
                "view": view,
                "status": "MISSING_IMAGE"
            })

            continue

        quality = check_image_quality(
            image_path
        )

        if quality is None or not quality["quality_passed"]:

            failed_views.append(view)

            view_results.append({
                "view": view,
                "status": "RETAKE_REQUIRED",
                "image_quality": quality
            })

            continue

        prediction = predict_single_image(
            model,
            image_path
        )

        prediction["view"] = view
        prediction["image_quality"] = quality
        prediction["status"] = "PREDICTION_AVAILABLE"

        view_results.append(prediction)

    # --------------------------------------
    # STOP IF IMAGE QUALITY FAILS
    # --------------------------------------

    if failed_views:

        result = {
            "inspection_id": inspection_id,
            "fruit_type": "mango",
            "status": "RETAKE_REQUIRED",
            "failed_views": failed_views,
            "model_version": MODEL_VERSION,
            "views": view_results,
            "timestamp": timestamp
        }

        result["record_path"] = save_inspection(result)

        return result

    # --------------------------------------
    # MAJORITY VOTING
    # --------------------------------------

    stage_counts = {
        stage: 0 for stage in CLASS_NAMES
    }

    for view in view_results:

        stage = view["maturity_stage"]
        stage_counts[stage] += 1

    majority_stage = max(
        CLASS_NAMES,
        key=lambda stage: stage_counts[stage]
    )

    agreement = (
        stage_counts[majority_stage] /
        len(view_results)
    )

    # --------------------------------------
    # AVERAGE CLASS SCORES
    # --------------------------------------

    average_scores = {}

    for stage in CLASS_NAMES:

        stage_scores = [
            view["scores"][stage]
            for view in view_results
        ]

        average_scores[stage] = round(
            float(np.mean(stage_scores)),
            6
        )

    average_stage = max(
        CLASS_NAMES,
        key=lambda stage: average_scores[stage]
    )

    average_model_score = average_scores[
        average_stage
    ]

    # --------------------------------------
    # COMPARE BOTH METHODS
    # --------------------------------------

    prediction_agreement = (
        majority_stage == average_stage
    )

    # --------------------------------------
    # MANUAL REVIEW DECISION
    # --------------------------------------

    review_reasons = []

    if agreement < AGREEMENT_THRESHOLD:

        review_reasons.append(
            "LOW_VIEW_AGREEMENT"
        )

    if average_model_score < CONFIDENCE_THRESHOLD:

        review_reasons.append(
            "LOW_MODEL_SCORE"
        )

    if not prediction_agreement:

        review_reasons.append(
            "MAJORITY_AND_AVERAGE_DISAGREE"
        )

    if review_reasons:

        final_status = "MANUAL_REVIEW_REQUIRED"

    else:

        final_status = "MULTIVIEW_RESULT_AVAILABLE"

    # --------------------------------------
    # FINAL INSPECTION RESULT
    # --------------------------------------

    result = {

        "inspection_id": inspection_id,

        "fruit_type": "mango",

        "method": "majority_vote_and_average_scores",

        "majority_prediction": majority_stage,

        "average_score_prediction": average_stage,

        "final_maturity_stage": (
            average_stage
            if prediction_agreement
            else None
        ),

        "average_class_scores": average_scores,

        "average_model_score": round(
            average_model_score, 4
        ),

        "view_agreement": round(
            agreement, 4
        ),

        "prediction_methods_agree": prediction_agreement,

        "view_counts": stage_counts,

        "status": final_status,

        "review_reasons": review_reasons,

        "model_version": MODEL_VERSION,

        "confidence_note": (
            "Model scores are not calibrated probabilities. "
            "The confidence threshold is preliminary and "
            "requires validation on independent labelled field images."
        ),

        "views": view_results,

        "timestamp": timestamp
    }

    result["record_path"] = save_inspection(result)

    return result


# ==========================================
# COMMAND LINE EXECUTION
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python -m src.multiview.multiview_predict IMAGE_FOLDER"
        )

        sys.exit(1)

    try:

        output = predict_multiple_views(
            sys.argv[1]
        )

        print("\n========== FARMAURA INSPECTION ==========\n")

        print(
            json.dumps(
                output,
                indent=4
            )
        )

        print("\n=========================================\n")

    except Exception as error:

        print(f"Inspection failed: {error}")

        sys.exit(1)
