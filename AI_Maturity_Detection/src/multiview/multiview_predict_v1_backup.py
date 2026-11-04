import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parents[2]

SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from image_quality import check_image_quality


# MODEL CONFIGURATION

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

# Preliminary review threshold.
# Must be calibrated using labelled field images.
CONFIDENCE_THRESHOLD = 0.65

# Minimum agreement between views.
AGREEMENT_THRESHOLD = 0.80


def load_mango_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return load_model(MODEL_PATH)


def find_view_image(folder, view):

    for extension in [".jpg", ".jpeg", ".png"]:

        image_path = folder / f"{view}{extension}"

        if image_path.exists():
            return image_path

    return None


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
            CLASS_NAMES[i]: round(float(scores[i]), 4)
            for i in range(len(CLASS_NAMES))
        },

        "model_score": round(
            float(scores[predicted_index]), 4
        )
    }


def predict_multiple_views(image_directory):

    image_directory = Path(image_directory)

    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    model = load_mango_model()

    view_results = []

    failed_views = []

    for view in VIEW_NAMES:

        image_path = find_view_image(
            image_directory,
            view
        )

        if image_path is None:

            raise FileNotFoundError(
                f"Missing image for view: {view}"
            )

        # IMAGE QUALITY CHECK

        quality = check_image_quality(image_path)

        if quality is None:

            failed_views.append(view)

            view_results.append({
                "view": view,
                "status": "IMAGE_UNREADABLE"
            })

            continue

        if not quality["quality_passed"]:

            failed_views.append(view)

            view_results.append({
                "view": view,
                "status": "RETAKE_REQUIRED",
                "image_quality": quality
            })

            continue

        # AI PREDICTION

        prediction = predict_single_image(
            model,
            image_path
        )

        prediction["view"] = view
        prediction["image_quality"] = quality
        prediction["status"] = "PREDICTION_AVAILABLE"

        view_results.append(prediction)

    # STOP IF ANY IMAGE FAILS QUALITY CHECK

    if failed_views:

        return {
            "fruit_type": "mango",
            "status": "RETAKE_REQUIRED",
            "failed_views": failed_views,
            "views": view_results,
            "model_version": "1.0",
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat()
        }

    # MAJORITY VOTING

    stage_counts = {}

    for result in view_results:

        stage = result["maturity_stage"]

        stage_counts[stage] = (
            stage_counts.get(stage, 0) + 1
        )

    ordered_stages = sorted(
        stage_counts,
        key=stage_counts.get,
        reverse=True
    )

    majority_stage = ordered_stages[0]

    agreement = (
        stage_counts[majority_stage] / len(view_results)
    )

    # Average model score for the selected class.
    selected_scores = [
        result["scores"][majority_stage]
        for result in view_results
    ]

    average_model_score = float(
        np.mean(selected_scores)
    )

    # CONFIDENCE / REVIEW DECISION

    if agreement < AGREEMENT_THRESHOLD:

        final_status = "MANUAL_REVIEW_REQUIRED"

    elif average_model_score < CONFIDENCE_THRESHOLD:

        final_status = "MANUAL_REVIEW_REQUIRED"

    else:

        final_status = "MULTIVIEW_RESULT_AVAILABLE"

    result = {

        "fruit_type": "mango",

        "method": "six_view_majority_vote",

        "final_maturity_stage": majority_stage,

        "average_model_score": round(
            average_model_score, 4
        ),

        "view_agreement": round(
            agreement, 4
        ),

        "status": final_status,

        "model_version": "1.0",

        "confidence_note": (
            "Model scores are not calibrated probabilities. "
            "Thresholds are preliminary and require field validation."
        ),

        "views": view_results,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python -m src.multiview.multiview_predict IMAGE_FOLDER"
        )

        sys.exit(1)

    output = predict_multiple_views(sys.argv[1])

    print(json.dumps(output, indent=4))
