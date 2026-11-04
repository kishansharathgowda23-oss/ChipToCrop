import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parents[2]

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


def load_mango_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return load_model(MODEL_PATH)


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

    model = load_mango_model()

    view_results = []

    for view in VIEW_NAMES:

        image_path = image_directory / f"{view}.jpg"

        if not image_path.exists():
            image_path = image_directory / f"{view}.jpeg"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for view: {view}"
            )

        prediction = predict_single_image(
            model,
            image_path
        )

        prediction["view"] = view

        view_results.append(prediction)

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

    if agreement < 0.5:

        final_status = "MANUAL_REVIEW_REQUIRED"

    else:

        final_status = "MULTIVIEW_RESULT_AVAILABLE"

    result = {
        "fruit_type": "mango",
        "method": "six_view_majority_vote",
        "final_maturity_stage": majority_stage,
        "view_agreement": round(agreement, 4),
        "status": final_status,
        "model_version": "1.0",
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
