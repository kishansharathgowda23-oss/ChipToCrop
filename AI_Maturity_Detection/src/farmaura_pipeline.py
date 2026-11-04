import json
import sys
from pathlib import Path

from image_quality import check_image_quality
from predict import predict_mango


def run_farmaura(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        return {
            "status": "error",
            "message": "Image not found"
        }

    print("\nFARMAURA AI PIPELINE STARTED")

    # STEP 1: Image quality check
    quality = check_image_quality(image_path)

    if quality is None:
        return {
            "status": "error",
            "message": "Image quality check failed"
        }

    if not quality["quality_passed"]:

        result = {
            "fruit_type": "mango",
            "status": "retake_required",
            "issues": quality["issues"],
            "image_quality": quality
        }

    else:

        # STEP 2: AI maturity prediction
        prediction = predict_mango(str(image_path))

        if prediction is None:
            return {
                "status": "error",
                "message": "Maturity prediction failed"
            }

        result = {
            "fruit_type": "mango",
            "maturity_stage": prediction["maturity_stage"],
            "confidence": prediction["model_score"],
            "model_version": prediction["model_version"],
            "status": prediction["status"],
            "image_quality": quality
        }

    # STEP 3: Save JSON result
    Path("results").mkdir(exist_ok=True)

    output_file = Path("results/latest_inspection.json")

    with output_file.open("w") as file:
        json.dump(result, file, indent=4)

    print("\nFARMAURA FINAL RESULT")
    print(json.dumps(result, indent=4))
    print("\nSaved to:", output_file)

    return result


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/farmaura_pipeline.py path/to/mango.jpg")

    else:
        run_farmaura(sys.argv[1])
