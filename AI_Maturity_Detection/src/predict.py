import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path("models/farmaura_best.keras")
IMAGE_SIZE = (224, 224)

# Alphabetical class order from the training output
CLASS_NAMES = [
    "partially_ripe",
    "ripe",
    "unripe"
]

def predict_mango(image_path):

    image_path = Path(image_path)

    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found:", MODEL_PATH)
        return

    if not image_path.exists():
        print("ERROR: Image not found:", image_path)
        return

    print("Loading FARMAURA model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Keep pixel scaling consistent with the trained model.
    predictions = model.predict(image_array, verbose=0)[0]

    index = int(np.argmax(predictions))
    stage = CLASS_NAMES[index]
    score = float(predictions[index])

    print("\n" + "=" * 45)
    print("       FARMAURA MANGO AI RESULT")
    print("=" * 45)
    print("Image:", image_path.name)
    print("Fruit: Mango")
    print("Maturity:", stage.replace("_", " ").title())
    print(f"Model score: {score * 100:.2f}%")

    print("\nClass scores:")
    for name, value in zip(CLASS_NAMES, predictions):
        print(f"{name}: {value * 100:.2f}%")

    if score < 0.65:
        status = "LOW CONFIDENCE - RETAKE IMAGE"
    else:
        status = "PREDICTION AVAILABLE"

    print("\nStatus:", status)
    print("=" * 45)

    return {
        "fruit_type": "mango",
        "maturity_stage": stage,
        "model_score": round(score, 4),
        "model_version": "1.0",
        "status": status
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/mango.jpg")
    else:
        predict_mango(sys.argv[1])
