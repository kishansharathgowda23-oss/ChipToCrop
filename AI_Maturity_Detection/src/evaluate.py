import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Configuration
TEST_DIR = Path("dataset/processed/mango/test")
MODEL_PATH = Path("models/farmaura_best.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("\nFARMAURA MODEL EVALUATION")
print("=" * 45)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate test data
loss, accuracy = model.evaluate(test_ds)

print("\nTEST RESULTS")
print("=" * 45)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# Generate predictions
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Save report
Path("results").mkdir(exist_ok=True)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("FARMAURA Mango Classification")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nEvaluation completed!")
print("Results saved in results/")
