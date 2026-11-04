from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import hashlib

SOURCE = Path("dataset/raw/mango")
OUTPUT = Path("dataset/processed/mango")

CLASSES = ["unripe", "partially_ripe", "ripe"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SEED = 42

images = []
labels = []
hashes = set()

print("\nFARMAURA DATASET SPLITTING")
print("=" * 40)

for class_name in CLASSES:
    folder = SOURCE / class_name

    for file in sorted(folder.iterdir()):

        if not file.is_file():
            continue

        if file.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(file) as img:
                img.verify()

            digest = hashlib.sha256(file.read_bytes()).hexdigest()

            # Skip exact duplicates
            if digest in hashes:
                print("Duplicate skipped:", file.name)
                continue

            hashes.add(digest)

            images.append(file)
            labels.append(class_name)

        except Exception:
            print("Invalid image skipped:", file)

print("Total unique valid images:", len(images))

# Split into training (70%) and temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

# Split temporary into validation (15%) and testing (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

splits = [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test)
]

for split_name, split_images, split_labels in splits:

    for image_path, label in zip(split_images, split_labels):

        destination = OUTPUT / split_name / label
        destination.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            image_path,
            destination / image_path.name
        )

print("\nDATASET SPLIT SUMMARY")
print("=" * 40)

for split_name, split_images, split_labels in splits:
    print(f"\n{split_name.upper()}: {len(split_images)}")

    for class_name in CLASSES:
        count = split_labels.count(class_name)
        print(f"  {class_name}: {count}")

print("\nDataset splitting completed successfully.")
