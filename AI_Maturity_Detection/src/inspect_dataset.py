from pathlib import Path
from PIL import Image
from collections import Counter
import hashlib

DATASET = Path("dataset/raw/mango")

CLASSES = [
    "unripe",
    "partially_ripe",
    "ripe",
    "overripe"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

total = 0
invalid = 0
duplicate_count = 0
seen_hashes = set()

all_dimensions = Counter()

print("\nFARMAURA - MANGO DATASET INSPECTION")
print("=" * 45)

for class_name in CLASSES:
    folder = DATASET / class_name

    if not folder.exists():
        print(f"\n{class_name}: Folder missing")
        continue

    valid = 0
    bad = 0
    dimensions = Counter()

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(file) as img:
                img.verify()

            with Image.open(file) as img:
                dimensions[img.size] += 1
                all_dimensions[img.size] += 1

            file_hash = hashlib.sha256(file.read_bytes()).hexdigest()

            if file_hash in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(file_hash)

            valid += 1

        except Exception:
            bad += 1
            invalid += 1

    total += valid

    print(f"\nClass: {class_name}")
    print(f"Valid images: {valid}")
    print(f"Invalid images: {bad}")
    print(f"Image dimensions: {dimensions.most_common(5)}")

print("\n" + "=" * 45)
print(f"TOTAL VALID IMAGES: {total}")
print(f"INVALID IMAGES: {invalid}")
print(f"EXACT DUPLICATE FILES: {duplicate_count}")
print(f"UNIQUE IMAGE DIMENSIONS: {len(all_dimensions)}")

if total == 0:
    print("\nDataset is empty. Add images before continuing.")
else:
    print("\nDataset inspection completed.")
