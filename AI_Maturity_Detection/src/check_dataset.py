from pathlib import Path
from PIL import Image

DATASET = Path("dataset/raw/mango")

classes = [
    "unripe",
    "partially_ripe",
    "ripe",
    "overripe"
]

valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

total = 0

for class_name in classes:
    folder = DATASET / class_name
    count = 0
    invalid = 0

    for file in folder.iterdir():
        if file.suffix.lower() not in valid_extensions:
            continue

        try:
            with Image.open(file) as img:
                img.verify()
            count += 1
        except Exception:
            invalid += 1

    total += count

    print(f"{class_name}: {count} valid images")
    print(f"Invalid images: {invalid}")
    print("-" * 30)

print(f"TOTAL VALID IMAGES: {total}")

