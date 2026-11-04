import cv2
import sys
from pathlib import Path


def check_image_quality(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print("ERROR: Image not found.")
        return None

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Cannot read image.")
        return None

    height, width = image.shape[:2]

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate sharpness
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Calculate average brightness
    brightness = float(gray.mean())

    # Calculate dark and overexposed pixel percentages
    dark_pixels = (gray <= 5).mean() * 100
    bright_pixels = (gray >= 250).mean() * 100

    problems = []

    # Resolution check
    if width < 224 or height < 224:
        problems.append("Image resolution is too low.")

    # Initial sharpness threshold
    # This is a starting value and must be calibrated.
    if sharpness < 20:
        problems.append("Image may be blurry.")

    # Average brightness check
    if brightness < 40:
        problems.append("Image may be too dark.")

    # Check actual extreme bright pixels
    if bright_pixels > 20:
        problems.append("Image may be overexposed.")

    # Check extreme dark pixels
    if dark_pixels > 50:
        problems.append("Image contains too many very dark pixels.")

    print("\n" + "=" * 45)
    print("       FARMAURA IMAGE QUALITY CHECK")
    print("=" * 45)

    print(f"Image: {image_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"Sharpness score: {sharpness:.2f}")
    print(f"Brightness: {brightness:.2f}")
    print(f"Dark pixels: {dark_pixels:.2f}%")
    print(f"Overexposed pixels: {bright_pixels:.2f}%")

    print("\nQUALITY RESULT")

    if problems:
        print("IMAGE NEEDS RETAKING")

        for problem in problems:
            print("-", problem)

    else:
        print("IMAGE PASSED INITIAL QUALITY CHECK")

    print("=" * 45)

    return {
        "width": width,
        "height": height,
        "sharpness": round(float(sharpness), 2),
        "brightness": round(brightness, 2),
        "quality_passed": len(problems) == 0,
        "issues": problems
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/image_quality.py image.jpg")

    else:
        check_image_quality(sys.argv[1])
