import os
import cv2
from PIL import Image, UnidentifiedImageError

DATA_DIR = "final_dataset"
removed = 0

def is_image_valid(path):
    """Check using PIL + OpenCV."""
    try:
        # PIL verify
        with Image.open(path) as img:
            img.verify()

        # OpenCV check
        img_cv = cv2.imread(path)
        if img_cv is None:
            return False
        
        # Try resizing
        cv2.resize(img_cv, (100, 100))
        return True

    except (UnidentifiedImageError, OSError, cv2.error):
        return False

def clean_dataset():
    global removed
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)

            # Skip non image files just in case
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                os.remove(path)
                removed += 1
                print(f"❌ Deleted non-image file: {path}")
                continue

            if not is_image_valid(path):
                os.remove(path)
                removed += 1
                print(f"🗑 Deleted corrupted: {path}")

    print("\n--------------------------------------")
    print(f"✔ Cleanup complete! Total removed: {removed}")
    print("--------------------------------------")

if __name__ == "__main__":
    clean_dataset()
