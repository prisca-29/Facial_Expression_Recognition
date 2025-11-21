import os
from PIL import Image

DATA_DIR = "final_dataset"
fixed = 0
deleted = 0

def fix_or_remove(path):
    global fixed, deleted
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")  # ensures valid format
            rgb.save(path, "JPEG")    # re-save & fix file
            fixed += 1
    except:
        os.remove(path)
        deleted += 1
        print(f"🗑 Deleted corrupt image: {path}")

def process():
    global fixed, deleted
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif",".webp")):
                fix_or_remove(path)

    print("\n--------------------------------------")
    print(f"✔ Images fixed: {fixed}")
    print(f"🗑 Images deleted: {deleted}")
    print("--------------------------------------")

if __name__ == "__main__":
    process()
