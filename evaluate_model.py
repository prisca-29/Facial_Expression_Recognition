import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "emotion_resnet50_finetuned.h5"
CLASS_NAMES_PATH = "class_names.npy"
TEST_DIR = "final_dataset"   # your dataset

IMG_SIZE = 224

print("🔁 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
print("✅ Model loaded:", class_names)


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


y_true = []
y_pred = []

print("\n🔍 Evaluating on dataset...\n")

for class_name in class_names:
    folder = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(folder):
        print(f"⚠ Skipping missing class: {folder}")
        continue

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # load and preprocess
        try:
            img = load_image(path)
        except:
            continue

        pred = model.predict(img, verbose=0)[0]
        pred_idx = np.argmax(pred)

        y_pred.append(pred_idx)
        y_true.append(class_names.index(class_name))

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("\n📊 Overall Accuracy:", round(acc * 100, 2), "%")

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
