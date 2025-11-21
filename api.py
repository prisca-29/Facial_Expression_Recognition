from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
CORS(app)

MODEL_PATH = "emotion_resnet50_finetuned.h5"
CLASS_NAMES_PATH = "class_names.npy"
IMG_SIZE = 224

print("🔁 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
print("✅ Model loaded:", class_names)


def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


# ------------------ PAGES ------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# ------------------ UPLOAD IMAGE PREDICTION ------------------

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    processed = preprocess_image(img)
    pred = model.predict(processed)[0]
    idx = np.argmax(pred)

    return jsonify({
        "label": class_names[idx],
        "confidence": float(pred[idx] * 100)
    })


if __name__ == "__main__":
    app.run(debug=True)
