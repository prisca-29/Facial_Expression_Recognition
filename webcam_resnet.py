import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "emotion_resnet50_finetuned.h5"
CLASS_NAMES_PATH = "class_names.npy"
IMG_SIZE = 224

# Global flag to exit from mouse click
exit_flag = False

def click_event(event, x, y, flags, param):
    global exit_flag
    if event == cv2.EVENT_LBUTTONDOWN and 10 < x < 120 and 10 < y < 50:
        print("🛑 Exit button clicked.")
        exit_flag = True

def main():
    global exit_flag
    exit_flag = False

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()
    print("Classes:", class_names)

    # Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # safer for Windows webcams

    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    print("\nWebcam started — press 'q' or click EXIT.\n")

    cv2.namedWindow("Emotion Recognition")
    cv2.setMouseCallback("Emotion Recognition", click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            # Preprocess
            try:
                face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            except:
                continue

            face_array = np.expand_dims(face_resized, 0)
            face_array = preprocess_input(face_array)

            pred = model.predict(face_array, verbose=0)[0]
            idx = np.argmax(pred)
            label = class_names[idx]
            conf = pred[idx] * 100

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Draw EXIT button
        cv2.rectangle(frame, (10, 10), (120, 50), (0, 0, 255), -1)
        cv2.putText(frame, "EXIT", (35, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Emotion Recognition", frame)

        # Check for EXIT flag
        if exit_flag:
            print("🛑 Exit via button")
            break

        # Keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exit via Q key")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Webcam closed safely")


if __name__ == "__main__":
    main()
