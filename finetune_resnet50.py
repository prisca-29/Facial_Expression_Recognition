import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam

DATA_DIR = "final_dataset"
MODEL_PATH = "emotion_resnet50.h5"
FINETUNE_MODEL = "emotion_resnet50_finetuned.h5"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

def main():
    print("\n📌 Loading existing trained model...")
    model = load_model(MODEL_PATH)

    print("📌 Unfreezing last 50 layers for fine-tuning...")
    for layer in model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    print("\n🚀 Fine-Tuning Started...\n")
    history = model.fit(train_gen, epochs=EPOCHS)

    print("\n💾 Saving fine-tuned model...")
    model.save(FINETUNE_MODEL)
    print(f"✔ Model saved as → {FINETUNE_MODEL}")

    print("\n🎉 Fine-Tuning Completed Successfully!")

if __name__ == "__main__":
    main()
