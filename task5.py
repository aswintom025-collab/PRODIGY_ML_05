

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

DATASET_NAME = "food101"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 8
MODEL_SAVE_PATH = "food_classifier_mobilenetv2.h5"
NUM_CLASSES = 101

def preprocess(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_for_training(ds, shuffle_buffer_size=1000):
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


(ds_train, ds_val), ds_info = tfds.load(
    DATASET_NAME,
    split=["train", "validation"],
    with_info=True,
    as_supervised=True
)
print("Dataset loaded.")
label_names = ds_info.features["label"].names  # list of class names (length 101)

train_ds = prepare_for_training(ds_train)
val_ds = prepare_for_training(ds_val)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base for initial training

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)  # MobileNet expects images scaled to [-1,1]
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(label_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training (head)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("Fine-tuning last layers...")
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 4
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1] if history.epoch else 0,
    epochs=total_epochs
)

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

calorie_lookup = {
    "pizza": 285,
    "burger": 354,
    "caesar_salad": 170,
    "sushi": 200,
    "ice_cream": 207,
    "apple_pie": 237,
    "french_fries": 312,
    "steak": 271,
    "omelette": 154,
    "baklava": 430,
    "guacamole": 150,
    "pancakes": 86,
}

FALLBACK_CALORIES = 250

import PIL.Image as Image

def predict_image_from_path(image_path, top_k=1):

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)  # shape (1, num_classes)
    probs = preds[0]
    top_indices = probs.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        label = label_names[idx]
        confidence = float(probs[idx])
        key = label.lower().replace(" ", "_")
        est_cal = calorie_lookup.get(key, FALLBACK_CALORIES)
        results.append((label, confidence, est_cal))
    return results

if __name__ == "__main__":
    test_path = "baklava.jpg"
    if os.path.exists(test_path):
        predictions = predict_image_from_path(test_path, top_k=3)
        print("Predictions:")
        for label, conf, cal in predictions:
            print(f" - {label} ({conf*100:.2f}%): ~{cal} kcal (typical serving)")
    else:
        print(f"No {test_path} found")
