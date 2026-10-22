
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==============================
# 1. KONFIGURASI
# ==============================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42

MODEL_PATH = BASE_DIR / "coffee_roast_model.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

# Periksa keberadaan folder dataset
if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
    raise FileNotFoundError(
        "Folder train dan test harus berada di folder yang sama "
        "dengan train_model.py."
    )

# ==============================
# 2. MEMUAT DATASET
# ==============================
print("Memuat dataset training...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

print("Memuat dataset validasi...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

print("Memuat dataset testing...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

class_names = train_ds.class_names
print("Urutan kelas:", class_names)

if len(class_names) != 4:
    raise ValueError(
        f"Seharusnya ada 4 kategori, tetapi ditemukan {len(class_names)}."
    )

if test_ds.class_names != class_names:
    raise ValueError(
        "Kategori train dan test tidak sama. Periksa nama folder."
    )

with open(CLASS_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# ==============================
# 3. OPTIMASI INPUT
# ==============================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==============================
# 4. MEMUAT MOBILENETV2
# ==============================
print("Memuat MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Bekukan model dasar agar pelatihan lebih ringan
base_model.trainable = False

# ==============================
# 5. MEMBANGUN KLASIFIKASI
# ==============================

# ==============================
# 5. MEMBANGUN KLASIFIKASI
# ==============================
inputs = layers.Input(shape=(224, 224, 3))

# MobileNetV2 membutuhkan nilai piksel dalam rentang -1 sampai 1.
# Normalisasi dilakukan di luar model, sehingga tidak ada Lambda.
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1,
)(inputs)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
)(x)

model = models.Model(inputs, outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==============================
# 6. MELATIH MODEL
# ==============================
print("\nMulai pelatihan model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ==============================
# 7. EVALUASI DATA TEST
# ==============================
print("\nMengevaluasi model dengan data test...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")

# ==============================
# 8. MENYIMPAN MODEL
# ==============================
model.save(MODEL_PATH)

print("\nPelatihan selesai!")
print(f"Model tersimpan: {MODEL_PATH}")
print(f"Label tersimpan: {CLASS_PATH}")
