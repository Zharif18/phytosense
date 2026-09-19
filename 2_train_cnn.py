"""
PhytoSense — 1D CNN Stress Classifier
=======================================
Trains a 1D CNN on the bioelectric signal dataset to classify plant state.
Architecture is deliberately small/fast so it can also run efficiently on
constrained inference targets (and converts cleanly to TF.js for the browser
dashboard demo).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json

SEQ_LEN = 500
CLASSES = ["Healthy", "Drought Stress", "Salt Stress", "Mechanical/Wound", "Heat Stress"]

df = pd.read_csv("D:\COLLEGE\project\Phytosense\signals\phytosense_dataset.csv")
t_cols = [f"t{i}" for i in range(SEQ_LEN)]
X = df[t_cols].values.astype("float32")
y = df["label"].values.astype("int32")

# --- Normalization ---
# Per-sample z-score normalization (removes baseline offset differences between
# electrode placements/plants — important since absolute mV baseline drifts
# with electrode contact quality in real hardware).
mu = X.mean(axis=1, keepdims=True)
sigma = X.std(axis=1, keepdims=True) + 1e-8
X_norm = (X - mu) / sigma

# Save global normalization stats (used for consistent unseen-signal scaling)
norm_stats = {"mean": float(X.mean()), "std": float(X.std())}
with open("D:\COLLEGE\project\Phytosense\signals\demo_samples.json", "w") as f:
    json.dump(norm_stats, f)

X_norm = X_norm[..., np.newaxis]  # (N, 500, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# --- Model ---
model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, 1)),
    keras.layers.Conv1D(16, kernel_size=9, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),

    keras.layers.Conv1D(32, kernel_size=7, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),

    keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),

    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(CLASSES), activation="softmax"),
], name="phytosense_1dcnn")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_test, y_pred, target_names=CLASSES, digits=3)
print(report)
with open("D:/COLLEGE/project/Phytosense/classification_report.txt", "w") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred)

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("epoch")
axes[0].legend()

im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks(range(len(CLASSES)))
axes[1].set_yticks(range(len(CLASSES)))
axes[1].set_xticklabels(CLASSES, rotation=45, ha="right")
axes[1].set_yticklabels(CLASSES)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title(f"Confusion Matrix (acc={test_acc:.3f})")
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        axes[1].text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("D:/COLLEGE/project/phytosense/training_results.png", dpi=110)

# --- Save model ---
model.save("D:/COLLEGE/project/phytosense/phytosense_cnn.keras")
print("\nSaved model to phytosense_cnn.keras")
