# comparison_eval.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# ======== USER CONFIG ==========
MODEL_A_PATH = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/skin_cancer_finetuned.keras"
MODEL_B_PATH = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/cnn_from_scratch_isic.keras"
DATA_DIR = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/Dataset-split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # use 16–32 for EfficientNetB2 if VRAM limited
# ===============================

# ----- Data Generators -----
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ----- Load Models -----
model_a = load_model(MODEL_A_PATH, compile=False)
model_b = load_model(MODEL_B_PATH, compile=False)

# 🔧 Fix for Sequential models that have no defined input
def ensure_model_built(model, img_shape=(224, 224, 3)):
    """Ensure the model has input/output tensors defined."""
    if not hasattr(model, 'inputs'):
        try:
            model.build((None, *img_shape))
        except:
            _ = model(tf.zeros((1, *img_shape)))

ensure_model_built(model_a, (224, 224, 3))
ensure_model_built(model_b, (224, 224, 3))

# ----- Helper to get probs & logits -----
def get_probs_and_logits(model, generator):
    generator.reset()
    probs = model.predict(generator, verbose=1)

    last_layer_input = model.layers[-1].input
    logits_model = tf.keras.Model(inputs=model.inputs, outputs=last_layer_input)

    generator.reset()
    logits = logits_model.predict(generator, verbose=1)
    return probs, logits

# Get outputs
probs_a, logits_a = get_probs_and_logits(model_a, val_generator)
probs_b, logits_b = get_probs_and_logits(model_b, val_generator)

y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

y_pred_a = np.argmax(probs_a, axis=1)
y_pred_b = np.argmax(probs_b, axis=1)

# ----- Basic metrics -----
print("=== MODEL A (fine-tuned) ===")
print("Accuracy:", accuracy_score(y_true, y_pred_a))
print(classification_report(y_true, y_pred_a, target_names=class_names, digits=4))

print("\n=== MODEL B (from scratch) ===")
print("Accuracy:", accuracy_score(y_true, y_pred_b))
print(classification_report(y_true, y_pred_b, target_names=class_names, digits=4))

# ----- Confusion Matrices -----
def plot_cm(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(title + " (counts)")
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1, cmap='Blues')
    plt.title(title + " (normalized)")
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.show()

plot_cm(y_true, y_pred_a, class_names, "Model A Confusion Matrix")
plot_cm(y_true, y_pred_b, class_names, "Model B Confusion Matrix")

# ----- Per-class Accuracy -----
def per_class_acc(cm):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(cm) / (cm.sum(axis=1) + 1e-12)

cm_a = confusion_matrix(y_true, y_pred_a)
cm_b = confusion_matrix(y_true, y_pred_b)

print("\nPer-class accuracy Model A:")
for name, acc in zip(class_names, per_class_acc(cm_a)):
    print(f"  {name}: {acc:.3f}")
print("\nPer-class accuracy Model B:")
for name, acc in zip(class_names, per_class_acc(cm_b)):
    print(f"  {name}: {acc:.3f}")

# ----- Expected Calibration Error (ECE) -----
def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(mask):
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.mean()) * abs(bin_conf - bin_acc)
    return ece

ece_a = compute_ece(probs_a, y_true)
ece_b = compute_ece(probs_b, y_true)
print(f"\nECE Model A: {ece_a:.4f}")
print(f"ECE Model B: {ece_b:.4f}")

# ----- Reliability Diagrams -----
def reliability_diagram(probs, labels, n_bins=15, title="Reliability"):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            accs[i] = accuracies[mask].mean()
            confs[i] = confidences[mask].mean()
        else:
            accs[i] = np.nan
            confs[i] = np.nan
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.plot(bin_centers, accs, marker='o', label='accuracy per bin')
    plt.plot(bin_centers, confs, marker='x', label='confidence per bin')
    plt.title(title)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

reliability_diagram(probs_a, y_true, title="Reliability Model A")
reliability_diagram(probs_b, y_true, title="Reliability Model B")

# ----- Temperature Scaling -----
def temperature_scale(logits, labels, init_temp=1.0, lr=0.01, epochs=200):
    temp = tf.Variable(initial_value=tf.constant(init_temp, dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    labels_onehot = tf.one_hot(labels, depth=logits.shape[1])
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            scaled = logits / temp
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=scaled)
            )
        grads = tape.gradient(loss, [temp])
        optimizer.apply_gradients(zip(grads, [temp]))
    return float(temp.numpy())

temp_a = temperature_scale(logits_a, y_true, init_temp=1.0, lr=0.01, epochs=300)
temp_b = temperature_scale(logits_b, y_true, init_temp=1.0, lr=0.01, epochs=300)

print(f"\nTemperature A: {temp_a:.4f}, Temperature B: {temp_b:.4f}")

# Apply scaling & recompute ECE
def apply_temperature(logits, temp):
    scaled = logits / temp
    return tf.nn.softmax(scaled, axis=1).numpy()

probs_a_scaled = apply_temperature(logits_a, temp_a)
probs_b_scaled = apply_temperature(logits_b, temp_b)

print("\nECE Model A after temp scaling:", compute_ece(probs_a_scaled, y_true))
print("ECE Model B after temp scaling:", compute_ece(probs_b_scaled, y_true))
