"""
EE4065 Homework 5 - Q2: Handwritten Digit Recognition from Digital Images
Section 12.9 Application: Handwritten Digit Recognition from Digital Images (50 points)

This script:
1. Loads the MNIST dataset
2. Extracts Hu Moments features from images (7 rotation-invariant features)
3. Trains an MLP (Multi-Layer Perceptron) neural network for digit classification
4. Evaluates the model and displays confusion matrix
5. Saves the trained Keras model for later TFLite conversion

Dataset: MNIST Handwritten Digits
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 digit classes (0-9)
"""

import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

# Callbacks from keras
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Load MNIST Dataset
# ============================================================================

print("=" * 60)
print("Q2: HANDWRITTEN DIGIT RECOGNITION FROM DIGITAL IMAGES")
print("=" * 60)
print("\nLoading MNIST dataset...")

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

print(f"Training images: {train_images.shape}")
print(f"Test images: {test_images.shape}")
print(f"Image size: {train_images[0].shape}")

# ============================================================================
# Feature Extraction: Hu Moments
# ============================================================================

print("\n" + "-" * 60)
print("Extracting Hu Moments Features")
print("-" * 60)

"""
Hu Moments are a set of 7 numbers calculated using central moments that are 
invariant to image transformations (translation, scale, rotation, and reflection).

These features are:
- Translation invariant (using central moments)
- Scale invariant (using normalized central moments)
- Rotation invariant (using combinations of normalized central moments)

This makes them ideal for digit recognition where the digit might be written
at different sizes or orientations.
"""

print("\nExtracting Hu Moments from training images...")
train_huMoments = np.empty((len(train_images), 7))
for train_idx, train_img in enumerate(train_images):
    train_moments = cv2.moments(train_img, True)  # True = binary image moments
    train_huMoments[train_idx] = cv2.HuMoments(train_moments).reshape(7)
    if (train_idx + 1) % 10000 == 0:
        print(f"  Processed {train_idx + 1}/{len(train_images)} training images")

print("\nExtracting Hu Moments from test images...")
test_huMoments = np.empty((len(test_images), 7))
for test_idx, test_img in enumerate(test_images):
    test_moments = cv2.moments(test_img, True)
    test_huMoments[test_idx] = cv2.HuMoments(test_moments).reshape(7)
    if (test_idx + 1) % 5000 == 0:
        print(f"  Processed {test_idx + 1}/{len(test_images)} test images")

print(f"\nFeature shape: {train_huMoments.shape}")

# ============================================================================
# Feature Normalization (Optional but recommended)
# ============================================================================

# Note: The book version doesn't normalize, but normalization can help
# You can uncomment the following lines if you want to normalize:
# 
# features_mean = np.mean(train_huMoments, axis=0)
# features_std = np.std(train_huMoments, axis=0)
# train_huMoments = (train_huMoments - features_mean) / features_std
# test_huMoments = (test_huMoments - features_mean) / features_std
# print("Features normalized (z-score normalization)")

# ============================================================================
# Build MLP Neural Network Model
# ============================================================================

print("\n" + "-" * 60)
print("Building MLP Model")
print("-" * 60)

# MLP Architecture:
# Input: 7 features (Hu Moments)
# Hidden Layer 1: 100 neurons, ReLU activation
# Hidden Layer 2: 100 neurons, ReLU activation
# Output: 10 classes (digits 0-9), Softmax activation

model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[7], activation="relu", name="hidden1"),
    keras.layers.Dense(100, activation="relu", name="hidden2"),
    keras.layers.Dense(10, activation="softmax", name="output")
])

model.summary()

# ============================================================================
# Train the Model
# ============================================================================

print("\n" + "-" * 60)
print("Training Model")
print("-" * 60)

model_save_path = os.path.join(MODELS_DIR, "hdr_mlp.h5")
categories = np.unique(test_labels)

# Compile model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Callbacks
mc_callback = ModelCheckpoint(
    model_save_path, 
    save_best_only=True, 
    monitor='val_loss',
    verbose=1
)
es_callback = EarlyStopping(
    monitor='loss', 
    patience=5, 
    verbose=1,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_huMoments, 
    train_labels, 
    epochs=200,  # Will stop early if loss plateaus
    verbose=1, 
    callbacks=[mc_callback, es_callback],
    validation_split=0.1,
    batch_size=128
)

# Load the best model
model = keras.models.load_model(model_save_path)

# ============================================================================
# Evaluate Model
# ============================================================================

print("\n" + "-" * 60)
print("Evaluating Model")
print("-" * 60)

# Predictions
nn_preds = model.predict(test_huMoments)
predicted_classes = np.argmax(nn_preds, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == test_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Per-class accuracy
print("\nPer-class accuracy:")
for i in range(10):
    class_acc = conf_matrix[i, i] / conf_matrix[i].sum()
    print(f"  Digit {i}: {class_acc * 100:.2f}%")

# ============================================================================
# Save Model
# ============================================================================

print("\n" + "-" * 60)
print("Saving Model")
print("-" * 60)

print(f"Model saved to: {model_save_path}")

# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Training Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Training Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
cm_display.plot(ax=axes[0, 2], cmap='Blues')
axes[0, 2].set_title(f"Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")

# Sample predictions
axes[1, 0].set_title("Sample Test Images and Predictions")
for i in range(5):
    idx = np.random.randint(0, len(test_images))
    ax_img = fig.add_axes([0.05 + i*0.06, 0.15, 0.05, 0.15])
    ax_img.imshow(test_images[idx], cmap='gray')
    pred = predicted_classes[idx]
    true = test_labels[idx]
    color = 'green' if pred == true else 'red'
    ax_img.set_title(f"P:{pred}/T:{true}", fontsize=8, color=color)
    ax_img.axis('off')
axes[1, 0].axis('off')

# Hu Moments visualization for one sample
sample_idx = 0
axes[1, 1].bar(range(7), train_huMoments[sample_idx])
axes[1, 1].set_xlabel('Hu Moment Index')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title(f'Hu Moments for Training Sample (Digit: {train_labels[sample_idx]})')
axes[1, 1].grid(True, axis='y')

# Model architecture summary
arch_text = """
Model Architecture:
─────────────────────
Input: 7 (Hu Moments)
  │
  ▼
Dense(100, ReLU)
  │
  ▼
Dense(100, ReLU)
  │
  ▼
Dense(10, Softmax)
  │
  ▼
Output: Digit (0-9)
"""
axes[1, 2].text(0.1, 0.5, arch_text, fontsize=10, family='monospace', 
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].set_title('MLP Architecture')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Q2_results.png"), dpi=150)
#plt.show()

print("\n" + "=" * 60)
print("Q2 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nOutputs:")
print(f"  - Model: {model_save_path}")
print(f"  - Results plot: {os.path.join(RESULTS_DIR, 'Q2_results.png')}")
print(f"\nNext step: Run 'convert_to_tflite.py' to convert for STM32")

