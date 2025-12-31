"""
EE4065 Homework 5 - Q1: Keyword Spotting from Audio Signals
Section 12.8 Application: Keyword Spotting from Audio Signals (50 points)

This script:
1. Loads the FSDD (Free Spoken Digit Dataset) audio files
2. Extracts MFCC features using CMSIS-DSP compatible functions
3. Trains an MLP (Multi-Layer Perceptron) neural network for digit classification
4. Evaluates the model and displays confusion matrix
5. Saves the trained Keras model for later TFLite conversion

Dataset: Free Spoken Digit Dataset (FSDD)
- 10 digits (0-9) spoken by multiple speakers
- 8000 Hz sample rate
- WAV format
"""

import os
import sys
import numpy as np
import scipy.signal as sig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

# Import MFCC feature extraction function
from mfcc_func import create_mfcc_features

# ============================================================================
# Configuration - Adjust paths as needed
# ============================================================================

# Path to FSDD dataset (Free Spoken Digit Dataset)
FSDD_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
    "Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main",
    "Data", "FSDD")

# Output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# MFCC Feature Extraction Parameters
# ============================================================================

FFTSize = 1024           # FFT window size
sample_rate = 8000       # Audio sample rate in Hz
numOfMelFilters = 20     # Number of Mel filter banks
numOfDctOutputs = 13     # Number of MFCC coefficients
window = sig.get_window("hamming", FFTSize)  # Hamming window

# ============================================================================
# Load and Prepare Dataset
# ============================================================================

print("=" * 60)
print("Q1: KEYWORD SPOTTING FROM AUDIO SIGNALS")
print("=" * 60)
print(f"\nLoading FSDD dataset from: {FSDD_PATH}")

# Get list of all recordings
recordings_list = [os.path.join(FSDD_PATH, rec_path) for rec_path in os.listdir(FSDD_PATH) if rec_path.endswith('.wav')]
print(f"Total recordings found: {len(recordings_list)}")

# Split data: use "yweweler" speaker for testing (leave-one-speaker-out)
test_list = {record for record in recordings_list if "yweweler" in os.path.basename(record)}
train_list = set(recordings_list) - test_list

print(f"Training samples: {len(train_list)}")
print(f"Test samples: {len(test_list)}")

# ============================================================================
# Extract MFCC Features
# ============================================================================

print("\nExtracting MFCC features...")
print(f"  FFT Size: {FFTSize}")
print(f"  Sample Rate: {sample_rate} Hz")
print(f"  Mel Filters: {numOfMelFilters}")
print(f"  MFCC Coefficients: {numOfDctOutputs}")
print(f"  Feature vector size: {numOfDctOutputs * 2}")

train_mfcc_features, train_labels = create_mfcc_features(
    train_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
)
test_mfcc_features, test_labels = create_mfcc_features(
    test_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
)

print(f"\nTraining features shape: {train_mfcc_features.shape}")
print(f"Test features shape: {test_mfcc_features.shape}")

# ============================================================================
# Build MLP Neural Network Model
# ============================================================================

print("\n" + "-" * 60)
print("Building MLP Model")
print("-" * 60)

# MLP Architecture:
# Input: 26 features (13 MFCC × 2 halves)
# Hidden Layer 1: 100 neurons, ReLU activation
# Hidden Layer 2: 100 neurons, ReLU activation
# Output: 10 classes (digits 0-9), Softmax activation

model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[numOfDctOutputs * 2], activation="relu", name="hidden1"),
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

# One-hot encode labels for categorical cross-entropy
ohe = OneHotEncoder(sparse_output=False)
train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1))
categories, test_labels_encoded = np.unique(test_labels, return_inverse=True)

# Compile model
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_mfcc_features, 
    train_labels_ohe, 
    epochs=100, 
    verbose=1,
    validation_split=0.1,
    batch_size=32
)

# ============================================================================
# Evaluate Model
# ============================================================================

print("\n" + "-" * 60)
print("Evaluating Model")
print("-" * 60)

# Predictions
nn_preds = model.predict(test_mfcc_features)
predicted_classes = np.argmax(nn_preds, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == test_labels_encoded)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels_encoded, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# ============================================================================
# Save Model and Results
# ============================================================================

print("\n" + "-" * 60)
print("Saving Model and Results")
print("-" * 60)

# Save Keras model
model_path = os.path.join(MODELS_DIR, "kws_mlp.h5")
model.save(model_path)
print(f"Model saved to: {model_path}")

# ============================================================================
# Visualization
# ============================================================================

# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss plot
axes[0].plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy plot
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

# Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories.astype(int))
cm_display.plot(ax=axes[2], cmap='Blues')
axes[2].set_title(f"Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Q1_results.png"), dpi=150)
#plt.show()

print("\n" + "=" * 60)
print("Q1 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nOutputs:")
print(f"  - Model: {model_path}")
print(f"  - Results plot: {os.path.join(RESULTS_DIR, 'Q1_results.png')}")
print(f"\nNext step: Run 'convert_to_tflite.py' to convert for STM32")

