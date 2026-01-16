"""
EE 4065 - Embedded Digital Image Processing - Homework 6
Section 13.7 Application: Handwritten Digit Recognition from Digital Images

This script trains all CNN models (SqueezeNet, EfficientNet, MobileNet, ResNet, ShuffleNet)
on the MNIST dataset for deployment on STM32 Nucleo-F446RE.

Author: Student
Date: 2026
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.squeezenet import SqueezeNet, SqueezeNetMini
from models.efficientnet import EfficientNet, EfficientNetMini
from models.mobilenet import MobileNetV2, MobileNetV2Mini
from models.resnet import ResNet, ResNet8, ResNet14, ResNet20
from models.shufflenet import ShuffleNet, ShuffleNetMini

# Configuration
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)  # Resize MNIST to 32x32 RGB
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

# Create save directory
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def prepare_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    (train_images, train_labels), (val_images, val_labels) = keras.datasets.mnist.load_data()
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Preprocess: expand to 3 channels and resize to 32x32
    def preprocess(images):
        # Add channel dimension
        images = np.expand_dims(images, axis=-1)
        # Repeat to 3 channels
        images = np.repeat(images, 3, axis=-1)
        # Resize to 32x32
        images = tf.image.resize(images, (32, 32)).numpy()
        # Normalize to [0, 1]
        images = images / 255.0
        return images.astype(np.float32)
    
    train_images = preprocess(train_images)
    val_images = preprocess(val_images)
    
    # One-hot encode labels
    train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    val_labels = keras.utils.to_categorical(val_labels, NUM_CLASSES)
    
    print(f"Input shape: {train_images.shape[1:]}")
    
    return (train_images, train_labels), (val_images, val_labels)


def get_callbacks(model_name):
    """Create training callbacks"""
    model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.h5")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, model_name, train_data, val_data):
    """Train a single model"""
    (train_images, train_labels) = train_data
    (val_images, val_labels) = val_data
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Print model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get callbacks
    callbacks = get_callbacks(model_name)
    
    # Train
    history = model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\n{model_name} - Validation Accuracy: {val_acc*100:.2f}%")
    
    return history, val_acc


def get_all_models():
    """
    Get all models to train.
    Returns dictionary of model_name: model
    
    Note: For STM32 Nucleo-F446RE with limited memory (128KB RAM, 512KB Flash),
    we include both standard and 'Mini' variants. The Mini variants are 
    specifically designed to fit in the constrained memory.
    """
    models = {}
    
    # SqueezeNet variants
    models['SqueezeNet'] = SqueezeNet(INPUT_SHAPE, NUM_CLASSES, weights=None, dropout=0.5)
    models['SqueezeNetMini'] = SqueezeNetMini(INPUT_SHAPE, NUM_CLASSES, dropout=0.3)
    
    # EfficientNet variants
    models['EfficientNet'] = EfficientNet(INPUT_SHAPE, NUM_CLASSES, width_coefficient=0.45)
    models['EfficientNetMini'] = EfficientNetMini(INPUT_SHAPE, NUM_CLASSES)
    
    # MobileNetV2 variants
    models['MobileNetV2'] = MobileNetV2(INPUT_SHAPE, NUM_CLASSES, alpha=0.35, dropout=0.2)
    models['MobileNetV2Mini'] = MobileNetV2Mini(INPUT_SHAPE, NUM_CLASSES, dropout=0.2)
    
    # ResNet variants (different depths)
    models['ResNet8'] = ResNet8(INPUT_SHAPE, NUM_CLASSES, dropout=0.2)
    models['ResNet14'] = ResNet14(INPUT_SHAPE, NUM_CLASSES, dropout=0.2)
    models['ResNet20'] = ResNet20(INPUT_SHAPE, NUM_CLASSES, dropout=0.2)
    
    # ShuffleNet variants
    models['ShuffleNet'] = ShuffleNet(INPUT_SHAPE, NUM_CLASSES, scale_factor=1.0)
    models['ShuffleNetMini'] = ShuffleNetMini(INPUT_SHAPE, NUM_CLASSES)
    
    return models


def plot_training_history(histories, save_path):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy
    ax1 = axes[0]
    for name, history in histories.items():
        ax1.plot(history.history['val_accuracy'], label=name)
    ax1.set_title('Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2 = axes[1]
    for name, history in histories.items():
        ax2.plot(history.history['val_loss'], label=name)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nTraining history plot saved to: {save_path}")


def main():
    """Main training function"""
    print("=" * 60)
    print("EE 4065 - Homework 6: Handwritten Digit Recognition")
    print("Training CNN Models for STM32 Nucleo-F446RE")
    print("=" * 60)
    
    # Check TensorFlow version
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    
    # Load data
    train_data, val_data = prepare_data()
    
    # Get all models
    models = get_all_models()
    
    # Train each model
    results = {}
    histories = {}
    
    for name, model in models.items():
        try:
            history, accuracy = train_model(model, name, train_data, val_data)
            results[name] = accuracy
            histories[name] = history
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = 0.0
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<15} {'Parameters':<15}")
    print("-" * 55)
    
    for name in sorted(results.keys(), key=lambda x: results[x], reverse=True):
        model = models[name]
        params = model.count_params()
        print(f"{name:<25} {results[name]*100:>6.2f}%        {params:>12,}")
    
    # Plot training history
    if histories:
        plot_path = os.path.join(MODEL_SAVE_DIR, 'training_history.png')
        plot_training_history(histories, plot_path)
    
    print(f"\nModels saved to: {MODEL_SAVE_DIR}")
    print("\nNext step: Run convert_to_tflite.py to convert models for STM32")


if __name__ == "__main__":
    main()

