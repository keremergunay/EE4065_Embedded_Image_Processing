#!/usr/bin/env python3
"""
ESP32-CAM Digit Recognition - Model Training Script
Question 4: Multi-Model Recognition

This script trains 4 lightweight CNN models for ESP32-CAM deployment:
1. SqueezeNetMini - Fire modules based
2. MobileNetV2Mini - Depthwise separable convolutions  
3. ResNet8 - Residual connections
4. EfficientNetMini - Compound scaling

Training Data: MNIST + Augmentation matching ESP32 preprocessing
Output: Quantized TFLite models (uint8 input, int8 weights)

Usage (Kaggle with TPU):
  1. Select TPU v3-8 as accelerator
  2. Run all cells

Author: EE4065 Final Project
"""

# ==================== DEPENDENCIES ====================
import subprocess
import sys

def install_packages():
    """Install required packages (for Kaggle/Colab)"""
    packages = ['tensorflow', 'numpy', 'matplotlib']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install_packages()

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ==================== TPU/GPU DETECTION ====================
def setup_accelerator():
    """Setup TPU if available, otherwise GPU/CPU"""
    try:
        # Try to detect TPU (Kaggle/Colab)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"✓ TPU detected: {tpu.cluster_spec().as_dict()['worker']}")
        print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
        return strategy, 'TPU'
    except ValueError:
        # No TPU, check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✓ GPU detected: {len(gpus)} GPU(s)")
            return strategy, 'GPU'
        else:
            strategy = tf.distribute.get_strategy()
            print("⚠ No TPU/GPU detected, using CPU")
            return strategy, 'CPU'

# Setup accelerator
STRATEGY, DEVICE_TYPE = setup_accelerator()

# ==================== CONFIGURATION ====================
INPUT_SIZE = 32  # Model input size (matches ESP32)
NUM_CLASSES = 10
# Increase batch size for TPU (128 * 8 replicas = 1024)
BATCH_SIZE = 128 * STRATEGY.num_replicas_in_sync if DEVICE_TYPE == 'TPU' else 128
EPOCHS = 150

print(f"Batch size: {BATCH_SIZE}")

# Output directory
OUTPUT_DIR = './trained_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DATA PREPROCESSING ====================
def preprocess_mnist_for_esp32(x_train, y_train, x_test, y_test):
    """
    Preprocess MNIST to match ESP32-CAM pipeline:
    - MNIST original: white digit (255) on black background (0) - 28x28
    - ESP32 output: white digit (255) on black background (0) - 32x32
    - Model input: 32x32x3 RGB (grayscale duplicated to 3 channels)
    """
    print("Preprocessing MNIST data...")
    
    # Resize from 28x28 to 32x32
    x_train = tf.image.resize(x_train[..., np.newaxis], (INPUT_SIZE, INPUT_SIZE)).numpy()
    x_test = tf.image.resize(x_test[..., np.newaxis], (INPUT_SIZE, INPUT_SIZE)).numpy()
    
    # Convert to 3 channels (grayscale -> RGB)
    x_train = np.repeat(x_train, 3, axis=-1)
    x_test = np.repeat(x_test, 3, axis=-1)
    
    # Normalize to [0, 1] - model expects this range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    return x_train, y_train, x_test, y_test

def create_augmentation_layer():
    """
    Data augmentation matching real-world conditions:
    - Slight rotation (handwriting angle variations)
    - Small translation (centering variations)
    - Noise (camera noise)
    """
    return keras.Sequential([
        layers.RandomRotation(0.1),  # ±18 degrees
        layers.RandomTranslation(0.1, 0.1),  # ±10% shift
        layers.RandomZoom(0.1),  # ±10% zoom
        layers.GaussianNoise(0.05),  # Camera noise
    ], name='augmentation')

# ==================== MODEL ARCHITECTURES ====================

def create_squeezenet_mini(input_shape=(32, 32, 3), num_classes=10):
    """
    SqueezeNetMini: Lightweight model using Fire modules
    ~55KB quantized
    """
    def fire_module(x, squeeze_filters, expand_filters, name):
        # Squeeze layer
        squeeze = layers.Conv2D(squeeze_filters, 1, activation='relu', 
                               padding='same', name=f'{name}_squeeze')(x)
        # Expand layers
        expand_1x1 = layers.Conv2D(expand_filters, 1, activation='relu',
                                   padding='same', name=f'{name}_expand1x1')(squeeze)
        expand_3x3 = layers.Conv2D(expand_filters, 3, activation='relu',
                                   padding='same', name=f'{name}_expand3x3')(squeeze)
        return layers.Concatenate(name=f'{name}_concat')([expand_1x1, expand_3x3])
    
    inputs = layers.Input(shape=input_shape, name='input_layer_1')
    
    # Initial convolution
    x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D(2, name='pool1')(x)  # 8x8
    
    # Fire modules
    x = fire_module(x, 8, 16, 'fire1')  # -> 32 channels
    x = fire_module(x, 8, 16, 'fire2')  # -> 32 channels
    x = layers.MaxPooling2D(2, name='pool2')(x)  # 4x4
    
    x = fire_module(x, 16, 32, 'fire3')  # -> 64 channels
    x = fire_module(x, 16, 32, 'fire4')  # -> 64 channels
    
    # Output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs, outputs, name='SqueezeNetMini')


def create_mobilenet_mini(input_shape=(32, 32, 3), num_classes=10):
    """
    MobileNetV2Mini: Depthwise separable convolutions with inverted residuals
    ~110KB quantized
    """
    def inverted_residual(x, filters, stride, expand_ratio, name):
        in_channels = x.shape[-1]
        expand_filters = in_channels * expand_ratio
        
        # Expand
        if expand_ratio != 1:
            expand = layers.Conv2D(expand_filters, 1, padding='same', name=f'{name}_expand')(x)
            expand = layers.BatchNormalization(name=f'{name}_expand_bn')(expand)
            expand = layers.ReLU(6.0, name=f'{name}_expand_relu')(expand)
        else:
            expand = x
        
        # Depthwise
        dw = layers.DepthwiseConv2D(3, strides=stride, padding='same', name=f'{name}_dw')(expand)
        dw = layers.BatchNormalization(name=f'{name}_dw_bn')(dw)
        dw = layers.ReLU(6.0, name=f'{name}_dw_relu')(dw)
        
        # Project
        proj = layers.Conv2D(filters, 1, padding='same', name=f'{name}_proj')(dw)
        proj = layers.BatchNormalization(name=f'{name}_proj_bn')(proj)
        
        # Residual connection
        if stride == 1 and in_channels == filters:
            return layers.Add(name=f'{name}_add')([x, proj])
        return proj
    
    inputs = layers.Input(shape=input_shape, name='input_layer_1')
    
    # Initial conv
    x = layers.Conv2D(16, 3, strides=2, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6.0, name='conv1_relu')(x)  # 16x16
    
    # Inverted residual blocks
    x = inverted_residual(x, 16, 1, 1, 'ir1')
    x = inverted_residual(x, 24, 2, 6, 'ir2')  # 8x8
    x = inverted_residual(x, 24, 1, 6, 'ir3')
    x = inverted_residual(x, 32, 2, 6, 'ir4')  # 4x4
    x = inverted_residual(x, 32, 1, 6, 'ir5')
    x = inverted_residual(x, 64, 1, 6, 'ir6')
    
    # Output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs, outputs, name='MobileNetV2Mini')


def create_resnet8(input_shape=(32, 32, 3), num_classes=10):
    """
    ResNet8: Simple residual network with 8 layers
    ~100KB quantized
    """
    def residual_block(x, filters, stride, name):
        shortcut = x
        
        # First conv
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', name=f'{name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.ReLU(name=f'{name}_relu1')(x)
        
        # Second conv
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        
        # Shortcut
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', 
                                     name=f'{name}_shortcut')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
        
        x = layers.Add(name=f'{name}_add')([x, shortcut])
        x = layers.ReLU(name=f'{name}_relu2')(x)
        return x
    
    inputs = layers.Input(shape=input_shape, name='input_layer_1')
    
    # Initial conv
    x = layers.Conv2D(16, 3, padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)  # 32x32
    
    # Residual blocks
    x = residual_block(x, 16, 1, 'res1')  # 32x32
    x = residual_block(x, 32, 2, 'res2')  # 16x16
    x = residual_block(x, 64, 2, 'res3')  # 8x8
    
    # Output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs, outputs, name='ResNet8')


def create_efficientnet_mini(input_shape=(32, 32, 3), num_classes=10):
    """
    EfficientNetMini: Simplified EfficientNet with MBConv blocks
    ~110KB quantized
    """
    def mbconv_block(x, filters, expand_ratio, stride, se_ratio=0.25, name=''):
        in_channels = x.shape[-1]
        expand_filters = in_channels * expand_ratio
        
        # Expansion
        if expand_ratio != 1:
            x_exp = layers.Conv2D(expand_filters, 1, padding='same', name=f'{name}_expand')(x)
            x_exp = layers.BatchNormalization(name=f'{name}_expand_bn')(x_exp)
            x_exp = layers.Activation('swish', name=f'{name}_expand_act')(x_exp)
        else:
            x_exp = x
        
        # Depthwise
        x_dw = layers.DepthwiseConv2D(3, strides=stride, padding='same', name=f'{name}_dw')(x_exp)
        x_dw = layers.BatchNormalization(name=f'{name}_dw_bn')(x_dw)
        x_dw = layers.Activation('swish', name=f'{name}_dw_act')(x_dw)
        
        # Squeeze-and-Excitation
        se_filters = max(1, int(in_channels * se_ratio))
        se = layers.GlobalAveragePooling2D(name=f'{name}_se_gap')(x_dw)
        se = layers.Dense(se_filters, activation='swish', name=f'{name}_se_reduce')(se)
        se = layers.Dense(expand_filters, activation='sigmoid', name=f'{name}_se_expand')(se)
        se = layers.Reshape((1, 1, expand_filters), name=f'{name}_se_reshape')(se)
        x_se = layers.Multiply(name=f'{name}_se_mul')([x_dw, se])
        
        # Project
        x_proj = layers.Conv2D(filters, 1, padding='same', name=f'{name}_proj')(x_se)
        x_proj = layers.BatchNormalization(name=f'{name}_proj_bn')(x_proj)
        
        # Residual
        if stride == 1 and in_channels == filters:
            return layers.Add(name=f'{name}_add')([x, x_proj])
        return x_proj
    
    inputs = layers.Input(shape=input_shape, name='input_layer_1')
    
    # Stem
    x = layers.Conv2D(16, 3, strides=2, padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_act')(x)  # 16x16
    
    # MBConv blocks
    x = mbconv_block(x, 16, 1, 1, name='mb1')
    x = mbconv_block(x, 24, 6, 2, name='mb2')  # 8x8
    x = mbconv_block(x, 24, 6, 1, name='mb3')
    x = mbconv_block(x, 32, 6, 2, name='mb4')  # 4x4
    x = mbconv_block(x, 32, 6, 1, name='mb5')
    
    # Head
    x = layers.Conv2D(64, 1, padding='same', name='head_conv')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Activation('swish', name='head_act')(x)
    
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs, outputs, name='EfficientNetMini')


# ==================== TRAINING ====================

def train_model(model, x_train, y_train, x_val, y_val, augmentation=None, name='model'):
    """Train a model with callbacks and augmentation"""
    
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'{OUTPUT_DIR}/{name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Note: model.compile() is called in main() inside strategy scope
    
    # Create augmented dataset if augmentation provided
    if augmentation:
        # Apply augmentation to training data
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)
        train_dataset = train_dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    return history


def convert_to_tflite(model, x_representative, name):
    """
    Convert Keras model to quantized TFLite
    - uint8 input (matches ESP32 preprocessing output)
    - int8 weights for efficiency
    """
    print(f"\nConverting {name} to TFLite...")
    
    # Representative dataset for quantization
    def representative_dataset():
        for i in range(min(1000, len(x_representative))):
            sample = x_representative[i:i+1].astype(np.float32)
            yield [sample]
    
    # Convert with full integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # Match ESP32 input
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save
    tflite_path = f'{OUTPUT_DIR}/{name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"  Saved: {tflite_path} ({size_kb:.1f} KB)")
    
    # Verify quantization parameters
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"  Input: {input_details['shape']}, dtype={input_details['dtype']}")
    print(f"  Input quant: scale={input_details['quantization_parameters']['scales'][0]:.6f}, "
          f"zp={input_details['quantization_parameters']['zero_points'][0]}")
    print(f"  Output: {output_details['shape']}, dtype={output_details['dtype']}")
    
    return tflite_path


def evaluate_tflite(tflite_path, x_test, y_test):
    """Evaluate TFLite model accuracy"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Get quantization parameters
    input_scale = input_details['quantization_parameters']['scales'][0]
    input_zp = input_details['quantization_parameters']['zero_points'][0]
    
    correct = 0
    for i in range(len(x_test)):
        # Quantize input
        input_data = (x_test[i:i+1] / input_scale + input_zp).astype(np.uint8)
        
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]
        
        pred = np.argmax(output)
        true = np.argmax(y_test[i])
        if pred == true:
            correct += 1
    
    accuracy = correct / len(x_test) * 100
    print(f"  TFLite accuracy: {accuracy:.2f}%")
    return accuracy


# ==================== MAIN ====================

def main():
    print("="*60)
    print("ESP32-CAM Digit Recognition - Model Training")
    print("="*60)
    
    # Load MNIST
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess
    x_train, y_train, x_test, y_test = preprocess_mnist_for_esp32(
        x_train, y_train, x_test, y_test
    )
    
    # Split validation
    val_split = 5000
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]
    
    # Create augmentation
    augmentation = create_augmentation_layer()
    
    # Define models
    models = [
        ('SqueezeNetMini', create_squeezenet_mini),
        ('MobileNetV2Mini', create_mobilenet_mini),
        ('ResNet8', create_resnet8),
        ('EfficientNetMini', create_efficientnet_mini),
    ]
    
    results = []
    
    for name, create_fn in models:
        # Create and compile model inside strategy scope for TPU/multi-GPU support
        with STRATEGY.scope():
            model = create_fn()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Train (datasets work outside scope)
        history = train_model(model, x_train, y_train, x_val, y_val, 
                             augmentation=augmentation, name=name)
        
        # Evaluate Keras model
        _, keras_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"  Keras accuracy: {keras_acc*100:.2f}%")
        
        # Convert to TFLite
        tflite_path = convert_to_tflite(model, x_train, name)
        
        # Evaluate TFLite
        tflite_acc = evaluate_tflite(tflite_path, x_test, y_test)
        
        results.append({
            'name': name,
            'keras_acc': keras_acc * 100,
            'tflite_acc': tflite_acc,
            'size_kb': os.path.getsize(tflite_path) / 1024
        })
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Keras Acc':>10} {'TFLite Acc':>12} {'Size (KB)':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['keras_acc']:>9.2f}% {r['tflite_acc']:>11.2f}% {r['size_kb']:>9.1f}")
    
    print(f"\nModels saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run convert_models.py to generate ESP32 header files")


if __name__ == '__main__':
    main()
