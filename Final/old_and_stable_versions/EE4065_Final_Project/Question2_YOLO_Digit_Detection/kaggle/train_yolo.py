#!/usr/bin/env python3
"""
Enhanced YOLO-Nano Training Script for ESP32-CAM
Question 2: Handwritten Digit Detection via YOLO

Features:
- Roboflow dataset support (real camera images)
- Improved synthetic data generation
- Strong data augmentation
- Two model outputs: synthetic-only and roboflow

Author: EE4065 Final Project
"""

# ==================== DEPENDENCIES ====================
import subprocess
import sys

def install_packages():
    """Install required packages (for Kaggle/Colab)"""
    packages = ['tensorflow', 'numpy', 'matplotlib', 'opencv-python', 'roboflow', 'albumentations']
    for pkg in packages:
        try:
            module_name = 'cv2' if pkg == 'opencv-python' else pkg
            __import__(module_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install_packages()

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import cv2
import albumentations as A

# ==================== TPU/GPU DETECTION ====================
def setup_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"✓ TPU detected: {tpu.cluster_spec().as_dict()['worker']}")
        return strategy, 'TPU'
    except ValueError:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✓ GPU detected: {len(gpus)} GPU(s)")
            return strategy, 'GPU'
        else:
            strategy = tf.distribute.get_strategy()
            print("⚠ No TPU/GPU detected, using CPU")
            return strategy, 'CPU'

STRATEGY, DEVICE_TYPE = setup_accelerator()

# ==================== CONFIGURATION ====================
INPUT_SIZE = 96
GRID_SIZE = 6
NUM_CLASSES = 10
NUM_ANCHORS = 2
BATCH_SIZE = 64 * STRATEGY.num_replicas_in_sync if DEVICE_TYPE == 'TPU' else 32
EPOCHS = 100  # Daha fazla epoch

ANCHORS = [
    [0.15, 0.3],  # Small digit
    [0.3, 0.6]    # Large digit
]

OUTPUT_DIR = './models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DATA AUGMENTATION ====================
augmentation_pipeline = A.Compose([
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=0.5),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==================== ROBOFLOW DATASET ====================
def download_roboflow_dataset():
    """Download dataset from Roboflow"""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="FSkmmYjYstrWV2v6mJA0")
        project = rf.workspace("labeling-dpvzj").project("my-first-project-7nvw3")
        version = project.version(1)
        dataset = version.download("yolov5")
        print(f"✓ Roboflow dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"⚠ Roboflow download failed: {e}")
        return None

def load_roboflow_data(dataset_path):
    """Load images and labels from Roboflow YOLOv5 format"""
    images = []
    labels = []
    
    for split in ['train', 'valid']:
        img_dir = os.path.join(dataset_path, split, 'images')
        lbl_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_path in glob.glob(os.path.join(img_dir, '*.*')):
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize to input size
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            
            # Load corresponding label
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')
            
            img_labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            img_labels.append({'cls': cls, 'box': [x, y, w, h]})
            
            images.append(img)
            labels.append(img_labels)
    
    print(f"✓ Loaded {len(images)} images from Roboflow dataset")
    return images, labels

# ==================== SYNTHETIC DATA GENERATOR ====================
class ImprovedSyntheticGenerator(keras.utils.Sequence):
    """Enhanced synthetic data with better augmentation"""
    def __init__(self, batch_size=32, samples_per_epoch=10000):
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        
        # Load MNIST
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        # Combine train and test for more variety
        self.x_all = np.concatenate([self.x_train, self.x_test])
        self.y_all = np.concatenate([self.y_train, self.y_test])
        self.num_samples = len(self.x_all)
        
    def __len__(self):
        return self.samples_per_epoch // self.batch_size
    
    def __getitem__(self, index):
        batch_x = np.zeros((self.batch_size, INPUT_SIZE, INPUT_SIZE, 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES), dtype=np.float32)
        
        for i in range(self.batch_size):
            img, labels = self._generate_sample()
            batch_x[i] = img[..., np.newaxis] / 255.0
            self._encode_labels(batch_y[i], labels)
            
        return batch_x, batch_y
    
    def _generate_sample(self):
        # Create realistic background (paper-like)
        bg_type = np.random.choice(['white', 'textured', 'gradient'])
        if bg_type == 'white':
            bg = np.ones((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8) * np.random.randint(230, 255)
        elif bg_type == 'textured':
            bg = np.random.randint(220, 255, (INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (5, 5), 0)
        else:  # gradient
            bg = np.linspace(220, 255, INPUT_SIZE).astype(np.uint8)
            bg = np.tile(bg, (INPUT_SIZE, 1))
        
        # Place 1-3 digits
        num_digits = np.random.randint(1, 4)
        labels = []
        occupied = []  # Track occupied regions to avoid overlap
        
        for _ in range(num_digits):
            idx = np.random.randint(0, self.num_samples)
            digit = self.x_all[idx].copy()
            cls = self.y_all[idx]
            
            # Scale digit
            scale = np.random.uniform(0.8, 2.0)
            new_size = int(28 * scale)
            new_size = min(new_size, INPUT_SIZE - 10)
            digit = cv2.resize(digit, (new_size, new_size))
            
            # Random rotation
            angle = np.random.randint(-20, 20)
            M = cv2.getRotationMatrix2D((new_size/2, new_size/2), angle, 1)
            digit = cv2.warpAffine(digit, M, (new_size, new_size), borderValue=0)
            
            # Find placement (avoid overlap)
            h, w = digit.shape
            max_attempts = 20
            placed = False
            
            for _ in range(max_attempts):
                x = np.random.randint(5, max(6, INPUT_SIZE - w - 5))
                y = np.random.randint(5, max(6, INPUT_SIZE - h - 5))
                
                # Check overlap
                overlap = False
                for ox, oy, ow, oh in occupied:
                    if (x < ox + ow and x + w > ox and y < oy + oh and y + h > oy):
                        overlap = True
                        break
                
                if not overlap:
                    placed = True
                    occupied.append((x, y, w, h))
                    break
            
            if not placed:
                continue
            
            # Blend digit onto background (darken)
            roi = bg[y:y+h, x:x+w]
            mask = digit > 30
            # Invert: MNIST is white-on-black, we want dark-on-light
            roi[mask] = np.minimum(roi[mask], 255 - digit[mask])
            bg[y:y+h, x:x+w] = roi
            
            # Calculate normalized box [cx, cy, w, h]
            cx = (x + w/2) / INPUT_SIZE
            cy = (y + h/2) / INPUT_SIZE
            nw = w / INPUT_SIZE
            nh = h / INPUT_SIZE
            labels.append({'cls': cls, 'box': [cx, cy, nw, nh]})
        
        return bg, labels
    
    def _encode_labels(self, target, labels):
        for label in labels:
            cls = label['cls']
            cx, cy, w, h = label['box']
            
            # Grid cell
            col = int(cx * GRID_SIZE)
            row = int(cy * GRID_SIZE)
            col = min(col, GRID_SIZE - 1)
            row = min(row, GRID_SIZE - 1)
            
            # Anchor assignment (simple: based on size)
            anchor_idx = 0 if h < 0.4 else 1
            
            # Relative to cell
            cell_x = cx * GRID_SIZE - col
            cell_y = cy * GRID_SIZE - row
            
            target[row, col, anchor_idx, 0] = 1.0  # Objectness
            target[row, col, anchor_idx, 1] = cell_x
            target[row, col, anchor_idx, 2] = cell_y
            target[row, col, anchor_idx, 3] = w
            target[row, col, anchor_idx, 4] = h
            target[row, col, anchor_idx, 5 + cls] = 1.0  # Class

# ==================== ROBOFLOW DATA GENERATOR ====================
class RoboflowGenerator(keras.utils.Sequence):
    """Generator for Roboflow dataset with augmentation"""
    def __init__(self, images, labels, batch_size=32, augment=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(images))
        
    def __len__(self):
        return len(self.images) // self.batch_size
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_x = np.zeros((self.batch_size, INPUT_SIZE, INPUT_SIZE, 1), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            img = self.images[idx].copy()
            labels = self.labels[idx]
            
            # Apply augmentation
            if self.augment and len(labels) > 0:
                bboxes = [lbl['box'] for lbl in labels]
                class_labels = [lbl['cls'] for lbl in labels]
                
                try:
                    transformed = augmentation_pipeline(
                        image=img, bboxes=bboxes, class_labels=class_labels
                    )
                    img = transformed['image']
                    new_labels = []
                    for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                        new_labels.append({'cls': cls, 'box': list(bbox)})
                    labels = new_labels
                except:
                    pass  # Use original if augmentation fails
            
            batch_x[i] = img[..., np.newaxis] / 255.0
            self._encode_labels(batch_y[i], labels)
            
        return batch_x, batch_y
    
    def _encode_labels(self, target, labels):
        for label in labels:
            cls = label['cls']
            cx, cy, w, h = label['box']
            
            col = int(cx * GRID_SIZE)
            row = int(cy * GRID_SIZE)
            col = min(max(col, 0), GRID_SIZE - 1)
            row = min(max(row, 0), GRID_SIZE - 1)
            
            anchor_idx = 0 if h < 0.4 else 1
            
            cell_x = cx * GRID_SIZE - col
            cell_y = cy * GRID_SIZE - row
            
            target[row, col, anchor_idx, 0] = 1.0
            target[row, col, anchor_idx, 1] = cell_x
            target[row, col, anchor_idx, 2] = cell_y
            target[row, col, anchor_idx, 3] = w
            target[row, col, anchor_idx, 4] = h
            if 0 <= cls < NUM_CLASSES:
                target[row, col, anchor_idx, 5 + cls] = 1.0

# ==================== MODEL ====================
def create_yolo_nano():
    """Ultra-lightweight YOLO for ESP32 - using ReLU for TFLite Micro compatibility"""
    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    # Stem
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # ReLU instead of LeakyReLU for TFLite Micro
    
    def dw_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    x = dw_block(x, 32, stride=1)
    x = layers.MaxPooling2D(2)(x)  # 24x24
    
    x = dw_block(x, 64, stride=1)
    x = layers.MaxPooling2D(2)(x)  # 12x12
    
    x = dw_block(x, 96, stride=1)
    x = layers.MaxPooling2D(2)(x)  # 6x6
    
    x = dw_block(x, 128, stride=1)  # 6x6x128
    
    # Detection Head
    out_filters = NUM_ANCHORS * (5 + NUM_CLASSES)  # 2 * 15 = 30
    outputs = layers.Conv2D(out_filters, 1, padding='same')(x)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES))(outputs)
    
    return Model(inputs, outputs, name="YOLO_Nano_v2")

# ==================== LOSS FUNCTION ====================
class YoloLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def call(self, y_true, y_pred):
        obj_mask = y_true[..., 0]
        noobj_mask = 1 - obj_mask
        
        true_box = y_true[..., 1:5]
        true_class = y_true[..., 5:]
        
        pred_box = tf.sigmoid(y_pred[..., 1:5])
        pred_obj = tf.sigmoid(y_pred[..., 0])
        pred_class = tf.nn.softmax(y_pred[..., 5:])
        
        # Coordinate Loss
        xy_loss = obj_mask * tf.reduce_sum(tf.square(true_box[..., 0:2] - pred_box[..., 0:2]), axis=-1)
        wh_loss = obj_mask * tf.reduce_sum(tf.square(tf.sqrt(true_box[..., 2:4] + 1e-8) - tf.sqrt(pred_box[..., 2:4] + 1e-8)), axis=-1)
        coord_loss = self.lambda_coord * tf.reduce_sum(xy_loss + wh_loss)
        
        # Objectness Loss (Focal Loss variant)
        obj_loss = tf.reduce_sum(obj_mask * tf.square(1 - pred_obj))
        noobj_loss = self.lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(pred_obj))
        
        # Class Loss
        class_loss = tf.reduce_sum(obj_mask * tf.reduce_sum(tf.square(true_class - pred_class), axis=-1))
        
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        return (coord_loss + obj_loss + noobj_loss + class_loss) / batch_size

# ==================== TRAINING ====================
def train_model(model, train_gen, val_gen, name, epochs=EPOCHS):
    """Train model with callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='loss'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(f'{OUTPUT_DIR}/{name}_best.h5', save_best_only=True, monitor='loss')
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def convert_to_tflite(model, name, representative_gen):
    """Convert to INT8 quantized TFLite"""
    def representative_dataset():
        for _ in range(100):
            data, _ = representative_gen[0]
            yield [data[:1]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # Keep float output for easier post-processing
    
    tflite_model = converter.convert()
    
    path = f'{OUTPUT_DIR}/{name}.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ {name}: {len(tflite_model)/1024:.1f} KB saved to {path}")
    return path

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("  YOLO-Nano Training for ESP32-CAM Digit Detection")
    print("=" * 60)
    
    # 1. Download Roboflow dataset
    print("\n[1/5] Downloading Roboflow dataset...")
    roboflow_path = download_roboflow_dataset()
    
    # 2. Prepare data generators
    print("\n[2/5] Preparing data generators...")
    synthetic_gen = ImprovedSyntheticGenerator(batch_size=BATCH_SIZE, samples_per_epoch=20000)
    
    roboflow_gen = None
    if roboflow_path:
        images, labels = load_roboflow_data(roboflow_path)
        if len(images) > 0:
            # Split into train/val
            split_idx = int(len(images) * 0.9)
            roboflow_train = RoboflowGenerator(images[:split_idx], labels[:split_idx], BATCH_SIZE, augment=True)
            roboflow_val = RoboflowGenerator(images[split_idx:], labels[split_idx:], BATCH_SIZE, augment=False)
            roboflow_gen = roboflow_train
    
    # 3. Train Synthetic Model (backup)
    print("\n[3/5] Training Synthetic Model (backup)...")
    with STRATEGY.scope():
        synthetic_model = create_yolo_nano()
        synthetic_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=YoloLoss())
    
    synthetic_model.summary()
    train_model(synthetic_model, synthetic_gen, None, "yolo_nano_synthetic", epochs=EPOCHS)
    
    # Convert synthetic model
    convert_to_tflite(synthetic_model, "yolo_nano_synthetic_int8", synthetic_gen)
    
    # 4. Train Roboflow Model (if available)
    if roboflow_gen:
        print("\n[4/5] Training Roboflow Model...")
        with STRATEGY.scope():
            roboflow_model = create_yolo_nano()
            roboflow_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=YoloLoss())
        
        train_model(roboflow_model, roboflow_gen, roboflow_val, "yolo_nano_roboflow", epochs=EPOCHS)
        convert_to_tflite(roboflow_model, "yolo_nano_roboflow_int8", roboflow_gen)
    else:
        print("\n[4/5] Skipping Roboflow model (dataset not available)")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\nModels saved in: {OUTPUT_DIR}/")
    print("  - yolo_nano_synthetic_int8.tflite (backup)")
    if roboflow_gen:
        print("  - yolo_nano_roboflow_int8.tflite (recommended)")
    print("\nNext: Run convert_q2_headers.py to create ESP32 header files")

if __name__ == '__main__':
    main()
