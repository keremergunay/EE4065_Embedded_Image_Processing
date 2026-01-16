#!/usr/bin/env python3
"""
YOLO-Nano Training - ROBOFLOW ONLY
Already have synthetic model? Run this instead to train only on Roboflow data.
"""

import subprocess
import sys

def install_packages():
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
EPOCHS = 100

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
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# ==================== ROBOFLOW DATASET ====================
def download_roboflow_dataset():
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
    images = []
    labels = []
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(dataset_path, split, 'images')
        lbl_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_path in glob.glob(os.path.join(img_dir, '*.*')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + '.txt')
            
            img_labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(float(parts[0]))  # FIX: Convert to int properly
                            x, y, w, h = map(float, parts[1:5])
                            # Clamp values to valid range
                            x = max(0.0, min(1.0, x))
                            y = max(0.0, min(1.0, y))
                            w = max(0.01, min(1.0, w))
                            h = max(0.01, min(1.0, h))
                            if 0 <= cls < NUM_CLASSES:
                                img_labels.append({'cls': cls, 'box': [x, y, w, h]})
            
            images.append(img)
            labels.append(img_labels)
    
    print(f"✓ Loaded {len(images)} images from Roboflow dataset")
    return images, labels

# ==================== DATA GENERATOR ====================
class RoboflowGenerator(keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=32, augment=True):
        super().__init__()  # FIX: Call parent init
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(images))
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return max(1, len(self.images) // self.batch_size)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.images))
        batch_indices = self.indices[start:end]
        
        actual_batch = len(batch_indices)
        batch_x = np.zeros((actual_batch, INPUT_SIZE, INPUT_SIZE, 1), dtype=np.float32)
        batch_y = np.zeros((actual_batch, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES), dtype=np.float32)
        
        for i, idx in enumerate(batch_indices):
            img = self.images[idx].copy()
            labels = self.labels[idx].copy()
            
            # Apply augmentation
            if self.augment and len(labels) > 0:
                try:
                    bboxes = [lbl['box'] for lbl in labels]
                    class_labels = [int(lbl['cls']) for lbl in labels]  # FIX: Ensure int
                    
                    transformed = augmentation_pipeline(
                        image=img, bboxes=bboxes, class_labels=class_labels
                    )
                    img = transformed['image']
                    new_labels = []
                    for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                        new_labels.append({'cls': int(cls), 'box': list(bbox)})  # FIX: int(cls)
                    labels = new_labels
                except Exception as e:
                    pass  # Use original if augmentation fails
            
            batch_x[i] = img[..., np.newaxis] / 255.0
            self._encode_labels(batch_y[i], labels)
            
        return batch_x, batch_y
    
    def _encode_labels(self, target, labels):
        for label in labels:
            cls = int(label['cls'])  # FIX: Ensure integer
            cx, cy, w, h = label['box']
            
            # Validate
            if not (0 <= cls < NUM_CLASSES):
                continue
            if not (0 < cx < 1 and 0 < cy < 1):
                continue
                
            col = int(cx * GRID_SIZE)
            row = int(cy * GRID_SIZE)
            col = min(max(col, 0), GRID_SIZE - 1)
            row = min(max(row, 0), GRID_SIZE - 1)
            
            anchor_idx = 0 if h < 0.4 else 1
            
            cell_x = cx * GRID_SIZE - col
            cell_y = cy * GRID_SIZE - row
            
            target[row, col, anchor_idx, 0] = 1.0  # Objectness
            target[row, col, anchor_idx, 1] = cell_x
            target[row, col, anchor_idx, 2] = cell_y
            target[row, col, anchor_idx, 3] = w
            target[row, col, anchor_idx, 4] = h
            target[row, col, anchor_idx, 5 + cls] = 1.0  # Class

# ==================== MODEL ====================
def create_yolo_nano():
    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    def dw_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    x = dw_block(x, 32, stride=1)
    x = layers.MaxPooling2D(2)(x)
    
    x = dw_block(x, 64, stride=1)
    x = layers.MaxPooling2D(2)(x)
    
    x = dw_block(x, 96, stride=1)
    x = layers.MaxPooling2D(2)(x)
    
    x = dw_block(x, 128, stride=1)
    
    out_filters = NUM_ANCHORS * (5 + NUM_CLASSES)
    outputs = layers.Conv2D(out_filters, 1, padding='same')(x)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 5 + NUM_CLASSES))(outputs)
    
    return Model(inputs, outputs, name="YOLO_Nano_v2")

# ==================== LOSS ====================
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
        
        xy_loss = obj_mask * tf.reduce_sum(tf.square(true_box[..., 0:2] - pred_box[..., 0:2]), axis=-1)
        wh_loss = obj_mask * tf.reduce_sum(tf.square(tf.sqrt(true_box[..., 2:4] + 1e-8) - tf.sqrt(pred_box[..., 2:4] + 1e-8)), axis=-1)
        coord_loss = self.lambda_coord * tf.reduce_sum(xy_loss + wh_loss)
        
        obj_loss = tf.reduce_sum(obj_mask * tf.square(1 - pred_obj))
        noobj_loss = self.lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(pred_obj))
        
        class_loss = tf.reduce_sum(obj_mask * tf.reduce_sum(tf.square(true_class - pred_class), axis=-1))
        
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        return (coord_loss + obj_loss + noobj_loss + class_loss) / batch_size

# ==================== TRAINING ====================
def convert_to_tflite(model, name, representative_gen):
    def representative_dataset():
        for _ in range(100):
            data, _ = representative_gen[0]
            yield [data[:1]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    path = f'{OUTPUT_DIR}/{name}.tflite'
    with open(path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ {name}: {len(tflite_model)/1024:.1f} KB saved to {path}")

def main():
    print("=" * 60)
    print("  YOLO-Nano Training - ROBOFLOW ONLY")
    print("=" * 60)
    
    # Download Roboflow dataset
    print("\n[1/3] Downloading Roboflow dataset...")
    roboflow_path = download_roboflow_dataset()
    
    if not roboflow_path:
        print("ERROR: Could not download Roboflow dataset!")
        return
    
    # Load data
    print("\n[2/3] Loading data...")
    images, labels = load_roboflow_data(roboflow_path)
    
    if len(images) < 10:
        print(f"ERROR: Not enough images! Found only {len(images)}")
        return
    
    # Split train/val
    split_idx = int(len(images) * 0.85)
    train_gen = RoboflowGenerator(images[:split_idx], labels[:split_idx], BATCH_SIZE, augment=True)
    val_gen = RoboflowGenerator(images[split_idx:], labels[split_idx:], BATCH_SIZE, augment=False)
    
    print(f"  Train: {split_idx} images, Val: {len(images) - split_idx} images")
    
    # Train
    print("\n[3/3] Training Roboflow Model...")
    with STRATEGY.scope():
        model = create_yolo_nano()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=YoloLoss())
    
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='loss'),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(f'{OUTPUT_DIR}/yolo_nano_roboflow_best.h5', save_best_only=True, monitor='loss')
    ]
    
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    
    # Convert to TFLite
    convert_to_tflite(model, "yolo_nano_roboflow_int8", train_gen)
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\nModel saved: {OUTPUT_DIR}/yolo_nano_roboflow_int8.tflite")
    print("Next: python convert_q2_headers.py roboflow")

if __name__ == '__main__':
    main()
