#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MicroYOLO v3 Training - KAGGLE VERSION
ESP32-CAM için Digit Detection modeli
GPU ile ~1-2 saat

Kaggle'da çalıştırmak için:
1. kaggle.com/notebooks adresine git
2. "New Notebook" tıkla
3. Sağ üstten Settings > Accelerator > GPU P100 seç
4. Settings > Internet > On (internet gerekli)
5. Bu dosyayı notebook'a yapıştır veya upload et
6. Run All
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
import random
import warnings
import urllib.request
import scipy.io
warnings.filterwarnings('ignore')

print('='*60)
print('MicroYOLO v3 Training - KAGGLE')
print('='*60)
print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')

# Kaggle çıktı dizini
OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== AYARLAR ====================
IMG_SIZE = 96
GRID_SIZE = 6
NUM_CLASSES = 10

BATCH_SIZE = 64
EPOCHS = 250
LEARNING_RATE = 0.001
TRAIN_SAMPLES = 40000
VAL_SAMPLES = 5000

print(f'Ayarlar: {EPOCHS} epoch, {TRAIN_SAMPLES} train, {VAL_SAMPLES} val')

# ==================== SVHN YUKLE ====================
print('\nSVHN yukleniyor...')

def download_svhn(max_retries=3):
    svhn_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    svhn_path = os.path.join(OUTPUT_DIR, 'svhn_train.mat')
    EXPECTED_SIZE = 26_000_000
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(svhn_path):
                if os.path.getsize(svhn_path) < EXPECTED_SIZE:
                    print(f'  Bozuk dosya, siliniyor...')
                    os.remove(svhn_path)
            
            if not os.path.exists(svhn_path):
                print(f'  Indiriliyor (deneme {attempt + 1}/{max_retries})...')
                urllib.request.urlretrieve(svhn_url, svhn_path)
                print(f'  Indirildi: {os.path.getsize(svhn_path)/1e6:.1f}MB')
            
            mat = scipy.io.loadmat(svhn_path)
            print('  SVHN dogrulandi!')
            
            X = mat['X']
            y = mat['y'].flatten()
            y[y == 10] = 0
            X = np.transpose(X, (3, 0, 1, 2))
            X_gray = np.mean(X, axis=3).astype(np.uint8)
            return X_gray, y
            
        except Exception as e:
            print(f'  Deneme {attempt + 1} basarisiz: {e}')
            if os.path.exists(svhn_path):
                os.remove(svhn_path)
            if attempt == max_retries - 1:
                raise RuntimeError(f'{max_retries} denemeden sonra SVHN indirilemedi')

svhn_x, svhn_y = download_svhn()

svhn_by_class = {}
for i in range(10):
    svhn_by_class[i] = svhn_x[svhn_y == i]
    print(f'  SVHN Sinif {i}: {len(svhn_by_class[i])} ornek')
print(f'SVHN toplam: {len(svhn_x)} ornek')

# ==================== GORUNTU OLUSTURMA ====================
def generate_clean_sample(img_size=96, max_digits=3):
    bg_type = random.choice(['white', 'light_gray', 'gradient'])
    
    if bg_type == 'white':
        img = np.full((img_size, img_size), random.randint(230, 255), dtype=np.uint8)
    elif bg_type == 'light_gray':
        img = np.full((img_size, img_size), random.randint(200, 240), dtype=np.uint8)
    else:
        start, end = random.randint(220, 255), random.randint(200, 230)
        gradient = np.linspace(start, end, img_size).astype(np.uint8)
        img = np.tile(gradient, (img_size, 1)).astype(np.uint8)
    
    labels = []
    placed_boxes = []
    
    for _ in range(random.randint(1, max_digits)):
        digit_class = random.randint(0, 9)
        if len(svhn_by_class[digit_class]) == 0:
            continue
        
        idx = random.randint(0, len(svhn_by_class[digit_class]) - 1)
        digit = svhn_by_class[digit_class][idx].copy()
        
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            h, w = digit.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            digit = cv2.warpAffine(digit, M, (w, h), borderValue=255)
        
        new_size = random.randint(18, 38)
        digit_resized = cv2.resize(digit, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        margin = 3
        max_x, max_y = img_size - new_size - margin, img_size - new_size - margin
        if max_x <= margin or max_y <= margin:
            continue
        
        for _ in range(20):
            x, y = random.randint(margin, max_x), random.randint(margin, max_y)
            new_box = (x-2, y-2, x+new_size+2, y+new_size+2)
            overlap = any(not (new_box[2]<b[0] or new_box[0]>b[2] or new_box[3]<b[1] or new_box[1]>b[3]) for b in placed_boxes)
            if not overlap:
                break
        else:
            continue
        
        placed_boxes.append(new_box)
        
        digit_norm = digit_resized.astype(np.float32)
        d_min, d_max = digit_norm.min(), digit_norm.max()
        if d_max > d_min:
            digit_norm = (digit_norm - d_min) / (d_max - d_min)
        alpha = np.clip((1.0 - digit_norm) * 1.5, 0, 1)
        ink = random.randint(10, 60)
        
        roi = img[y:y+new_size, x:x+new_size].astype(np.float32)
        blended = roi * (1 - alpha) + ink * alpha
        img[y:y+new_size, x:x+new_size] = np.clip(blended, 0, 255).astype(np.uint8)
        
        labels.append([digit_class, (x+new_size/2)/img_size, (y+new_size/2)/img_size, new_size/img_size, new_size/img_size])
    
    if random.random() < 0.2:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img, labels

def labels_to_yolo_output(labels, grid_size=6, num_classes=10):
    output = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=np.float32)
    for label in labels:
        class_id, x_center, y_center, width, height = label
        grid_x = min(int(x_center * grid_size), grid_size - 1)
        grid_y = min(int(y_center * grid_size), grid_size - 1)
        x_offset = x_center * grid_size - grid_x
        y_offset = y_center * grid_size - grid_y
        if output[grid_y, grid_x, 4] == 0:
            output[grid_y, grid_x, 0] = x_offset
            output[grid_y, grid_x, 1] = y_offset
            output[grid_y, grid_x, 2] = width
            output[grid_y, grid_x, 3] = height
            output[grid_y, grid_x, 4] = 1.0
            output[grid_y, grid_x, 5 + int(class_id)] = 1.0
    return output

print('Goruntu olusturma tanimlandi.')

# ==================== ORNEK GOSTER ====================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax in axes.flat:
    img, labels = generate_clean_sample(max_digits=2)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Digits: {[int(l[0]) for l in labels]}')
    ax.axis('off')
plt.suptitle('Clean SVHN Training Images')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
plt.show()

# ==================== VERI SETI OLUSTUR ====================
print('\nEgitim verisi olusturuluyor...')
X_train, y_train = [], []
for i in range(TRAIN_SAMPLES):
    if i % 5000 == 0:
        print(f'  {i}/{TRAIN_SAMPLES} ({100*i/TRAIN_SAMPLES:.0f}%)')
    img, labels = generate_clean_sample(IMG_SIZE, max_digits=3)
    X_train.append(img)
    y_train.append(labels_to_yolo_output(labels, GRID_SIZE, NUM_CLASSES))

X_train = np.array(X_train, dtype=np.float32) / 255.0
y_train = np.array(y_train, dtype=np.float32)
X_train = np.stack([X_train, X_train, X_train], axis=-1)

print('Validation verisi olusturuluyor...')
X_val, y_val = [], []
for i in range(VAL_SAMPLES):
    if i % 1000 == 0:
        print(f'  {i}/{VAL_SAMPLES}')
    img, labels = generate_clean_sample(IMG_SIZE, max_digits=3)
    X_val.append(img)
    y_val.append(labels_to_yolo_output(labels, GRID_SIZE, NUM_CLASSES))

X_val = np.array(X_val, dtype=np.float32) / 255.0
X_val = np.stack([X_val, X_val, X_val], axis=-1)
y_val = np.array(y_val, dtype=np.float32)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')

# ==================== MODEL ====================
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def depthwise_separable_conv(x, filters, kernel_size=3, strides=1, name_prefix=''):
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}_dw')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_dw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name_prefix}_dw_relu')(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False, name=f'{name_prefix}_pw')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_pw_bn')(x)
    x = layers.ReLU(6.0, name=f'{name_prefix}_pw_relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, strides=1, name_prefix=''):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=f'{name_prefix}_conv')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn')(x)
    x = layers.ReLU(6.0, name=f'{name_prefix}_relu')(x)
    return x

class YOLOOutputLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        bbox_conf = tf.sigmoid(x[..., :5])
        classes = tf.nn.softmax(x[..., 5:])
        return tf.concat([bbox_conf, classes], axis=-1)
    
    def get_config(self):
        return super().get_config()

def create_micro_yolo_v3():
    inputs = keras.Input(shape=(96, 96, 3), name='input')
    x = conv_block(inputs, 16, 3, strides=2, name_prefix='stem')
    x = depthwise_separable_conv(x, 32, name_prefix='stage1')
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_1')
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_2')
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_1')
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_2')
    x = layers.Dropout(0.15)(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    x = depthwise_separable_conv(x, 128, name_prefix='head1')
    x = depthwise_separable_conv(x, 128, name_prefix='head2')
    x = conv_block(x, 64, kernel_size=1, name_prefix='head3')
    x = layers.Conv2D(15, 1, padding='same', name='output_conv')(x)
    outputs = YOLOOutputLayer(name='yolo_output')(x)
    return Model(inputs, outputs, name='MicroYOLO_v3')

print('\nModel olusturuluyor...')
model = create_micro_yolo_v3()
print(f'Toplam parametre: {model.count_params():,}')

# ==================== LOSS ====================
def yolo_loss_v3(y_true, y_pred, label_smoothing=0.05):
    obj_mask = y_true[..., 4:5]
    noobj_mask = 1.0 - obj_mask
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    xy_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., :2] - y_pred[..., :2]))
    wh_true = tf.sqrt(tf.abs(y_true[..., 2:4]) + 1e-6)
    wh_pred = tf.sqrt(tf.abs(y_pred[..., 2:4]) + 1e-6)
    wh_loss = tf.reduce_sum(obj_mask * tf.square(wh_true - wh_pred))
    
    conf_pred = tf.clip_by_value(y_pred[..., 4:5], 1e-7, 1.0 - 1e-7)
    conf_loss_obj = -tf.reduce_sum(obj_mask * tf.math.log(conf_pred))
    conf_loss_noobj = -tf.reduce_sum(noobj_mask * tf.math.log(1.0 - conf_pred))
    
    class_true = y_true[..., 5:]
    class_true = class_true * (1 - label_smoothing) + label_smoothing / 10
    class_pred = tf.clip_by_value(y_pred[..., 5:], 1e-7, 1.0 - 1e-7)
    class_loss = -tf.reduce_sum(obj_mask * class_true * tf.math.log(class_pred))
    
    total = lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + lambda_noobj * conf_loss_noobj + class_loss
    return total / tf.cast(tf.shape(y_true)[0], tf.float32)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=yolo_loss_v3)
print('Model derlendi.')

# ==================== EGITIM ====================
model_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v3_best.keras')
callbacks = [
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1),
]

print('\n' + '='*60)
print('EGITIM BASLIYOR')
print(f'Epoch: {EPOCHS}, Batch: {BATCH_SIZE}')
print('='*60)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=1)

print('\nEGITIM TAMAMLANDI!')

# ==================== INT8 QUANTIZATION ====================
print('\n=== INT8 Quantization ===')
model = keras.models.load_model(model_path,
    custom_objects={'yolo_loss_v3': yolo_loss_v3, 'YOLOOutputLayer': YOLOOutputLayer})

def representative_dataset():
    for _ in range(500):
        img, _ = generate_clean_sample(max_digits=1)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        yield [np.expand_dims(img_rgb, 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v3_int8.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model: {len(tflite_model) / 1024:.1f} KB')

# ==================== C HEADER OLUSTUR ====================
print('\n=== C Header Olusturuluyor ===')
def create_c_header(tflite_path, header_path):
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    with open(header_path, 'w') as f:
        f.write('// MicroYOLO v3 for ESP32-CAM - Kaggle Generated\n')
        f.write(f'// Model size: {len(model_data)} bytes\n\n')
        f.write('#ifndef MICRO_YOLO_MODEL_H\n#define MICRO_YOLO_MODEL_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write('#define MICRO_YOLO_INPUT_SIZE 96\n')
        f.write('#define MICRO_YOLO_GRID_SIZE 6\n')
        f.write('#define MICRO_YOLO_NUM_CLASSES 10\n\n')
        f.write(f'const float input_scale = {inp["quantization"][0]}f;\n')
        f.write(f'const int input_zero_point = {inp["quantization"][1]};\n')
        f.write(f'const float output_scale = {out["quantization"][0]}f;\n')
        f.write(f'const int output_zero_point = {out["quantization"][1]};\n\n')
        f.write('const char* digit_labels[] = {"0","1","2","3","4","5","6","7","8","9"};\n')
        f.write('const int NUM_CLASSES = 10;\n\n')
        f.write(f'const unsigned int digit_model_len = {len(model_data)};\n')
        f.write('alignas(8) const unsigned char digit_model[] = {\n')
        for i, byte in enumerate(model_data):
            if i % 16 == 0: f.write('  ')
            f.write(f'0x{byte:02x},')
            if i % 16 == 15: f.write('\n')
        f.write('\n};\n\n#endif\n')
    
    print(f'Header: {header_path} ({len(model_data)/1024:.1f} KB)')

header_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v3_model.h')
create_c_header(tflite_path, header_path)

print('\n' + '='*60)
print('TAMAMLANDI!')
print('='*60)
print(f'\nDosyalar {OUTPUT_DIR} klasorunde:')
print('  - micro_yolo_v3_best.keras')
print('  - micro_yolo_v3_int8.tflite')
print('  - micro_yolo_v3_model.h')
print('\nKaggle\'dan indirmek icin:')
print('  Sag tarafta "Output" sekmesine tikla')
print('  Dosyalari indir')
