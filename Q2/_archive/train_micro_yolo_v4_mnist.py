#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MicroYOLO v4 Training - MNIST FOCUSED (Maximum Accuracy)
ESP32-CAM için Digit Detection - El yazısı rakamlar için optimize edilmiş

Hedef: Beyaz kağıt üzerinde siyah el yazısı rakamlar
Veri Seti: %80 MNIST + %20 SVHN (çeşitlilik için)

Kaggle'da çalıştırmak için:
1. kaggle.com/notebooks > New Notebook
2. Settings > Accelerator > GPU P100
3. Settings > Internet > On
4. Bu kodu yapıştır ve Run All
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Zaten var
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('MicroYOLO v4 Training - MNIST FOCUSED')
print('El yazısı rakamlar için optimize edilmiş')
print('='*60)
print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')

OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== AYARLAR ====================
IMG_SIZE = 96
GRID_SIZE = 6
NUM_CLASSES = 10

BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.001
TRAIN_SAMPLES = 80000  # Artırıldı: 50K -> 80K
VAL_SAMPLES = 10000    # Artırıldı: 8K -> 10K
print(f'Ayarlar: {EPOCHS} epoch, {TRAIN_SAMPLES} train, {VAL_SAMPLES} val')
print('Veri Seti: %100 MNIST (el yazısı rakamlar)')

# ==================== MNIST YUKLE ====================
print('\nMNIST yukleniyor...')
from tensorflow.keras.datasets import mnist
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

# Tüm MNIST verisini kullan
mnist_x_all = np.concatenate([mnist_x_train, mnist_x_test])
mnist_y_all = np.concatenate([mnist_y_train, mnist_y_test])

mnist_by_class = {}
for i in range(10):
    mnist_by_class[i] = mnist_x_all[mnist_y_all == i]
    print(f'  MNIST Sinif {i}: {len(mnist_by_class[i])} ornek')
print(f'MNIST toplam: {len(mnist_x_all)} ornek')

# ==================== AUGMENTATION (GÜÇLÜ - Overfitting önlemek için) ====================
def random_rotation(img, max_angle=25):  # 20 -> 25
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

def random_scale(img, scale_range=(0.7, 1.3)):  # 0.8-1.2 -> 0.7-1.3
    scale = random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    result = np.full((h, w), 255, dtype=np.uint8)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    
    if scale > 1:
        crop_y = (new_h - h) // 2
        crop_x = (new_w - w) // 2
        result = resized[crop_y:crop_y+h, crop_x:crop_x+w]
    else:
        result[max(0,y_off):max(0,y_off)+new_h, max(0,x_off):max(0,x_off)+new_w] = resized
    
    return result

def random_translate(img, max_shift=6):  # 4 -> 6
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)

def random_noise(img, strength=15):  # 10 -> 15
    if random.random() < 0.5:  # 0.3 -> 0.5
        noise = np.random.normal(0, strength, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def random_blur(img):
    if random.random() < 0.35:  # 0.2 -> 0.35
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

def random_erode_dilate(img):
    if random.random() < 0.4:  # 0.3 -> 0.4
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)
    return img

def random_perspective(img, strength=0.1):
    """Hafif perspektif bozulması"""
    if random.random() < 0.3:
        h, w = img.shape[:2]
        offset = int(w * strength)
        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [w - random.randint(0, offset), random.randint(0, offset)],
            [random.randint(0, offset), h - random.randint(0, offset)],
            [w - random.randint(0, offset), h - random.randint(0, offset)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderValue=255)
    return img

# ==================== KAMERA-GERÇEKÇİ AUGMENTATION ====================
def add_shadow(img):
    """Rastgele gölge ekle"""
    if random.random() < 0.3:
        h, w = img.shape[:2]
        x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
        shadow = random.uniform(0.7, 0.9)
        img = img.copy()
        img[:, x1:x2] = (img[:, x1:x2] * shadow).astype(np.uint8)
    return img

def add_jpeg_artifacts(img):
    """JPEG sıkıştırma artifaktları"""
    if random.random() < 0.25:
        quality = random.randint(50, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return img

def add_brightness_variation(img):
    """Parlaklık değişimi (kamera aydınlatma farklılıkları)"""
    if random.random() < 0.4:
        factor = random.uniform(0.85, 1.15)
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def add_edge_glare(img):
    """Kenar parlaması (pencere/ışık yansıması)"""
    if random.random() < 0.2:
        h, w = img.shape[:2]
        # Bir kenardan parlama
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        gradient = np.ones((h, w), dtype=np.float32)
        size = random.randint(10, 30)
        
        if edge == 'top':
            gradient[:size, :] = np.tile(np.linspace(1.2, 1.0, size).reshape(-1, 1), (1, w))
        elif edge == 'bottom':
            gradient[-size:, :] = np.tile(np.linspace(1.0, 1.2, size).reshape(-1, 1), (1, w))
        elif edge == 'left':
            gradient[:, :size] = np.tile(np.linspace(1.2, 1.0, size), (h, 1))
        else:
            gradient[:, -size:] = np.tile(np.linspace(1.0, 1.2, size), (h, 1))
        
        return np.clip(img * gradient, 0, 255).astype(np.uint8)
    return img

def augment_mnist_digit(digit):
    """MNIST rakamını augment et - GÜÇLÜ"""
    digit = digit.copy()
    
    # Rotasyon - daha sık ve geniş açı
    if random.random() < 0.7:
        digit = random_rotation(digit, 20)
    
    # Scale - daha sık
    if random.random() < 0.5:
        digit = random_scale(digit, (0.75, 1.25))
    
    # Translate - daha sık
    if random.random() < 0.6:
        digit = random_translate(digit, 5)
    
    # Erode/Dilate
    digit = random_erode_dilate(digit)
    
    # Perspective
    digit = random_perspective(digit, 0.08)
    
    return digit

# ==================== GORUNTU OLUSTURMA ====================
def generate_training_sample(img_size=96, max_digits=3):
    """El yazısı rakamlar için optimize edilmiş görüntü oluştur - KAMERA GERÇEKÇİ"""
    
    # Arka plan türü seç
    bg_type = random.choice(['uniform', 'gradient_h', 'gradient_v', 'textured'])
    
    if bg_type == 'uniform':
        bg_val = random.randint(220, 255)
        img = np.full((img_size, img_size), bg_val, dtype=np.uint8)
    elif bg_type == 'gradient_h':
        start, end = random.randint(210, 250), random.randint(230, 255)
        gradient = np.linspace(start, end, img_size).astype(np.uint8)
        img = np.tile(gradient, (img_size, 1)).astype(np.uint8)
    elif bg_type == 'gradient_v':
        start, end = random.randint(210, 250), random.randint(230, 255)
        gradient = np.linspace(start, end, img_size).reshape(-1, 1).astype(np.uint8)
        img = np.tile(gradient, (1, img_size)).astype(np.uint8)
    else:  # textured
        bg_val = random.randint(220, 250)
        img = np.full((img_size, img_size), bg_val, dtype=np.uint8)
        noise = np.random.randint(-12, 12, (img_size, img_size), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 200, 255).astype(np.uint8)
    
    labels = []
    placed_boxes = []
    num_digits = random.randint(1, max_digits)
    
    for _ in range(num_digits):
        # MNIST digit seç
        digit_class = random.randint(0, 9)
        idx = random.randint(0, len(mnist_by_class[digit_class]) - 1)
        digit = mnist_by_class[digit_class][idx].copy()
        digit = augment_mnist_digit(digit)
        
        # Boyut: 22-55 piksel (kamera görüntülerine uygun - daha büyük rakamlar)
        new_size = random.randint(22, 55)
        digit_resized = cv2.resize(digit, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        # Pozisyon bul
        margin = 4
        max_x = img_size - new_size - margin
        max_y = img_size - new_size - margin
        if max_x <= margin or max_y <= margin:
            continue
        
        for _ in range(30):
            x = random.randint(margin, max_x)
            y = random.randint(margin, max_y)
            new_box = (x-3, y-3, x+new_size+3, y+new_size+3)
            
            overlap = False
            for box in placed_boxes:
                if not (new_box[2] < box[0] or new_box[0] > box[2] or 
                        new_box[3] < box[1] or new_box[1] > box[3]):
                    overlap = True
                    break
            if not overlap:
                break
        else:
            continue
        
        placed_boxes.append(new_box)
        
        # MNIST: pixel değeri = ink yoğunluğu (255 = tam siyah ink)
        mask = digit_resized.astype(np.float32) / 255.0
        
        # Hafif blur ile kenarları yumuşat
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Mürekkep rengi (siyah/koyu gri)
        ink = random.randint(0, 40)
        
        # Blend
        roi = img[y:y+new_size, x:x+new_size].astype(np.float32)
        blended = roi * (1 - mask) + ink * mask
        img[y:y+new_size, x:x+new_size] = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Label
        x_center = (x + new_size / 2) / img_size
        y_center = (y + new_size / 2) / img_size
        width = new_size / img_size
        height = new_size / img_size
        labels.append([digit_class, x_center, y_center, width, height])
    
    # Son augmentation - KAMERA GERÇEKÇİ
    img = random_blur(img)
    img = random_noise(img, 10)
    img = add_shadow(img)
    img = add_brightness_variation(img)
    img = add_edge_glare(img)
    img = add_jpeg_artifacts(img)
    
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

print('\nGoruntu olusturma tanimlandi.')

# ==================== ORNEK GOSTER ====================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax in axes.flat:
    img, labels = generate_training_sample(max_digits=2)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Digits: {[int(l[0]) for l in labels]}')
    ax.axis('off')
plt.suptitle('Training Samples - MNIST Focused (Handwritten)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
plt.show()

# ==================== VERI SETI OLUSTUR ====================
print('\nEgitim verisi olusturuluyor...')
X_train, y_train = [], []
for i in range(TRAIN_SAMPLES):
    if i % 5000 == 0:
        print(f'  {i}/{TRAIN_SAMPLES} ({100*i/TRAIN_SAMPLES:.0f}%)')
    img, labels = generate_training_sample(IMG_SIZE, max_digits=3)
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
    img, labels = generate_training_sample(IMG_SIZE, max_digits=3)
    X_val.append(img)
    y_val.append(labels_to_yolo_output(labels, GRID_SIZE, NUM_CLASSES))

X_val = np.array(X_val, dtype=np.float32) / 255.0
X_val = np.stack([X_val, X_val, X_val], axis=-1)
y_val = np.array(y_val, dtype=np.float32)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')

# ==================== MODEL (Biraz daha büyük) ====================
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

def create_micro_yolo_v4():
    """Overfitting önlenmiş model - daha az parametre, daha fazla regularization"""
    inputs = keras.Input(shape=(96, 96, 3), name='input')
    
    # Stem
    x = conv_block(inputs, 16, 3, strides=2, name_prefix='stem')  # 48x48, 24->16
    
    # Stage 1
    x = depthwise_separable_conv(x, 32, name_prefix='stage1_1')  # 48->32
    x = layers.MaxPooling2D(2, name='pool1')(x)  # 24x24
    x = layers.Dropout(0.15)(x)  # Yeni dropout
    
    # Stage 2
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_1')  # 96->64
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_2')
    x = layers.MaxPooling2D(2, name='pool2')(x)  # 12x12
    x = layers.Dropout(0.2)(x)  # Yeni dropout
    
    # Stage 3
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_1')  # 192->128
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_2')
    x = layers.Dropout(0.35)(x)  # 0.2->0.35
    x = layers.MaxPooling2D(2, name='pool3')(x)  # 6x6
    
    # Head
    x = depthwise_separable_conv(x, 128, name_prefix='head1')  # 192->128
    x = layers.Dropout(0.3)(x)  # Yeni dropout
    x = depthwise_separable_conv(x, 96, name_prefix='head2')  # 128->96
    x = conv_block(x, 48, kernel_size=1, name_prefix='head3')  # 64->48
    
    # Output
    x = layers.Conv2D(15, 1, padding='same', name='output_conv')(x)
    outputs = YOLOOutputLayer(name='yolo_output')(x)
    
    return Model(inputs, outputs, name='MicroYOLO_v4')

print('\nModel olusturuluyor...')
model = create_micro_yolo_v4()
print(f'Toplam parametre: {model.count_params():,}')
model.summary()

# ==================== LOSS (FOCAL LOSS EKLENDİ) ====================
def yolo_loss_v4(y_true, y_pred, label_smoothing=0.03, focal_gamma=2.0):
    """
    YOLO Loss with Focal Loss for hard example mining
    focal_gamma: 0 = normal CE, 2.0 = recommended for imbalanced data
    """
    obj_mask = y_true[..., 4:5]
    noobj_mask = 1.0 - obj_mask
    lambda_coord = 5.0
    lambda_noobj = 0.5
    
    # Coordinate loss (unchanged)
    xy_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., :2] - y_pred[..., :2]))
    wh_true = tf.sqrt(tf.abs(y_true[..., 2:4]) + 1e-6)
    wh_pred = tf.sqrt(tf.abs(y_pred[..., 2:4]) + 1e-6)
    wh_loss = tf.reduce_sum(obj_mask * tf.square(wh_true - wh_pred))
    
    # Confidence loss with Focal Loss
    conf_pred = tf.clip_by_value(y_pred[..., 4:5], 1e-7, 1.0 - 1e-7)
    
    # Focal weight: (1-p)^gamma for positive, p^gamma for negative
    focal_weight_obj = tf.pow(1.0 - conf_pred, focal_gamma)
    focal_weight_noobj = tf.pow(conf_pred, focal_gamma)
    
    conf_loss_obj = -tf.reduce_sum(obj_mask * focal_weight_obj * tf.math.log(conf_pred))
    conf_loss_noobj = -tf.reduce_sum(noobj_mask * focal_weight_noobj * tf.math.log(1.0 - conf_pred))
    
    # Class loss with label smoothing
    class_true = y_true[..., 5:]
    class_true = class_true * (1 - label_smoothing) + label_smoothing / 10
    class_pred = tf.clip_by_value(y_pred[..., 5:], 1e-7, 1.0 - 1e-7)
    
    # Focal class loss
    focal_class_weight = tf.pow(1.0 - class_pred, focal_gamma)
    class_loss = -tf.reduce_sum(obj_mask * class_true * focal_class_weight * tf.math.log(class_pred))
    
    total = lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + lambda_noobj * conf_loss_noobj + class_loss
    return total / tf.cast(tf.shape(y_true)[0], tf.float32)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=yolo_loss_v4)
print('Model derlendi.')

# ==================== EGITIM ====================
model_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v4_best.keras')
callbacks = [
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),  # 50->25
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),  # 20->10
]

print('\n' + '='*60)
print('EGITIM BASLIYOR - MNIST FOCUSED')
print(f'Epoch: {EPOCHS}, Batch: {BATCH_SIZE}')
print(f'Veri: {TRAIN_SAMPLES} train, {VAL_SAMPLES} val')
print('='*60)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=1)

print('\nEGITIM TAMAMLANDI!')

# ==================== TEST ====================
print('\n=== Test (FLOAT32) ===')
def decode_predictions(pred, conf_threshold=0.3):
    detections = []
    for gy in range(6):
        for gx in range(6):
            conf = pred[gy, gx, 4]
            if conf > conf_threshold:
                class_probs = pred[gy, gx, 5:]
                class_id = np.argmax(class_probs)
                final_conf = conf * class_probs[class_id]
                if final_conf > conf_threshold:
                    detections.append([class_id, final_conf])
    return detections

correct = 0
total = 0
for _ in range(500):
    img, labels = generate_training_sample(max_digits=1)
    if len(labels) == 0: continue
    true_class = int(labels[0][0])
    total += 1
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    pred = model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
    dets = decode_predictions(pred, 0.2)
    if len(dets) > 0 and int(dets[0][0]) == true_class:
        correct += 1

print(f'FLOAT32 Dogruluk: {correct}/{total} = {100*correct/total:.1f}%')

# ==================== INT8 QUANTIZATION ====================
print('\n=== INT8 Quantization ===')
model = keras.models.load_model(model_path,
    custom_objects={'yolo_loss_v4': yolo_loss_v4, 'YOLOOutputLayer': YOLOOutputLayer})

def representative_dataset():
    for _ in range(500):
        img, _ = generate_training_sample(max_digits=1)
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        yield [np.expand_dims(img_rgb, 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v4_int8.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model: {len(tflite_model) / 1024:.1f} KB')

# ==================== INT8 TEST ====================
print('\n=== INT8 Test ===')
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale = input_details[0]['quantization'][0]
input_zp = input_details[0]['quantization'][1]

int8_correct = 0
int8_total = 0
for _ in range(500):
    img, labels = generate_training_sample(max_digits=1)
    if len(labels) == 0: continue
    true_class = int(labels[0][0])
    int8_total += 1
    
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    img_int8 = ((img_rgb / input_scale) + input_zp).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], img_int8.reshape(1, 96, 96, 3))
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_scale = output_details[0]['quantization'][0]
    output_zp = output_details[0]['quantization'][1]
    output_float = (output.astype(np.float32) - output_zp) * output_scale
    
    best_conf, best_cls = -1, -1
    for gy in range(6):
        for gx in range(6):
            conf = output_float[gy, gx, 4]
            if conf > best_conf:
                best_conf = conf
                best_cls = np.argmax(output_float[gy, gx, 5:15])
    
    if best_cls == true_class:
        int8_correct += 1

print(f'INT8 Dogruluk: {int8_correct}/{int8_total} = {100*int8_correct/int8_total:.1f}%')

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
        f.write('// MicroYOLO v4 for ESP32-CAM - MNIST Focused (Handwritten Digits)\n')
        f.write(f'// Model size: {len(model_data)} bytes\n')
        f.write(f'// Training: {TRAIN_SAMPLES} samples, {EPOCHS} epochs\n')
        f.write('// Dataset: 100% MNIST (Handwritten Digits)\n\n')
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

header_path = os.path.join(OUTPUT_DIR, 'micro_yolo_v4_model.h')
create_c_header(tflite_path, header_path)

# ==================== EGITIM GRAFIGI ====================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.subplot(1, 2, 2)
if len(history.history['val_loss']) > 50:
    plt.plot(history.history['val_loss'][-50:])
    plt.xlabel('Epoch (last 50)')
else:
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
plt.ylabel('Val Loss')
plt.title('Validation Loss')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
plt.show()

print('\n' + '='*60)
print('TAMAMLANDI!')
print('='*60)
print(f'\nDosyalar {OUTPUT_DIR} klasorunde:')
print('  - micro_yolo_v4_best.keras (FLOAT32)')
print('  - micro_yolo_v4_int8.tflite (INT8)')
print('  - micro_yolo_v4_model.h (ESP32 icin)')
print('\nESP32 icin:')
print('  1. micro_yolo_v4_model.h dosyasini indir')
print('  2. digit_detection klasorune kopyala')
print('  3. #include "micro_yolo_v4_model.h"')
