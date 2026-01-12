#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MicroYOLO v2 Training Script - ESP32-CAM
CPU ile calisir, tahmini sure: 20-40 dakika
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TF uyarilarini azalt

import tensorflow as tf
import numpy as np
import cv2
import random
import warnings
warnings.filterwarnings('ignore')

print('='*50)
print('MicroYOLO v2 Training - ESP32-CAM')
print('='*50)
print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')
print()

# ==================== SABITLER ====================
IMG_SIZE = 96
GRID_SIZE = 6
NUM_CLASSES = 10
OUTPUT_PER_CELL = 5 + NUM_CLASSES

# CPU icin optimize edilmis ayarlar
BATCH_SIZE = 32
EPOCHS = 150          # CPU icin azaltildi
LEARNING_RATE = 0.001
TRAIN_SAMPLES = 15000  # CPU icin azaltildi
VAL_SAMPLES = 3000

print(f'Ayarlar: {EPOCHS} epoch, {TRAIN_SAMPLES} train, {VAL_SAMPLES} val')
print()

# ==================== MNIST YUKLE ====================
print('MNIST yukleniyor...')
from tensorflow.keras.datasets import mnist
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

mnist_by_class = {}
for i in range(10):
    mnist_by_class[i] = mnist_x_train[mnist_y_train == i]
print(f'MNIST yuklendi: {len(mnist_x_train)} ornek')

# ==================== DATA AUGMENTATION ====================
def random_perspective(img, strength=0.1):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    offset = int(w * strength)
    pts2 = np.float32([
        [random.randint(0, offset), random.randint(0, offset)],
        [w - random.randint(0, offset), random.randint(0, offset)],
        [random.randint(0, offset), h - random.randint(0, offset)],
        [w - random.randint(0, offset), h - random.randint(0, offset)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)

def random_rotation(img, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)

def random_blur(img):
    if random.random() < 0.3:
        kernel = random.choice([1, 3])
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img

def random_noise(img, strength=10):
    if random.random() < 0.4:
        noise = np.random.normal(0, strength, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def random_brightness_contrast(img):
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

def random_erode_dilate(img):
    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)
    return img

def augment_digit(digit_img):
    if random.random() < 0.3:
        digit_img = random_perspective(digit_img, 0.15)
    if random.random() < 0.5:
        digit_img = random_rotation(digit_img, 25)
    digit_img = random_erode_dilate(digit_img)
    return digit_img

# ==================== GORUNTU OLUSTURMA ====================
def generate_realistic_sample(img_size=96, max_digits=3):
    # Acik kagit arka plan
    bg_choices = [
        random.randint(220, 255),
        random.randint(200, 235),
        random.randint(190, 220),
    ]
    bg_color = random.choice(bg_choices)
    img = np.full((img_size, img_size), bg_color, dtype=np.uint8)
    
    if random.random() < 0.6:
        noise = np.random.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    labels = []
    num_digits = random.randint(1, max_digits)
    placed_boxes = []
    
    for _ in range(num_digits):
        digit_class = random.randint(0, 9)
        digit_idx = random.randint(0, len(mnist_by_class[digit_class]) - 1)
        digit = mnist_by_class[digit_class][digit_idx].copy()
        digit = augment_digit(digit)
        
        scale = random.uniform(0.25, 0.55)
        new_size = int(28 * scale * img_size / 28)
        new_size = max(12, min(new_size, img_size - 4))
        
        digit_resized = cv2.resize(digit, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        
        ink_colors = [random.randint(0, 40), random.randint(10, 50), random.randint(0, 30)]
        ink_color = random.choice(ink_colors)
        
        max_attempts = 20
        for attempt in range(max_attempts):
            x = random.randint(0, img_size - new_size)
            y = random.randint(0, img_size - new_size)
            new_box = (x, y, x + new_size, y + new_size)
            overlap = False
            for box in placed_boxes:
                if not (new_box[2] < box[0] or new_box[0] > box[2] or new_box[3] < box[1] or new_box[1] > box[3]):
                    overlap = True
                    break
            if not overlap:
                break
        else:
            continue
        
        placed_boxes.append(new_box)
        roi = img[y:y+new_size, x:x+new_size]
        mask = digit_resized.astype(np.float32) / 255.0
        blended = (roi * (1.0 - mask) + ink_color * mask).astype(np.uint8)
        img[y:y+new_size, x:x+new_size] = blended
        
        x_center = (x + new_size / 2) / img_size
        y_center = (y + new_size / 2) / img_size
        width = new_size / img_size
        height = new_size / img_size
        labels.append([digit_class, x_center, y_center, width, height])
    
    img = random_blur(img)
    img = random_noise(img, 8)
    img = random_brightness_contrast(img)
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

# ==================== VERI SETI OLUSTUR ====================
print('Egitim verisi olusturuluyor...')
X_train, y_train = [], []
for i in range(TRAIN_SAMPLES):
    if i % 3000 == 0:
        print(f'  {i}/{TRAIN_SAMPLES} ({100*i/TRAIN_SAMPLES:.0f}%)')
    img, labels = generate_realistic_sample(IMG_SIZE, max_digits=3)
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
    img, labels = generate_realistic_sample(IMG_SIZE, max_digits=3)
    X_val.append(img)
    y_val.append(labels_to_yolo_output(labels, GRID_SIZE, NUM_CLASSES))

X_val = np.array(X_val, dtype=np.float32) / 255.0
X_val = np.stack([X_val, X_val, X_val], axis=-1)
y_val = np.array(y_val, dtype=np.float32)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print()

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
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return super().get_config()

def create_micro_yolo_v2():
    inputs = keras.Input(shape=(96, 96, 3), name='input')
    x = conv_block(inputs, 16, 3, strides=2, name_prefix='stem')
    x = depthwise_separable_conv(x, 32, name_prefix='stage1')
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_1')
    x = depthwise_separable_conv(x, 64, name_prefix='stage2_2')
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_1')
    x = depthwise_separable_conv(x, 128, name_prefix='stage3_2')
    x = layers.MaxPooling2D(2, name='pool3')(x)
    x = depthwise_separable_conv(x, 128, name_prefix='head1')
    x = conv_block(x, 64, kernel_size=1, name_prefix='head2')
    x = layers.Conv2D(15, 1, padding='same', name='output_conv')(x)
    outputs = YOLOOutputLayer(name='yolo_output')(x)
    return Model(inputs, outputs, name='MicroYOLO_v2')

print('Model olusturuluyor...')
model = create_micro_yolo_v2()
print(f'Toplam parametre: {model.count_params():,}')

# ==================== LOSS ====================
def yolo_loss_v2(y_true, y_pred):
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
    class_pred = tf.clip_by_value(y_pred[..., 5:], 1e-7, 1.0 - 1e-7)
    class_loss = -tf.reduce_sum(obj_mask * class_true * tf.math.log(class_pred))
    
    total = lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + lambda_noobj * conf_loss_noobj + class_loss
    return total / tf.cast(tf.shape(y_true)[0], tf.float32)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=yolo_loss_v2)
print('Model derlendi.')
print()

# ==================== EGITIM ====================
callbacks = [
    ModelCheckpoint('micro_yolo_v2_best.keras', monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
]

print('='*50)
print('EGITIM BASLIYOR')
print(f'Epoch: {EPOCHS}, Batch: {BATCH_SIZE}')
print('='*50)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, verbose=1)

print()
print('='*50)
print('EGITIM TAMAMLANDI!')
print('='*50)

# ==================== FLOAT32 TEST ====================
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

print('\n=== FLOAT32 Model Test ===')
class_correct = {i: 0 for i in range(10)}
class_total = {i: 0 for i in range(10)}

for _ in range(500):
    img, labels = generate_realistic_sample(max_digits=1)
    if len(labels) == 0: continue
    true_class = int(labels[0][0])
    class_total[true_class] += 1
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    pred = model.predict(np.expand_dims(img_rgb, 0), verbose=0)[0]
    dets = decode_predictions(pred, 0.2)
    if len(dets) > 0 and int(dets[0][0]) == true_class:
        class_correct[true_class] += 1

print('FLOAT32 Sinif Basina Dogruluk:')
total_c, total_s = 0, 0
for i in range(10):
    if class_total[i] > 0:
        acc = 100*class_correct[i]/class_total[i]
        print(f'  Rakam {i}: {class_correct[i]}/{class_total[i]} = {acc:.1f}%')
        total_c += class_correct[i]
        total_s += class_total[i]
print(f'Toplam: {total_c}/{total_s} = {100*total_c/total_s:.1f}%')

# ==================== INT8 QUANTIZATION ====================
print('\n=== INT8 Quantization ===')
model = keras.models.load_model('micro_yolo_v2_best.keras',
                                 custom_objects={'yolo_loss_v2': yolo_loss_v2, 'YOLOOutputLayer': YOLOOutputLayer})

def representative_dataset_balanced():
    for digit in range(10):
        for _ in range(50):
            img, _ = generate_realistic_sample(max_digits=1)
            img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
            yield [np.expand_dims(img_rgb, 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_balanced
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open('micro_yolo_v2_int8.tflite', 'wb') as f:
    f.write(tflite_model)
print(f'TFLite model: {len(tflite_model) / 1024:.1f} KB')

# ==================== INT8 TEST ====================
print('\n=== INT8 Model Test ===')
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale = input_details[0]['quantization'][0]
input_zp = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zp = output_details[0]['quantization'][1]

print(f'Input scale={input_scale:.6f}, zp={input_zp}')
print(f'Output scale={output_scale:.6f}, zp={output_zp}')

int8_correct = {i: 0 for i in range(10)}
int8_total = {i: 0 for i in range(10)}

for _ in range(500):
    img, labels = generate_realistic_sample(max_digits=1)
    if len(labels) == 0: continue
    true_class = int(labels[0][0])
    int8_total[true_class] += 1
    
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    img_int8 = ((img_rgb / input_scale) + input_zp).astype(np.int8)
    input_data = img_int8.reshape(1, 96, 96, 3)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output_float = (output.astype(np.float32) - output_zp) * output_scale
    
    best_conf, best_cls = -1, -1
    for gy in range(6):
        for gx in range(6):
            conf = output_float[gy, gx, 4]
            if conf > best_conf:
                best_conf = conf
                best_cls = np.argmax(output_float[gy, gx, 5:15])
    
    if best_cls == true_class:
        int8_correct[true_class] += 1

print('INT8 Sinif Basina Dogruluk:')
total_c, total_s = 0, 0
for i in range(10):
    if int8_total[i] > 0:
        acc = 100*int8_correct[i]/int8_total[i]
        print(f'  Rakam {i}: {int8_correct[i]}/{int8_total[i]} = {acc:.1f}%')
        total_c += int8_correct[i]
        total_s += int8_total[i]
print(f'Toplam: {total_c}/{total_s} = {100*total_c/total_s:.1f}%')

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
        f.write('// MicroYOLO v2 for ESP32-CAM - Auto-generated\n')
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

create_c_header('micro_yolo_v2_int8.tflite', 'micro_yolo_v2_model.h')

print('\n' + '='*50)
print('TAMAMLANDI!')
print('='*50)
print('Dosyalar:')
print('  - micro_yolo_v2_best.keras (FLOAT32)')
print('  - micro_yolo_v2_int8.tflite (INT8)')
print('  - micro_yolo_model.h (ESP32 icin)')
print('\nmicro_yolo_model.h dosyasini ESP32 projesine kopyala!')
