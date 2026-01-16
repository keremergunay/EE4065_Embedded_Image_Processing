# Question 2: YOLO Digit Detection - Training Scripts

## Overview
Enhanced YOLO-Nano training for ESP32-CAM digit detection with:
- Roboflow real camera dataset support
- Improved synthetic MNIST generation
- Strong data augmentation (Albumentations)
- ReLU activation (TFLite Micro compatible)

## Quick Start (Kaggle/Colab)

### 1. Train Models
```python
# Upload train_yolo.py to Kaggle and run:
!python train_yolo.py
```

This will:
- Download Roboflow dataset automatically
- Train TWO models:
  - `yolo_nano_synthetic_int8.tflite` (backup - MNIST based)
  - `yolo_nano_roboflow_int8.tflite` (recommended - real camera data)

### 2. Download Models
After training, download from `./models/`:
- `yolo_nano_synthetic_int8.tflite`
- `yolo_nano_roboflow_int8.tflite`

### 3. Convert to ESP32 Header
```bash
# For Roboflow model (recommended):
python convert_q2_headers.py roboflow

# For Synthetic model (backup):
python convert_q2_headers.py synthetic
```

### 4. Upload to ESP32
Open `esp32_cam/digit_detection/digit_detection.ino` in Arduino IDE and upload.

## Model Details
- Input: 96x96 grayscale
- Output: 6x6 grid, 2 anchors, 10 classes
- Size: ~40-60 KB (INT8 quantized)
- Activation: ReLU (TFLite Micro compatible)

## Dataset
Roboflow: labeling-dpvzj/my-first-project-7nvw3 v1
