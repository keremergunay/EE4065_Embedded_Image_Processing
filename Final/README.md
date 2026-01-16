# EE4065 - Embedded Systems Applications Final Project
## Semester: Fall 2025-26

**Instructor:** Prof. Dr. Cem Ünsalan  
**Team Members:**
- **Kerem Ergünay**
- **Tarık Erdoğan**

---

## 📌 Project Overview
This repository contains the implementation of the final project for the EE4065 Embedded Systems Applications course. The project focuses on deploying advanced Computer Vision and Signal Processing algorithms on the **ESP32-CAM** platform, utilizing **TensorFlow Lite for Microcontrollers**.

The project is divided into several questions (Q1-Q6), covering topics from basic image processing to complex multi-model inference pipelines.

---

## 📁 Project Structure

### [Q1: Image Preprocessing (Thresholding)](./Q1_Thresholding)
Implementation of histogram-based image processing techniques on ESP32.
- **Features:** Grayscale conversion, Histogram calculation, Binary Thresholding.
- **Goal:** Prepare images for optimal OCR/digit recognition.

### [Q2: Handwritten Digit Detection (YOLO)](./Q2)
Real-time digit detection using a custom YOLOv5-nano model trained on both MNIST and custom Roboflow datasets.
- **Model:** YOLOv5-nano (INT8 Quantized).
- **Features:** Bounding box regression, Multi-digit detection, Web interface with threshold controls.

### [Q3: Signal Sampling & Resampling](./Q3_Sampling)
Investigation of sampling theorems and signal reconstruction on embedded systems.
- **Features:** Downsampling, Upsampling algorithms, Aliasing analysis.

### [Q4: Multi-Model Recognition](./Question4_MultiModel_Recognition)
A complex pipeline combining detection and recognition.
- **Pipeline:** Preprocessing -> Detection -> Classification.
- **Goal:** End-to-end system for reading and interpreting handwritten strings.

### [Q5a: FOMO (Faster Objects, More Objects)](./Q5a_Fomo)
Implementation of Edge Impulse's FOMO algorithm for ultra-fast object detection.
- **Speed:** ~50ms inference time (vs ~200ms YOLO).
- **Use Case:** High-speed counting and presence detection of digits.

---

## 🛠️ Hardware & Software

### Hardware
- **Main Board:** AI-Thinker ESP32-CAM
- **Camera:** OV2640 / GC2145
- **Processor:** ESP32-S (240MHz Dual Core)
- **Memory:** 4MB PSRAM (Critical for TFLite models)

### Software Stack
- **IDE:** Arduino IDE 2.0+ / VS Code
- **Framework:** ESP-IDF / Arduino Core for ESP32
- **ML Framework:** TensorFlow Lite for Microcontrollers
- **Tools:** Python (Model training), Edge Impulse (FOMO), Google Colab

---

## 🚀 How to Run

1. **Setup:** Install ESP32 board support and `TensorFlowLite_ESP32` library in Arduino IDE.
2. **Select Question:** Navigate to the specific folder (e.g., `Q2/esp32_cam/digit_detection`).
3. **Configure:** Update WiFi credentials and camera pin definitions in the `.ino` file.
4. **Upload:** Flash the code to ESP32-CAM (Select "Huge APP" partition scheme).
5. **Monitor:** Use Serial Monitor to get the IP address and access the Web Interface.

---


